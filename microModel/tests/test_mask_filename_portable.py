"""infer.db mask_filename follows the portable directory contract.

Written CWD-relative via canonical_directory (the SAME convention as the
`directory` column — masks may live in a different subtree than the images,
so the stored value keeps its path with the bare filename as the last
segment). Consumers resolve it through resolve_directory: process CWD
first, dataset-root fallback; legacy absolute values pass through.
"""

import json
import os

import numpy as np
import pytest
import tifffile
from PIL import Image

from microBase.db_contracts import canonical_directory

from microModel.reduction import _load_cell_image


@pytest.fixture()
def scene(tmp_path, monkeypatch):
    """A dataset dir as the process CWD: one 16x16 image + mask, object 1."""
    monkeypatch.chdir(tmp_path)
    img = np.zeros((16, 16, 1), dtype=np.uint16)
    img[4:10, 4:10] = 3000
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 4:10] = 1
    tifffile.imwrite(str(tmp_path / "img_ch1.tiff"), img)
    Image.fromarray(mask).save(str(tmp_path / "img_cp_masks_cell.png"))
    return tmp_path


def _view():
    return {
        "channels": None, "with_masking": True, "fixed_reference": False,
        "clip_low": 0.1, "clip_high": 99.9, "normalize_method": "per_channel",
        "max_value": 65535.0, "aug": None,
    }


def _row(scene, mask_filename):
    return {
        "directory": canonical_directory(str(scene)),
        "filename": json.dumps(["img_ch1.tiff"]),
        "mask_filename": mask_filename,
        "label": 1,
        "_root": str(scene),
    }


def test_relative_mask_filename_resolves_via_cwd(scene):
    """The new write form: canonical_directory → relative under the CWD."""
    d = _row(scene, canonical_directory(str(scene / "img_cp_masks_cell.png")))
    assert not os.path.isabs(d["mask_filename"])
    assert _load_cell_image(d, "whole_image", _view()) is not None


def test_relative_mask_resolves_via_root_when_cwd_elsewhere(scene, tmp_path,
                                                            monkeypatch):
    """Reading from a DIFFERENT CWD: the dataset-root fallback finds it."""
    d = _row(scene, canonical_directory(str(scene / "img_cp_masks_cell.png")))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert _load_cell_image(d, "whole_image", _view()) is not None


def test_legacy_absolute_mask_filename_still_works(scene):
    """Pre-convention DBs hold absolute mask paths — they pass through."""
    d = _row(scene, str(scene / "img_cp_masks_cell.png").replace("\\", "/"))
    assert os.path.isabs(d["mask_filename"])
    assert _load_cell_image(d, "whole_image", _view()) is not None
