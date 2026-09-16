"""Feature extraction for the label app — cached per root, keyed by bundle.

The label app consumes the SAME cached teacher-branch features as
deduplication (SSL bundles) plus a classify-bundle variant that also
extracts per-class probabilities from the trained head. Cache layout and
invalidation are shared with deduplication: a cache file lives under
``<save_dir>/features/`` and is keyed by the bundle identity (path +
mtime + size), the complete file list, the channel/normalization settings
and the seed — adding files or changing the bundle invalidates it
automatically.
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from microBase import CellDataset, canonical_directory

from ..dataset import SingleCellDataset
from ..deduplication import _cache_path, _error
from ..infer import _forward_pass, _validate_channel_count
from ..utils import atomic_npz_save, load_npz_cache

logger = logging.getLogger(__name__)


def _extract_root_features_cls(entry, meta, model, device, output_dir,
                               bundle_id, seed, dl_cfg, only=None):
    """Cache-aware feature + probability extraction for a train bundle.

    Mirrors deduplication._extract_root_features (same cache layout, keyed by
    bundle identity — a bundle file is either SSL or classify, so the two
    never collide) but runs the classification model: row features come from
    model(x)'s second output, per-class probabilities from the first
    (sigmoid for multi-label bundles, softmax otherwise). `only` restricts
    the extraction to a subset of the root (the label file-list mode).

    Returns (paths, raw_paths, feats, probs): paths are PORTABLE
    (CWD-relative forward-slash under the CWD, absolute fallback — the
    canonical_directory convention the label DB itself stores), paths is
    the normcase identity, raw_paths keeps the real case.
    """
    root = entry["path"]
    cell_ds = CellDataset(root, channel_layout=entry["channel_layout"],
                          image_pattern=entry["image_pattern"])
    if len(cell_ds) == 0:
        _error(f"no TIFF files found in {root}")
    _validate_channel_count(len(cell_ds.intensity_colnames),
                            entry["channels"], meta, root)
    md = cell_ds.metadata
    if only is not None:
        want = set(only)
        md = md[[os.path.normcase(os.path.abspath(p)) in want
                 for p in md["path"]]].reset_index(drop=True)
        if len(md) == 0:
            _error(f"none of the file-list files exist in {root}")

    rel_paths = sorted(
        os.path.relpath(p, root).replace("\\", "/") for p in md["path"])
    cpath = _cache_path(output_dir, entry, bundle_id, rel_paths, None, seed)

    if os.path.exists(cpath):
        cached = load_npz_cache(cpath)
        if cached:
            paths = [str(p) for p in cached["paths"]]
            raw_paths = [str(p) for p in cached["paths_raw"]]
            feats = cached["feats"]
            probs = cached["probs"]
            logger.info("Feature cache hit: %s (%d cells, dim %d)",
                        os.path.basename(cpath), len(paths), feats.shape[1])
            return paths, raw_paths, feats, probs

    required_meta = ("augmentation_infer", "normalize_method",
                     "normalize_with_masking", "clip_low", "clip_high",
                     "normalize_fixed_reference")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        _error(f"bundle meta missing required keys: {missing}")
    dataset = SingleCellDataset(
        [(cell_ds, int(i)) for i in np.arange(len(md))],
        {"_": 0}, ["_"] * len(md),
        channels=entry["channels"],
        augmentation_spec=meta["augmentation_infer"],
        normalize_method=meta["normalize_method"],
        clip_low=meta["clip_low"], clip_high=meta["clip_high"],
        with_masking=meta["normalize_with_masking"],
        fixed_reference=bool(meta["normalize_fixed_reference"]),
        max_value=_resolve_max_value_entry_local(entry))

    num_workers = dl_cfg.get("num_workers", 4)
    loader_kwargs = dict(batch_size=dl_cfg.get("batch_size", 128),
                         shuffle=False,
                         num_workers=num_workers,
                         persistent_workers=dl_cfg.get(
                             "persistent_workers", True) and num_workers > 0)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_cfg.get("prefetch_factor", 2)
    loader = DataLoader(dataset, **loader_kwargs)

    logger.info("Extracting train-bundle features for %s (%d cells) ...",
                root, len(md))
    all_logits, all_feats = _forward_pass(loader, model, device, True, True,
                                          write_pred_class=True)
    feats = torch.cat(all_feats, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0)
    # Multi-label bundles train with BCE -> independent sigmoids; everything
    # else is a softmax distribution (same convention as infer).
    if meta.get("loss") == "bce":
        probs = torch.sigmoid(logits).numpy()
    else:
        probs = torch.softmax(logits, dim=1).numpy()

    raw_paths = [canonical_directory(
        os.path.abspath(md.iloc[int(i)]["path"])) for i in range(len(md))]
    paths = [os.path.normcase(p) for p in raw_paths]

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    atomic_npz_save(cpath, paths=np.array(paths),
                    paths_raw=np.array(raw_paths), feats=feats, probs=probs)
    logger.info("Cached features to %s", cpath)
    return paths, raw_paths, feats, probs


def _resolve_max_value_entry_local(entry):
    """Per-root max_value with the same validation as deduplication.

    (A tiny local copy so this module does not reach into deduplication's
    private config parser — the semantics are: required, numeric, > 0.)
    """
    mv = entry.get("max_value")
    if mv is None:
        _error(f"root {entry['path']}: max_value is required (e.g. 65535 "
               "for 16-bit, 255 for 8-bit)")
    try:
        mv = float(mv)
    except (TypeError, ValueError):
        _error(f"root {entry['path']}: max_value must be a number, got {mv!r}")
    if mv <= 0:
        _error(f"root {entry['path']}: max_value must be > 0, got {mv}")
    return mv
