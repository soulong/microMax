"""Portable-path contract for the label project.

The label DB, its exports and the feature cache store PORTABLE paths —
CWD-relative forward-slash when the files live under the process CWD,
absolute fallback otherwise (the microBase canonical_directory convention
shared with deduplication's curated.csv). This test runs the real server
startup with the process CWD inside a temp project so every path takes the
portable branch, then checks registration, the sources table, the export
frame, image rendering and a restart.
"""

import os

import numpy as np
import tifffile

from microModel.label import LabelServer


def _make_server(tmp_path, monkeypatch):
    """Real startup (Flask run stubbed) with CWD = tmp_path, relative config."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cells"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.uniform(0, 65535, size=(16, 16)).astype(np.uint16)
        tifffile.imwrite(root / f"cell_{i:02d}.tif", arr)
    config = {
        "save_dir": "proj",
        "model": None,
        "data": {"file_dir": ["cells"], "channels": [1],
                 "channel_layout": None, "max_value": 65535},
    }
    srv = LabelServer(config, open_browser=False)
    srv.app.run = lambda **kw: None
    srv.start()
    return srv


def test_label_project_stores_portable_paths(tmp_path, monkeypatch):
    srv = _make_server(tmp_path, monkeypatch)
    assert len(srv.cells) == 3
    for cell in srv.cells:
        raw = cell["raw_path"]
        assert raw.startswith("cells/") and not os.path.isabs(raw)
        assert cell["filepath"] == os.path.normcase(raw)
        assert cell["source"] == "cells"
        assert os.path.exists(os.path.abspath(raw))      # resolves back
    # Source display settings sit under the same portable root key.
    assert srv.db.source_settings("cells") is not None
    # The export frame is a portable file list, curated.csv style.
    lb = srv.db.add_label("bright")
    srv.db.set_label(srv.cells[0]["cell_id"], lb["label_id"], state=1,
                     session_id=srv.session_id)
    df = srv.db.export_frame()
    assert list(df["filepath"]) == [srv.cells[0]["raw_path"]]
    # The image endpoint resolves the portable path back to the file.
    c = srv.app.test_client()
    r = c.get(f"/api/image?filepath={srv.cells[0]['filepath']}&max_px=64")
    assert r.status_code == 200 and r.mimetype == "image/png"


def test_label_restart_reuses_portable_cells(tmp_path, monkeypatch):
    srv = _make_server(tmp_path, monkeypatch)
    lb = srv.db.add_label("bright")
    srv.db.set_label(srv.cells[0]["cell_id"], lb["label_id"], state=1,
                     session_id=srv.session_id)
    # A second startup against the same portable paths inserts NO new cells
    # (INSERT OR IGNORE hits) and keeps the human decision.
    srv2 = _make_server(tmp_path, monkeypatch)
    assert len(srv2.cells) == 3
    assert srv2.db.stats()["labeled"] == 1
