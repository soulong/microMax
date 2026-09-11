"""Shared microVis test fixtures.

Keeps per-user state (``~/.micromax``) out of the developer's home: every
test that opens a MainWindow persists a window size on close, which must
land in the test's tmp directory instead.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_user_defaults(tmp_path, monkeypatch):
    import microVis.user_defaults as ud

    monkeypatch.setattr(ud, "DEFAULTS_FILE", tmp_path / ".micromax")
