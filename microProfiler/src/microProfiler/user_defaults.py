"""User-level defaults persisted under ``~/.micromax``.

Per-USER, cross-dataset state (``session.yml`` is the per-dataset layer).
Today: microProfiler's inference step remembers the last run's model
bundle, reducer pickle(s) and baseline cluster.pkl, so the next GUI start
pre-fills the Inference panel with them. The file is merged key-wise
(unknown future keys survive) and written atomically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from microBase.config import load_yaml, save_yaml

logger = logging.getLogger(__name__)

DEFAULTS_DIR = Path.home() / ".micromax"
DEFAULTS_FILE = DEFAULTS_DIR / "microprofiler.yml"


def get_user_defaults() -> dict:
    """The whole user-defaults file as a dict ({} when absent/unreadable)."""
    try:
        return load_yaml(DEFAULTS_FILE) or {}
    except Exception:
        logger.warning("Could not read %s", DEFAULTS_FILE, exc_info=True)
        return {}


def update_user_defaults(section: str, updates: dict) -> None:
    """Deep-merge ``updates`` into ``<section>`` of the user-defaults file.

    Stored values should be ABSOLUTE paths — the defaults must survive
    whatever dataset or CWD the next run starts from. A None value overwrites
    with null; to leave a key untouched simply omit it from ``updates``.
    Failures are logged, never raised — remembering preferences must not
    break a finished pipeline run.
    """
    try:
        data = get_user_defaults()
        sect = dict(data.get(section) or {})
        sect.update(updates)
        data[section] = sect
        DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_yaml(DEFAULTS_FILE, data)
        logger.debug("user defaults updated: %s -> %s", section, updates)
    except Exception:
        logger.warning("Could not write %s", DEFAULTS_FILE, exc_info=True)
