"""User-level GUI defaults shared by microProfiler and microVis.

The single config file lives at ``~/.micromax`` (a FILE, not a directory);
microVis owns the ``microvis`` section of it and microProfiler the
``microprofiler`` section, so both GUIs share one file while keeping their
own state. It is GUI-preferences only — the pipeline run config is a separate
YAML passed to the CLI.

Today microVis stores its main window size there. Updates are key-merged
(unknown future keys survive) and atomic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from microBase.config import load_yaml, save_yaml

logger = logging.getLogger("microVis.user_defaults")

# The shared GUI config file: ONE file for both desktop GUIs.
DEFAULTS_FILE = Path.home() / ".micromax"
# This app's top-level section inside the shared file.
_APP_SECTION = "microvis"


def get_user_defaults() -> dict:
    """This app's section of the shared ``~/.micromax`` config.

    Returns {} when the file is absent/unreadable or the section is missing.
    """
    if not DEFAULTS_FILE.exists():
        return {}
    try:
        data = load_yaml(DEFAULTS_FILE) or {}
    except Exception:
        logger.warning("Could not read %s", DEFAULTS_FILE, exc_info=True)
        return {}
    return data.get(_APP_SECTION) or {}


def update_user_defaults(section: str, updates: dict) -> None:
    """Deep-merge ``updates`` into ``microvis.<section>`` of ``~/.micromax``.

    Failures are logged, never raised — remembering preferences must not
    break a finished GUI session.
    """
    try:
        if DEFAULTS_FILE.exists():
            data = load_yaml(DEFAULTS_FILE) or {}
        else:
            data = {}
        app = dict(data.get(_APP_SECTION) or {})
        sect = dict(app.get(section) or {})
        sect.update(updates)
        app[section] = sect
        data[_APP_SECTION] = app
        save_yaml(DEFAULTS_FILE, data)
        logger.debug("user defaults updated: %s.%s -> %s", _APP_SECTION, section, updates)
    except Exception:
        logger.warning("Could not write %s", DEFAULTS_FILE, exc_info=True)
