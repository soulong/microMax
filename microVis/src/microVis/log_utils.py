"""Logger setup for microVis."""
from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path

_SETUP_DONE = False


def _ensure_std_streams() -> None:
    """Redirect sys.stdout/stderr to os.devnull when None (pythonw.exe).

    ``pythonw.exe`` launches the process without a console, leaving both
    streams as ``None``.  Libraries such as ``logging.StreamHandler``
    assume they are writable and crash with ``'NoneType' object has no
    attribute 'write'``.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


_FILE_HANDLER = None


def setup_logging(console_level: int = logging.INFO) -> None:
    """Configure console logging for the GUI (idempotent).

    Console (stdout, falling back to stderr): ``console_level`` (default
    INFO) with a compact ``[HH:MM:SS] LEVEL | message`` format, so a
    terminal launch shows what the app is doing — same style as
    microProfiler. ``--debug`` on the command line promotes it to DEBUG.

    Handlers sit on the ROOT logger so every package's logs (microVis,
    microBase merges, ...) reach the sinks. The file sink is attached per
    dataset via :func:`set_log_file` — each dataset directory keeps its
    own microVis.log.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    if "--debug" in sys.argv:
        console_level = logging.DEBUG

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is not None:
        console = logging.StreamHandler(stream)
        console.setLevel(console_level)
        console.setFormatter(fmt)
        root.addHandler(console)


def set_log_file(path) -> None:
    """Log everything (DEBUG) to ``path`` — one file per dataset directory.

    Called when a dataset finishes loading; a later dataset retargets the
    handler (the old file keeps the session's earlier records).
    """
    global _FILE_HANDLER
    root = logging.getLogger()
    if _FILE_HANDLER is not None:
        old = _FILE_HANDLER
        _FILE_HANDLER = None
        root.removeHandler(old)
        old.close()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(path), mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    _FILE_HANDLER = handler
    atexit.register(logging.shutdown)


def get_logger(name: str = "microVis") -> logging.Logger:
    return logging.getLogger(name)
