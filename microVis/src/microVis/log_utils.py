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
    microProfiler. microVis logs at INFO level only.

    Handlers sit on the ROOT logger so every package's logs (microVis,
    microBase merges, ...) reach the sinks. The file sink is attached per
    dataset via :func:`set_log_file` — each dataset directory keeps its
    own microVis.log.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(console_level)

    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is not None:
        console = logging.StreamHandler(stream)
        console.setLevel(console_level)
        console.setFormatter(fmt)
        root.addHandler(console)

    # Quiet noisy third-party libraries: fontTools logs EVERY step of its
    # PDF font subsetting (matplotlib exports with pdf.fonttype=42), which
    # would otherwise flood the console and the per-dataset log on each
    # "Export PDF". Their children (fontTools.subset, ...) inherit the level.
    for name in ("fontTools", "matplotlib.font_manager", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def set_log_file(path) -> None:
    """Log INFO records to ``path`` — one file per dataset directory.

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
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    _FILE_HANDLER = handler
    atexit.register(logging.shutdown)


def get_logger(name: str = "microVis") -> logging.Logger:
    return logging.getLogger(name)
