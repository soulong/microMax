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


def setup_logging(console_level: int = logging.INFO) -> None:
    """Configure logging for the GUI (idempotent).

    Two sinks:
    - console (stdout, falling back to stderr): ``console_level`` (default
      INFO) with a compact ``[HH:MM] LEVEL | message`` format, so a terminal
      launch shows what the app is doing — same style as microProfiler;
    - file: everything (DEBUG) appended to %TEMP%/microVis.log.

    ``--debug`` on the command line promotes the console to DEBUG.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    if "--debug" in sys.argv:
        console_level = logging.DEBUG

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    root = logging.getLogger("microVis")
    root.setLevel(logging.DEBUG)

    # ── Console: user-visible progress ──
    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is not None:
        console = logging.StreamHandler(stream)
        console.setLevel(console_level)
        console.setFormatter(fmt)
        root.addHandler(console)

    # ── File: full detail for post-mortem inspection ──
    log_dir = Path(
        os.environ.get("TEMP", os.environ.get("TMP", os.environ.get("TMPDIR", "/tmp")))
    )
    handler = logging.FileHandler(str(log_dir / "microVis.log"),
                                  mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)

    atexit.register(logging.shutdown)


def get_logger(name: str = "microVis") -> logging.Logger:
    return logging.getLogger(name)
