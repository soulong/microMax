"""Logging configuration for microProfiler."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_LEVEL: int = logging.INFO

# The current per-dataset file sink (set by :func:`set_log_file`); tracked so
# a later dataset retargets the handler instead of stacking file handlers.
_FILE_HANDLER: logging.FileHandler | None = None


def _ensure_std_streams() -> None:
    """Redirect ``sys.stdout`` / ``sys.stderr`` to ``os.devnull`` when ``None``.

    ``pythonw.exe`` (used by the Windows shortcut) launches the process
    without a console, leaving both streams as ``None``.  Libraries such
    as ``tqdm`` and ``logging.StreamHandler`` assume they are writable
    and crash with ``'NoneType' object has no attribute 'write'``.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def set_default_logging_level(level: int) -> None:
    """Set the default logging level for subsequent setup_logging calls.

    When set, all future ``setup_logging()`` invocations (including those
    with ``clear_existing=False``) will use this level unless an explicit
    level is passed.
    """
    global _DEFAULT_LEVEL
    _DEFAULT_LEVEL = level


def setup_logging(
    name: str = "microProfiler",
    level: Optional[int] = None,
    log_file: Path | None = None,
    clear_existing: bool = True,
) -> logging.Logger:
    """Configure and return a logger instance."""
    if level is None:
        level = _DEFAULT_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if clear_existing and logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if clear_existing or not logger.hasHandlers():
        stream = sys.stdout if sys.stdout is not None else sys.stderr
        if stream is not None:
            console = logging.StreamHandler(stream)
            console.setLevel(level)
            console.setFormatter(fmt)
            logger.addHandler(console)

    if log_file:
        log_file = Path(log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: run_pipeline/run_step re-call setup_logging with
        # clear_existing=False; a second FileHandler for the same file would
        # duplicate every log record.
        if not any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_file
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    # Suppress noisy tifffile warnings (non-critical TIFF tag issues)
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    return logger


def set_log_file(path) -> None:
    """Attach/replace the per-dataset file sink (``<dataset>/microProfiler.log``).

    The terminal handler (and any explicit ``--log-file`` handler) stays
    attached; the previous dataset's file handler is removed — its file keeps
    the records already written. Called when a dataset is loaded and when a
    pipeline run targets a dataset directory.
    """
    global _FILE_HANDLER
    logger = logging.getLogger("microProfiler")

    path = Path(path).resolve()
    if _FILE_HANDLER is not None and Path(_FILE_HANDLER.baseFilename).resolve() == path:
        return
    if _FILE_HANDLER is not None:
        old = _FILE_HANDLER
        _FILE_HANDLER = None
        logger.removeHandler(old)
        old.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logger.level or _DEFAULT_LEVEL)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    _FILE_HANDLER = handler
