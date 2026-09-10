"""Image I/O: TIFF and mask readers.

Three channel layouts supported:
- None  : file is single-channel (one channel per file)
- "CHW" : file is multi-channel, pages/axes interpreted as (C, H, W)
- "HWC" : file is multi-channel, axes interpreted as (H, W, C)

All readers return arrays in (H, W, C) layout to callers.

Missing/unreadable files raise :class:`ImageReadError` (the pipeline
quarantines the broken row and continues); layout/shape/config mistakes raise
:class:`DatasetError` / :class:`ConfigError` so they are never mistaken for a
corrupt file. No reader calls ``sys.exit``.

Writing is intentionally NOT provided here: microProfiler writes with its
own zlib-compressed writer, and there is no other writer consumer.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import TiffFile

from microBase.errors import ConfigError, DatasetError, ImageReadError

__all__ = [
    "ImageReadError",
    "compile_pattern",
    "detect_tiff_properties",
    "normalize_tiff_array",
    "read_image",
    "read_tiff_channels",
    "read_mask",
]


def _coerce_path(path):
    return Path(path) if not isinstance(path, Path) else path


def _read_or_raise(path, what="image"):
    """Read a file with PIL; missing/unreadable files raise ImageReadError.

    Error messages keep the caller's terminology via ``what`` (e.g. "image",
    "mask").
    """
    path = _coerce_path(path)
    if not path.exists():
        raise ImageReadError(path, f"{what} file not found: {path}")
    try:
        with Image.open(path) as im:
            return np.array(im)
    except Exception as e:
        raise ImageReadError(path, f"failed to read {what} {path}: {e}") from e


def compile_pattern(pattern, name="image_pattern"):
    """Compile a regex pattern; invalid regex raises ConfigError.

    None passes through; compiled `re.Pattern` instances are returned as-is.
    """
    if pattern is None:
        return None
    if isinstance(pattern, re.Pattern):
        return pattern
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ConfigError(f"invalid regex for {name}: {e}") from e


def detect_tiff_properties(path, channel_layout):
    """Read a TIFF's shape, dtype, and channel count.

    Returns (img_shape_2d, n_channels, dtype) — the same position contract as
    normalize_tiff_array (shape, n_channels, extra). Used by both ImageDataset
    and CellDataset to auto-detect image properties from the first file.
    """
    try:
        with TiffFile(path) as tif:
            arr = tif.asarray()
    except Exception as e:
        raise ImageReadError(path, f"failed to read TIFF {path}: {e}") from e
    img_shape, n_channels, arr = normalize_tiff_array(arr, channel_layout, path)
    return img_shape, n_channels, arr.dtype


def normalize_tiff_array(arr, channel_layout, path=None):
    """Squeeze trailing singletons and validate array shape for the given layout.

    Returns (img_shape_2d, n_channels, squeezed_arr) — the canonical
    (shape, n_channels, extra) contract shared with detect_tiff_properties.
    A shape that does not match the layout raises DatasetError.
    """
    where = f" at {path}" if path else ""

    if channel_layout is None:
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise DatasetError(
                f"channel_layout=None expects 2D TIFF, got shape {arr.shape}{where}"
            )
        return arr.shape, 1, arr

    if channel_layout == "CHW":
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim != 3:
            raise DatasetError(
                f"{channel_layout} TIFF expected 3D array, got shape {arr.shape}{where}"
            )
        return arr.shape[1:], arr.shape[0], arr

    if channel_layout == "HWC":
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise DatasetError(
                f"{channel_layout} TIFF expected 3D array, got shape {arr.shape}{where}"
            )
        return arr.shape[:2], arr.shape[-1], arr

    raise ConfigError(
        f"channel_layout must be None, 'CHW' or 'HWC', got '{channel_layout}'"
    )


def read_image(path):
    """Read a single-channel image (TIFF/PNG/JPEG). Returns 2D array (H, W).

    Uses PIL for format-agnostic reading. Multi-channel PACKED files
    (RGB/RGBA, last axis 3-4) are reduced to their first channel. Any other
    3D shape (e.g. a (C, H, W) multi-page TIFF fed to the one-channel-per-file
    path) is a layout mistake and raises DatasetError instead of being
    silently mis-sliced. Callers wanting multi-channel data should use
    `read_tiff_channels` (TIFF-only).
    """
    arr = _read_or_raise(path, "image")
    if arr.ndim == 3:
        # single-channel file but stored as (1, H, W) or (H, W, 1) — squeeze
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] in (3, 4):
            # RGB/RGBA — take first channel
            arr = arr[:, :, 0]
        else:
            raise DatasetError(
                f"single-channel image reader got a {arr.shape} array at {path} "
                f"— this looks like a multi-channel (CHW?) file; use "
                f"read_tiff_channels / set channel_layout instead."
            )
    if arr.ndim != 2:
        raise DatasetError(
            f"expected single-channel image, got shape {arr.shape} at {path}"
        )
    return arr


def read_tiff_channels(path, channels, channel_layout="CHW"):
    """Read a multi-channel TIFF and return (H, W, C) array.

    channels: list of 1-based channel indices to extract (e.g. [1, 2, 3]).
    channel_layout:
        None  : file is 2D single-channel; `channels` must be exactly [1].
        "CHW" : pages/axes interpreted as (C, H, W).
        "HWC" : last axis is C, axes interpreted as (H, W, C).
    Missing/unreadable files raise ImageReadError; argument/layout mistakes
    raise ConfigError / DatasetError.
    """
    path = _coerce_path(path)
    if not path.exists():
        raise ImageReadError(path, f"TIFF file not found: {path}")
    if channel_layout not in (None, "CHW", "HWC"):
        raise ConfigError(
            f"channel_layout must be None, 'CHW' or 'HWC', got '{channel_layout}'"
        )
    if not channels:
        # At the library level an empty channel list is an error (the
        # pipeline is the only place where empty channels mean "skip").
        raise ConfigError(f"channels list must not be empty at {path}")
    try:
        with TiffFile(path) as tif:
            arr = tif.asarray()
    except Exception as e:
        raise ImageReadError(path, f"failed to read TIFF {path}: {e}") from e

    _, n_channels, arr = normalize_tiff_array(arr, channel_layout, path)

    if channel_layout is None:
        if list(channels) != [1]:
            raise ConfigError(
                f"channel_layout=None only supports channels=[1], got {channels} "
                f"at {path}"
            )
        return arr[:, :, None]

    idx = [c - 1 for c in channels]
    for i in idx:
        if i < 0 or i >= n_channels:
            raise DatasetError(
                f"channel index out of range (got {channels}, "
                f"file has {n_channels} channels) at {path}"
            )
    if channel_layout == "CHW":
        return np.transpose(arr[idx], (1, 2, 0))
    else:  # HWC
        return arr[:, :, idx]


def read_mask(path):
    """Read a mask file (PNG or TIFF). Returns 2D integer array (H, W).

    Missing/unreadable files raise ImageReadError.
    """
    arr = _read_or_raise(path, "mask")
    if arr.ndim == 3:
        # RGB mask — reduce to labels by taking first channel
        arr = arr[:, :, 0]
    return arr
