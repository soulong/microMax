"""Image I/O: TIFF and mask readers/writers.

Three channel layouts supported:
- None  : file is single-channel (one channel per file)
- "CHW" : file is multi-channel, pages/axes interpreted as (C, H, W)
- "HWC" : file is multi-channel, axes interpreted as (H, W, C)

All readers return arrays in (H, W, C) layout to callers.
"""

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import TiffFile, imwrite


def _coerce_path(path):
    return Path(path) if not isinstance(path, Path) else path


def compile_pattern(pattern, name="image_pattern"):
    """Compile a regex pattern, hard-exiting on invalid regex.

    None passes through; compiled `re.Pattern` instances are returned as-is.
    """
    if pattern is None:
        return None
    if isinstance(pattern, re.Pattern):
        return pattern
    try:
        return re.compile(pattern)
    except re.error as e:
        print(f"Error: invalid regex for {name}: {e}", file=sys.stderr)
        sys.exit(1)


def detect_tiff_properties(path, channel_layout):
    """Read a TIFF's shape, dtype, and channel count.

    Returns (img_shape_2d, dtype, n_channels). Used by both ImageDataset and
    CellDataset to auto-detect image properties from the first file.
    """
    try:
        with TiffFile(path) as tif:
            arr = tif.asarray()
    except Exception as e:
        print(
            f"Error: failed to read first TIFF for shape detection {path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    img_shape, n_channels, arr = normalize_tiff_array(arr, channel_layout, path)
    return img_shape, arr.dtype, n_channels


def normalize_tiff_array(arr, channel_layout, path=None):
    """Squeeze trailing singletons and validate array shape for the given layout.

    Returns (img_shape_2d, n_channels, squeezed_arr).
    Exits with error if the array shape doesn't match the expected layout.
    """
    where = f" at {path}" if path else ""

    if channel_layout is None:
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            print(
                f"Error: channel_layout=None expects 2D TIFF, got shape "
                f"{arr.shape}{where}",
                file=sys.stderr,
            )
            sys.exit(1)
        return arr.shape, 1, arr

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 3:
        print(
            f"Error: {channel_layout} TIFF expected 3D array, got shape "
            f"{arr.shape}{where}",
            file=sys.stderr,
        )
        sys.exit(1)

    if channel_layout == "CHW":
        return arr.shape[1:], arr.shape[0], arr
    else:  # HWC
        return arr.shape[:2], arr.shape[-1], arr


def read_tiff(path):
    """Read a single-channel image (TIFF/PNG/JPEG). Returns 2D array (H, W).

    Uses PIL for format-agnostic reading. Multi-channel files (RGB/RGBA) are
    reduced to their first channel. Callers wanting multi-channel data should
    use `read_tiff_channels` (TIFF-only).
    """
    path = _coerce_path(path)
    if not path.exists():
        print(f"Error: image file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with Image.open(path) as im:
            arr = np.array(im)
    except Exception as e:
        print(f"Error: failed to read image {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if arr.ndim == 3:
        # single-channel file but stored as (1, H, W) or (H, W, 1) — squeeze
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        else:
            # RGB/RGBA — take first channel
            arr = arr[:, :, 0]
    if arr.ndim != 2:
        print(
            f"Error: expected single-channel image, got shape {arr.shape} at {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return arr


def read_tiff_channels(path, channels, channel_layout="CHW"):
    """Read a multi-channel TIFF and return (H, W, C) array.

    channels: list of 1-based channel indices to extract (e.g. [1, 2, 3]).
    channel_layout:
        None  : file is 2D single-channel; `channels` must be exactly [1].
        "CHW" : pages/axes interpreted as (C, H, W).
        "HWC" : last axis is C, axes interpreted as (H, W, C).
    """
    path = _coerce_path(path)
    if not path.exists():
        print(f"Error: TIFF file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if channel_layout not in (None, "CHW", "HWC"):
        print(
            f"Error: channel_layout must be None, 'CHW' or 'HWC', got '{channel_layout}'",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with TiffFile(path) as tif:
            arr = tif.asarray()
    except Exception as e:
        print(f"Error: failed to read TIFF {path}: {e}", file=sys.stderr)
        sys.exit(1)

    _, n_channels, arr = normalize_tiff_array(arr, channel_layout, path)

    if channel_layout is None:
        if list(channels) != [1]:
            print(
                f"Error: channel_layout=None only supports channels=[1], got {channels} "
                f"at {path}",
                file=sys.stderr,
            )
            sys.exit(1)
        return arr[:, :, None]

    idx = [c - 1 for c in channels]
    for i in idx:
        if i < 0 or i >= n_channels:
            print(
                f"Error: channel index out of range (got {channels}, "
                f"file has {n_channels} channels) at {path}",
                file=sys.stderr,
            )
            sys.exit(1)
    if channel_layout == "CHW":
        return np.transpose(arr[idx], (1, 2, 0))
    else:  # HWC
        return arr[:, :, idx]


def read_mask(path):
    """Read a mask file (PNG or TIFF). Returns 2D integer array (H, W)."""
    path = _coerce_path(path)
    if not path.exists():
        print(f"Error: mask file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with Image.open(path) as im:
            arr = np.array(im)
    except Exception as e:
        print(f"Error: failed to read mask {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if arr.ndim == 3:
        # RGB mask — reduce to labels by taking first channel or luminance
        arr = arr[:, :, 0]
    return arr


def write_tiff(path, image, channel_layout="CHW"):
    """Write a TIFF file.

    image: (H, W) for single-channel, or (H, W, C) for multi-channel.
    channel_layout: how to store multi-channel — "CHW" (pages) or "HWC".
    """
    path = _coerce_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        imwrite(str(path), image)
        return
    if image.ndim != 3:
        print(
            f"Error: write_tiff expected 2D or 3D array, got shape {image.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    if channel_layout == "CHW":
        # transpose (H, W, C) -> (C, H, W) and write as multi-page
        out = np.transpose(image, (2, 0, 1))
        imwrite(str(path), out, photometric="minisblack")
    elif channel_layout == "HWC":
        imwrite(str(path), image, photometric="minisblack")
    else:
        print(
            f"Error: channel_layout must be 'CHW' or 'HWC', got '{channel_layout}'",
            file=sys.stderr,
        )
        sys.exit(1)


def write_mask(path, mask):
    """Write a 2D integer mask as PNG."""
    path = _coerce_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint16)).save(str(path))
