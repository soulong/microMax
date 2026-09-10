"""Normalize module: per-channel / global / null z-score with percentile clip.

Independent of dataset and augmentation. Accepts (H, W, C) float arrays
(usually after augmentation, before tensorization). Returns float32 (H, W, C).

Clipping is per-channel and always applied: each channel is percentile-clipped
and min-max rescaled to [0, 1] ((clipped - lo) / (hi - lo)). The z-score then
runs on the rescaled values: per-channel for method="per_channel", pooled
across channels for method="global"; method="null" returns the rescaled array
as-is.

If a mask is provided, statistics are computed only over foreground pixels.
"""

import numpy as np

from .errors import ConfigError, DataError


def _clip_channel(channel, clip_low, clip_high, mask=None):
    """Clip a single 2D channel by percentile and min-max rescale to [0, 1].

    Returns float32 array: (clip(channel, lo, hi) - lo) / (hi - lo), or zeros
    for degenerate channels (empty region or hi <= lo).
    If mask is given, percentiles are computed over foreground pixels only.
    """
    region = _get_region(channel, mask)
    if region.size == 0:
        return np.zeros_like(channel, dtype=np.float32)
    lo = np.percentile(region, clip_low) if clip_low > 0 else region.min()
    hi = np.percentile(region, clip_high) if clip_high < 100 else region.max()
    if hi <= lo:
        # Degenerate: constant or near-constant channel. Avoid div-by-zero.
        return np.zeros_like(channel, dtype=np.float32)
    clipped = np.clip(channel, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _get_region(channel, mask):
    """Return channel values where mask is truthy, or full channel if mask None."""
    if mask is None:
        return channel
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return channel[mask]


def normalize(image, mask=None, method="per_channel",
              clip_low=0.05, clip_high=99.95):
    """Normalize a (H, W, C) array via z-score after percentile clip + rescale.

    Args:
        image: (H, W, C) array, any numeric dtype.
        mask:  optional (H, W) bool/int array. If given, stats computed only
               on foreground (mask truthy). Background is zeroed after norm.
        method: "per_channel" — independent z-score per channel
                "global"     — single z-score across all channels
                "null" / None — no z-score (still applies clip + rescale)
        clip_low, clip_high: percentile bounds in [0, 100].

    Returns:
        (H, W, C) float32 array. Background (mask falsy) is set to 0.
    """
    if image.ndim != 3:
        raise DataError(
            f"normalize expects (H, W, C) array, got shape {image.shape}"
        )

    img = image.astype(np.float32, copy=True)
    h, w, c = img.shape

    # Apply per-channel percentile clipping first
    clipped = np.empty_like(img)
    for ch in range(c):
        clipped[:, :, ch] = _clip_channel(img[:, :, ch], clip_low, clip_high, mask)

    # Background zeroing helper
    def _apply_mask(arr):
        if mask is None:
            return arr
        m = mask.astype(bool) if mask.dtype != bool else mask
        arr[~m] = 0
        return arr

    if method in ("null", None):
        return _apply_mask(clipped)

    if method == "per_channel":
        out = np.empty_like(clipped)
        for ch in range(c):
            region = _get_region(clipped[:, :, ch], mask)
            if region.size == 0:
                out[:, :, ch] = 0
                continue
            mean = region.mean()
            std = region.std()
            if std < 1e-6:
                out[:, :, ch] = 0
            else:
                out[:, :, ch] = (clipped[:, :, ch] - mean) / std
        return _apply_mask(out)

    if method == "global":
        region = _get_region(clipped, mask)
        if region.size == 0:
            return np.zeros_like(clipped)
        mean = region.mean()
        std = region.std()
        if std < 1e-6:
            return np.zeros_like(clipped)
        out = (clipped - mean) / std
        return _apply_mask(out)

    raise ConfigError(
        f"unknown normalize method '{method}'. "
        f"Use 'per_channel', 'global', or 'null'."
    )
