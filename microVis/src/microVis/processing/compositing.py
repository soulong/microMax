from __future__ import annotations

import numpy as np

from microVis._settings import CHANNEL_COLORS, DEFAULT_CHANNEL_COLORS


def composite_image(
    image_data: np.ndarray,
    channel_names: list[str],
    channel_config: dict[str, dict],
    mask_data: np.ndarray | None = None,
    mask_colors: dict[int, tuple[float, float, float]] | None = None,
    mask_alpha: float = 0.3,
) -> np.ndarray:
    """Composite multi-channel image data into 8-bit RGB.

    Args:
        image_data: (H, W, C) float64 or uint16 array.
        channel_names: List of channel names matching image_data axis order.
        channel_config: {ch_name: {"enabled": bool, "color": str, "vmin": float, "vmax": float}}.
        mask_data: (H, W) integer label mask.
        mask_colors: {label: (R, G, B)} per-object colors. If None, generates random colors.
        mask_alpha: 0.0–1.0 fill opacity.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    h, w, _ = image_data.shape
    composite = np.zeros((h, w, 3), dtype=np.float64)

    for i, ch_name in enumerate(channel_names):
        cfg = channel_config.get(ch_name, {})
        if not cfg.get("enabled", True):
            continue

        color_val = cfg.get("color", _default_color(i))
        if isinstance(color_val, (list, tuple)):
            color = tuple(float(c) for c in color_val)
        else:
            color = CHANNEL_COLORS.get(color_val, (1.0, 1.0, 1.0))
        vmin = float(cfg.get("vmin", 0))
        vmax = float(cfg.get("vmax", 1))

        channel = image_data[:, :, i].astype(np.float64)
        if vmax > vmin:
            channel = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            channel = np.zeros_like(channel)

        for j in range(3):
            composite[:, :, j] += channel * color[j]

    composite = np.clip(composite, 0.0, 1.0)

    if mask_data is not None and mask_alpha > 0:
        labels = np.unique(mask_data)
        labels = labels[labels > 0]
        if mask_colors is None:
            rng = np.random.RandomState(42)
            mask_colors = {
                int(lbl): tuple(rng.rand(3).tolist()) for lbl in labels
            }
        for lbl in labels:
            color = mask_colors.get(int(lbl))
            if color is None:
                continue
            mask_binary = mask_data == lbl
            for j in range(3):
                composite[:, :, j] = np.where(
                    mask_binary,
                    composite[:, :, j] * (1 - mask_alpha) + color[j] * mask_alpha,
                    composite[:, :, j],
                )

    return (composite * 255).astype(np.uint8)


def _default_color(idx: int) -> str:
    default_order = DEFAULT_CHANNEL_COLORS
    if idx < len(default_order):
        return default_order[idx]
    return "white"
