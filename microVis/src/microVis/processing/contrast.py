from __future__ import annotations

import numpy as np
from skimage import exposure


def apply_contrast(image: np.ndarray, method: str, **params) -> np.ndarray:
    """Apply contrast enhancement to a normalized [0,1] float image.

    Args:
        image: (H, W) float array in [0, 1].
        method: "none", "gamma", "histogram_equalization".
        **params: Method-specific parameters.

    Returns:
        (H, W) float array in [0, 1].
    """
    if method == "none":
        return image

    if method == "gamma":
        gamma = float(params.get("gamma", 1.0))
        return image ** gamma

    if method == "histogram_equalization":
        return exposure.equalize_hist(image)

    return image


def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert pixel values.

    For float images in [0, 1], returns 1 - image.
    For integer dtypes, maps min→max and max→min linearly.
    """
    if np.issubdtype(image.dtype, np.floating):
        return 1.0 - image
    return image.dtype.type(image.max()) - image + image.dtype.type(image.min())
