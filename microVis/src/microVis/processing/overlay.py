from __future__ import annotations

import numpy as np
from skimage import measure


def extract_polygons(mask: np.ndarray, min_area: int = 5) -> list[tuple[int, np.ndarray]]:
    """Extract boundary polygons from a label mask.

    Pads each binary object with a 1-pixel False border before contour
    extraction so that edge-touching objects produce a single closed
    contour instead of multiple open fragments.

    Returns:
        List of (label_value, contour_xy) tuples where contour_xy is (N, 2)
        in (row, col) order. Contour coordinates are shifted back to the
        original mask coordinate system (may range from -0.5 to H-0.5).
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]
    polygons: list[tuple[int, np.ndarray]] = []
    for lbl in labels:
        binary = mask == lbl
        if np.sum(binary) < min_area:
            continue
        padded = np.pad(binary, 1, mode="constant", constant_values=False)
        contours = measure.find_contours(padded, level=0.5)
        for contour in contours:
            contour = contour - 1  # shift back to original coords
            polygons.append((int(lbl), contour))
    return polygons
