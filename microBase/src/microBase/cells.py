"""Single-cell cropping from whole-image volumes + masks.

Pure functions. The shared seam used by:
- microBase.ImageDataset.get_cropped_cell / get_cropped_cells
- microModel.WholeImageCellDataset (training/inference cropping)
- microModel.vis_interactive (click-to-view-image)
- microVis.ObjectExportWorker (object export to disk)

All functions accept (H, W, C) volume and (H, W) integer mask.
Returns crops in (H, W, C) layout, with background pixels zeroed.
"""

import numpy as np

from .errors import DataError


def get_labels(mask):
    """Return sorted unique non-zero labels in mask (the cell labels)."""
    ids = np.unique(mask)
    return sorted(int(i) for i in ids if i > 0)


def crop_cell(volume, mask, label, padding=4):
    """Crop a single cell from a multi-channel volume using its mask label.

    Args:
        volume: (H, W, C) array, raw intensity values.
        mask:   (H, W) integer array, label = label at this cell's pixels.
        label: integer label of the cell to crop.
        padding: pixels of padding around the bounding box. Clamped to image bounds.

    Returns:
        (crop, cell_mask, bbox) where:
          crop      : (h, w, C) array, background zeroed (mask != label -> 0)
          cell_mask : (h, w) bool array, True at this cell's pixels
          bbox      : (x, y, w, h) tuple — the padded bbox used for cropping
        Raises DataError for shape/label mistakes or a label with zero pixels.
    """
    if volume.ndim != 3:
        raise DataError(
            f"crop_cell expects (H, W, C) volume, got shape {volume.shape}")

    if label < 1:
        # Label 0 is background — np.where(mask == 0) would match the whole
        # image and the zero-pixel guard below would never fire.
        raise DataError(f"label must be a positive integer, got {label}.")

    if mask.shape != volume.shape[:2]:
        # A mismatched mask would broadcast-error below; raise with the
        # project's standard clear message instead.
        raise DataError(
            f"mask shape {mask.shape} does not match volume {volume.shape[:2]}."
        )
    padding = max(0, int(padding))

    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        raise DataError(
            f"label {label} has zero pixels in mask — possible corruption.")

    h_full, w_full = mask.shape
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(h_full, int(ys.max()) + padding + 1)
    x_min = max(0, int(xs.min()) - padding)
    x_max = min(w_full, int(xs.max()) + padding + 1)

    crop = volume[y_min:y_max, x_min:x_max, :].copy()
    mask_crop = mask[y_min:y_max, x_min:x_max]
    fg = mask_crop == label
    # Zero background pixels in every channel
    crop[~fg] = 0
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    return crop, fg, bbox


def crop_all_cells(volume, mask, padding=4, labels=None):
    """Crop every cell in mask. Convenience wrapper around crop_cell.

    Args:
        volume: (H, W, C) array.
        mask:   (H, W) integer array.
        padding: pixels of padding.
        labels: optional iterable of labels to crop. If None, crops all.

    Returns:
        list of (crop, cell_mask, bbox, label) tuples.
    """
    if labels is None:
        labels = get_labels(mask)
    results = []
    for cid in labels:
        crop, cell_mask, bbox = crop_cell(volume, mask, cid, padding=padding)
        results.append((crop, cell_mask, bbox, cid))
    return results


def edge_pixel_ratio(mask, edge_tolerance: int = 2):
    """Per-object edge-pixel / perimeter-pixel ratio for a (H, W) label mask.

    For each label:
    - Perimeter pixels = the object's inner outline (pixels with at least one
      4-neighbor of a different label; out-of-bounds counts as different, so
      image-border pixels are perimeter).
    - Edge pixels = perimeter pixels within `edge_tolerance` of any image edge.
    - ratio = edge_pixels / perimeter_pixels (0.0 to 1.0).

    An object half-clipped by the image border scores ~0.5; a fully interior
    object scores 0.0 — a high ratio therefore flags objects that are mostly
    outside the frame (segmentation filters them out; profiling reports them).

    Returns {label: ratio} for every non-zero label in the mask. Vectorized
    (whole-mask outline + bincount — identical values to a per-label
    scipy.ndimage.binary_erosion formulation, without needing scipy).
    """
    mask = np.asarray(mask)
    H, W = mask.shape

    # Inner outline of every object at once: a labeled pixel is perimeter iff
    # any 4-neighbor differs (out-of-bounds neighbors read as 0).
    left = np.zeros_like(mask); left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask); right[:, :-1] = mask[:, 1:]
    up = np.zeros_like(mask); up[1:, :] = mask[:-1, :]
    down = np.zeros_like(mask); down[:-1, :] = mask[1:, :]
    perimeter = (mask != 0) & ((mask != left) | (mask != right)
                               | (mask != up) | (mask != down))

    # Band of pixels within edge_tolerance of any image edge.
    n = edge_tolerance + 1
    edge_band = np.zeros((H, W), dtype=bool)
    edge_band[:n, :] = True
    edge_band[-n:, :] = True
    edge_band[:, :n] = True
    edge_band[:, -n:] = True

    labels = np.unique(mask[perimeter])
    labels = labels[labels > 0]
    result = {int(lbl): 0.0 for lbl in np.unique(mask) if lbl > 0}
    if labels.size == 0:
        return result

    lab = mask[perimeter]
    perim_counts = np.bincount(lab)
    edge_counts = np.bincount(lab, weights=edge_band[perimeter].astype(np.float64))
    for lbl in labels:
        total = int(perim_counts[lbl])
        result[int(lbl)] = float(edge_counts[lbl]) / total if total else 0.0
    return result
