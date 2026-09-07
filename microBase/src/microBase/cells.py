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
import sys


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
        Exits with error if the cell has zero pixels in mask (corruption).
    """
    if volume.ndim != 3:
        print(
            f"Error: crop_cell expects (H, W, C) volume, got shape {volume.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    if label < 1:
        # Label 0 is background — np.where(mask == 0) would match the whole
        # image and the zero-pixel guard below would never fire.
        print(
            f"Error: label must be a positive integer, got {label}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if mask.shape != volume.shape[:2]:
        # A mismatched mask would broadcast-error below; exit with the
        # project's standard clear message instead.
        print(
            f"Error: mask shape {mask.shape} does not match volume "
            f"{volume.shape[:2]}.",
            file=sys.stderr,
        )
        sys.exit(1)
    padding = max(0, int(padding))

    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        print(
            f"Error: label {label} has zero pixels in mask — possible corruption.",
            file=sys.stderr,
        )
        sys.exit(1)

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
