"""Scan the single-cell crops under D:/Model/sc_dataset/collection and
report every image whose edge_pixel_ratio exceeds 0.4 (microProfiler's
default segmentation filter threshold).

Algorithm: microBase.cells.edge_pixel_ratio (the exact function microProfiler
uses in segmentation filtering and object profiling), applied with its
default edge_tolerance=2. For a pre-cropped single cell the "mask" is the
non-zero foreground (crop_cell zeroes the background), so the ratio measures
how much of the cell's outline lies on the crop border — i.e. cells that
were clipped by their source field boundary.

Read-only: writes a CSV report, deletes nothing.
"""
import os
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r"C:\Users\haohe\GitHub\microMax\microBase\src")
import numpy as np
import tifffile
from microBase.cells import edge_pixel_ratio

ROOT = r"D:\Model\sc_dataset\collection"
# The four single-cell crop sets. Excluded: opencell/images (600x600
# whole-FOV fields), hpa/images (whole-field PNGs) and the *_cp_masks_cell.png
# field masks.
ROOTS = [
    os.path.join(ROOT, "NCOA2_293T_63x_epi"),
    os.path.join(ROOT, "hpa", "single_cell"),
    os.path.join(ROOT, "opencell", "single_cell"),
    os.path.join(ROOT, "p53"),
]
OUT_CSV = os.path.join(ROOT, "_to_delete_edge_ratio_gt_0.4.csv")
THRESHOLD = 0.42


def foreground(arr):
    """Non-zero foreground of a crop, normalized to a 2-D boolean mask."""
    fg = arr != 0
    if fg.ndim == 3:
        # (H, W, C) unless the last dim is clearly not channels
        if fg.shape[-1] <= 8 or fg.shape[0] > 8:
            fg = fg.any(axis=2)
        else:
            fg = fg.any(axis=0)
    return fg


def process(path):
    """One file -> (path, ratio or None, error or None)."""
    try:
        arr = tifffile.imread(path)
        fg = foreground(arr).astype(np.int32)
        if fg.sum() == 0:
            return (path, None, "empty crop (all zero)")
        ratio = edge_pixel_ratio(fg)[1]
        return (path, ratio, None)
    except Exception as e:  # report, never abort the sweep
        return (path, None, f"{type(e).__name__}: {e}")


def main():
    t0 = time.time()
    files = []
    for root in ROOTS:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith((".tiff", ".tif")):
                    files.append(os.path.join(dirpath, name))
    print(f"Found {len(files)} single-cell TIFFs "
          f"({time.time() - t0:.1f}s)", flush=True)

    hits = []       # (path, ratio) with ratio > THRESHOLD
    errors = []     # (path, message)
    n = 0
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        for path, ratio, err in ex.map(process, files, chunksize=64):
            n += 1
            if err is not None:
                errors.append((path, err))
            elif ratio is not None and ratio > THRESHOLD:
                hits.append((path, ratio))
            if n % 20000 == 0:
                print(f"  {n}/{len(files)} scanned, {len(hits)} over "
                      f"threshold ({time.time() - t0:.0f}s)", flush=True)

    hits.sort(key=lambda t: (t[0], -t[1]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "edge_pixel_ratio"])
        for path, ratio in hits:
            w.writerow([os.path.relpath(path, ROOT), f"{ratio:.4f}"])

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Scanned : {len(files)}")
    print(f"Errors  : {len(errors)}")
    print(f"> {THRESHOLD}: {len(hits)} files  -> {OUT_CSV}")
    for path, err in errors[:20]:
        print("  ERR", os.path.relpath(path, ROOT), err)


if __name__ == "__main__":
    main()
