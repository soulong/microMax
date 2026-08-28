"""Datasets backed by microBase.CellDataset / microBase.ImageDataset.

Pipeline: load (raw HWC) -> convert to float [0, 1] by data.max_value ->
extract mask (pixel != 0) -> augment (geometric + resize; mask co-transformed)
-> normalize (z-score with mask) -> CHW tensor.

Integer images (8/12/16-bit) are divided by max_value on load, so every
view pipeline and the fixed-reference stats share one [0, 1] domain (no
ToFloat steps in augmentation specs).

Normalization has two modes: per-view self-normalization (default) and
fixed-reference (normalize.fixed_reference: true), where clip bounds +
z-score stats are computed once on the RAW image and applied as a fixed
transform to every (augmented) view — preserving linear photometric
augmentation that self-normalization would cancel (affine-equivariance).

For SSL pretraining, SSLMultiViewDataset generates N augmented views per image
by building N separate augmentation pipelines (one per view spec).
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from microBase import (
    CellDataset,
    ImageDataset,
    build_pipeline,
    apply,
    normalize,
    read_mask,
)
from microBase.cells import get_labels


def _to_float_max(img_hwc, max_value):
    """Convert an integer image to float32 in [0, 1] by dividing by max_value.

    max_value=None leaves the image untouched (raw units). The domain of the
    fixed-reference stats and every augmented view is the pipeline's leading
    transform — with ToFloat removed from all specs, the conversion here IS
    the domain definition, so stats and views always match.
    """
    if max_value is None:
        return img_hwc
    return img_hwc.astype(np.float32) / float(max_value)


# ----------------------------------------------------------------------------
# Shared pipeline helper — used by all datasets.
# ----------------------------------------------------------------------------

def _compute_ref_stats(img_hwc, channels, with_masking, clip_low, clip_high,
                       method):
    """Compute FIXED clip + rescale + z-score statistics on the RAW image (pre-augment).

    Per-view self-normalization is affine-equivariant: any linear brightness/
    contrast augmentation is mathematically canceled by recomputing percentiles
    + mean/std on the augmented image. Computing the stats once on the raw cell
    and applying them as a fixed transform preserves photometric augmentation.

    Mirrors microBase.normalize exactly — per-channel percentile clip +
    min-max rescale to [0, 1], then z-score — so with an empty augmentation
    pipeline fixed_reference=True and per-view normalization are identical:
      - clip bounds are always per-channel (foreground percentiles;
        clip_low <= 0 -> min, clip_high >= 100 -> max)
      - "per_channel": per-channel mean/std of the rescaled foreground
      - "global": pooled mean/std over ALL channels' rescaled foreground
      - "null": mean=0, std=1 (rescale only, no z-score)
    Degenerate channels (empty region, hi <= lo) are rescaled to zeros,
    matching microBase._clip_channel; under "global" their zeros still
    contribute to the pooled stats, also matching microBase.
    """
    if channels is not None:
        ch_idx = [c - 1 for c in channels]
        img_hwc = img_hwc[:, :, ch_idx]
    mask_b = (img_hwc != 0).any(axis=2).astype(bool) if with_masking else None

    n_ch = img_hwc.shape[2]
    rescaled = np.empty_like(img_hwc, dtype=np.float32)
    ch_stats = []
    for c in range(n_ch):
        ch = img_hwc[:, :, c]
        region = ch[mask_b] if mask_b is not None else ch.reshape(-1)
        if region.size == 0:
            rescaled[:, :, c] = 0.0
            ch_stats.append({"lo": 0.0, "hi": 0.0, "mean": 0.0, "std": 0.0})
            continue
        lo = float(np.percentile(region, clip_low)) if clip_low > 0 else float(region.min())
        hi = float(np.percentile(region, clip_high)) if clip_high < 100 else float(region.max())
        if hi <= lo:
            rescaled[:, :, c] = 0.0
            ch_stats.append({"lo": 0.0, "hi": 0.0, "mean": 0.0, "std": 0.0})
            continue
        rescaled[:, :, c] = (np.clip(ch, lo, hi) - lo) / (hi - lo)
        ch_stats.append({"lo": lo, "hi": hi, "mean": 0.0, "std": 0.0})

    if method == "per_channel":
        for c, s in enumerate(ch_stats):
            if s["hi"] > s["lo"]:
                cregion = (rescaled[:, :, c][mask_b] if mask_b is not None
                           else rescaled[:, :, c].reshape(-1))
                s["mean"], s["std"] = float(cregion.mean()), float(cregion.std())
    elif method == "global":
        pooled = rescaled[mask_b] if mask_b is not None else rescaled.reshape(-1)
        if pooled.size > 0:
            pmean, pstd = float(pooled.mean()), float(pooled.std())
            for s in ch_stats:
                if s["hi"] > s["lo"]:
                    s["mean"], s["std"] = pmean, pstd
    elif method == "null":
        for s in ch_stats:
            s["mean"], s["std"] = 0.0, 1.0
    return ch_stats


def _normalize_fixed(img_hwc, mask, ref_stats):
    """Apply a fixed clip + rescale + z-score transform (stats from _compute_ref_stats).

    Identical semantics to microBase.normalize: per-channel percentile clip,
    min-max rescale to [0, 1], then z-score (per-channel for "per_channel",
    pooled for "global", rescale-only for "null"), background zeroed.
    """
    out = np.empty_like(img_hwc, dtype=np.float32)
    for c, s in enumerate(ref_stats):
        if s["hi"] > s["lo"] and s["std"] > 1e-6:
            clipped = np.clip(img_hwc[:, :, c].astype(np.float32), s["lo"], s["hi"])
            rescaled = (clipped - s["lo"]) / (s["hi"] - s["lo"])
            out[:, :, c] = (rescaled - s["mean"]) / s["std"]
        else:
            out[:, :, c] = 0.0
    if mask is not None:
        m = mask.astype(bool) if mask.dtype != bool else mask
        out[~m] = 0.0
    return out


def _cell_to_tensor(img_hwc, channels, aug_pipeline,
                    normalize_method, clip_low, clip_high, with_masking,
                    ref_stats=None):
    """Common single-cell pipeline: channel subset -> mask -> augment -> normalize -> CHW tensor.

    ref_stats=None uses per-view self-normalization (microBase.normalize);
    a ref_stats list (from _compute_ref_stats on the raw image) applies the
    fixed transform instead (normalize.fixed_reference: true).
    """
    if channels is not None:
        ch_idx = [c - 1 for c in channels]
        img_hwc = img_hwc[:, :, ch_idx]
    mask = (img_hwc != 0).any(axis=2).astype(np.uint8) if with_masking else None
    img_hwc, mask = apply(aug_pipeline, img_hwc, mask)
    if ref_stats is not None:
        img_hwc = _normalize_fixed(img_hwc, mask, ref_stats)
    else:
        img_hwc = normalize(img_hwc, mask=mask, method=normalize_method,
                            clip_low=clip_low, clip_high=clip_high)
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)).astype(np.float32))


# ----------------------------------------------------------------------------
# Split / subsample helpers — adapted to CellDataset's metadata DataFrame.
# ----------------------------------------------------------------------------

def stratified_split(records, val_ratio, seed):
    """Split records into train/val, stratified by 'label'.

    val_ratio: 0 disables validation (val = []); values in (0, 1) keep at
    least one record per class with >= 2 samples. A ratio >= 1 would empty
    the train set — callers must validate (train hard-exits).
    """
    rng = np.random.default_rng(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r.get("label"), []).append(r)
    train, val = [], []
    for lab, items in by_label.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_val = int(len(items) * val_ratio)
        if val_ratio > 0 and len(items) > 1:
            n_val = max(1, n_val)
        for i in idx[:n_val]:
            val.append(items[i])
        for i in idx[n_val:]:
            train.append(items[i])
    return train, val


def subsample(items, sample_max, sample_by, seed,
              label_key="label", root_key="root"):
    """Randomly sub-sample a list of dict items (records or inference entries).

    sample_by='per_class': take up to sample_max per (root, label) group.
    sample_by='per_dataset': take up to sample_max total per root with
        class-balanced stratification.
    """
    if sample_max is None:
        return items

    rng = np.random.default_rng(seed)
    total = len(items)
    if sample_max >= total:
        return list(items)

    def _label_of(i):
        lab = items[i].get(label_key)
        return lab if lab is not None else "__unlabeled__"

    def _root_of(i):
        return items[i].get(root_key, "__unknown_root__")

    if sample_by == 'per_class':
        by_group = {}
        for i in range(total):
            key = (_root_of(i), _label_of(i))
            by_group.setdefault(key, []).append(i)
        chosen = []
        for indices in by_group.values():
            n = min(sample_max, len(indices))
            chosen.extend(rng.choice(indices, size=n, replace=False).tolist())
        return [items[i] for i in chosen]

    elif sample_by == 'per_dataset':
        by_root = {}
        for i in range(total):
            by_root.setdefault(_root_of(i), []).append(i)
        result = []
        for indices in by_root.values():
            by_label = {}
            for i in indices:
                by_label.setdefault(_label_of(i), []).append(i)
            n_classes = len(by_label)
            if n_classes == 0:
                continue
            per_class = sample_max // n_classes
            remainder = sample_max - per_class * n_classes
            chosen = []
            for group_indices in by_label.values():
                n = min(per_class, len(group_indices))
                chosen.extend(rng.choice(group_indices, size=n, replace=False).tolist())
            if remainder > 0:
                chosen_set = set(chosen)
                remaining_pools = {
                    lab: [i for i in gi if i not in chosen_set]
                    for lab, gi in by_label.items()
                }
                for lab in rng.permutation(list(remaining_pools.keys())):
                    if remainder <= 0:
                        break
                    pool = remaining_pools[lab]
                    if pool:
                        chosen.append(pool.pop(0))
                        remainder -= 1
            result.extend(items[i] for i in chosen)
        return result

    else:
        print(f"Error: unknown sample_by='{sample_by}'", file=sys.stderr)
        sys.exit(1)


# ----------------------------------------------------------------------------
# SSLMultiViewDataset — for SSL pretraining (N augmented views per image).
# ----------------------------------------------------------------------------

class SSLMultiViewDataset(Dataset):
    """Multi-view dataset for SSL pretraining.

    Builds N augmentation pipelines (one per view spec). __getitem__ returns
    a list of N view tensors. No labels.
    """

    def __init__(self, cell_dataset, indices, channels,
                 augmentation_specs,              # list of N view specs
                 normalize_method, clip_low, clip_high, with_masking,
                 fixed_reference=False, max_value=None):
        self.cell_dataset = cell_dataset
        self.indices = list(indices)
        self.channels = list(channels) if channels is not None else None
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.with_masking = with_masking
        self.fixed_reference = fixed_reference
        self.max_value = max_value
        self.aug_pipelines = [build_pipeline(spec) for spec in augmentation_specs]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_idx = self.indices[idx]
        img_hwc = _to_float_max(self.cell_dataset.get_cell(cell_idx), self.max_value)
        # Fixed-reference mode: stats computed ONCE on the raw cell, then
        # applied as a fixed transform to every view (photometric aug survives).
        ref_stats = None
        if self.fixed_reference:
            ref_stats = _compute_ref_stats(
                img_hwc, self.channels, self.with_masking,
                self.clip_low, self.clip_high, self.normalize_method)
        views = [_cell_to_tensor(img_hwc, self.channels, pipe,
                                  self.normalize_method, self.clip_low,
                                  self.clip_high, self.with_masking, ref_stats)
                 for pipe in self.aug_pipelines]
        return views


# ----------------------------------------------------------------------------
# SingleCellDataset — for train (single view + label).
# ----------------------------------------------------------------------------

class SingleCellDataset(Dataset):
    """Single-view dataset for train. Supports (cell_dataset, cell_idx) pairs
    spanning multiple CellDataset roots.

    Labels come from the `labels` list (parallel to `pairs`), built by train
    from label_csv or label_from_dir resolution — the dataset never re-derives
    a label from the file path, so label_csv labels are honored verbatim.
    """

    def __init__(self, pairs, label_to_idx, labels,
                 channels=None,
                 augmentation_spec=None,
                 normalize_method="per_channel",
                 clip_low=0.05, clip_high=99.95,
                 with_masking=False,
                 fixed_reference=False,
                 max_value=None):
        self.pairs = list(pairs)
        self.labels = list(labels)
        self.label_to_idx = label_to_idx
        self.channels = list(channels) if channels is not None else None
        self.with_masking = with_masking
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.fixed_reference = fixed_reference
        self.max_value = max_value
        self.aug_pipeline = build_pipeline(augmentation_spec) if augmentation_spec else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cell_ds, cell_idx = self.pairs[idx]
        img_hwc = _to_float_max(cell_ds.get_cell(cell_idx), self.max_value)
        ref_stats = None
        if self.fixed_reference:
            ref_stats = _compute_ref_stats(
                img_hwc, self.channels, self.with_masking,
                self.clip_low, self.clip_high, self.normalize_method)
        tensor = _cell_to_tensor(
            img_hwc, self.channels, self.aug_pipeline,
            self.normalize_method, self.clip_low, self.clip_high, self.with_masking,
            ref_stats)
        label = self.labels[idx]
        label_idx = self.label_to_idx.get(label, -1)
        if label_idx < 0:
            raise ValueError(
                f"Label {label!r} is not in label_to_idx "
                f"{sorted(self.label_to_idx)} — check label_csv/label_from_dir "
                f"resolution for row {idx} ({cell_ds.metadata.iloc[cell_idx].get('path')})."
            )
        return tensor, label_idx


# ----------------------------------------------------------------------------
# WholeImageCellDataset — for whole-image feature extraction.
# ----------------------------------------------------------------------------

class WholeImageCellDataset(Dataset):
    """Whole-image dataset backed by microBase.ImageDataset.

    Holds an ImageDataset, builds _flat_index by listing cell labels per row
    (reading only masks, not full images). __getitem__ calls
    ImageDataset.get_cropped_cell then runs the same pipeline.
    """

    def __init__(self, image_dataset, mask_name, channels=None,
                 augmentation_spec=None,
                 normalize_method="per_channel",
                 clip_low=0.05, clip_high=99.95,
                 with_masking=False, fixed_reference=False, padding=4,
                 max_value=None):
        self.image_dataset = image_dataset
        self.mask_name = mask_name
        self.channels = list(channels) if channels is not None else None
        self.with_masking = with_masking
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.fixed_reference = fixed_reference
        self.padding = padding
        self.max_value = max_value
        if augmentation_spec:
            self.aug_pipeline = build_pipeline(augmentation_spec)
        else:
            self.aug_pipeline = None

        # Build flat index: list of (row_idx, label)
        self._flat_index = []
        self._field_stems = []
        md = self.image_dataset.metadata
        for row_idx in range(len(md)):
            row = md.iloc[row_idx]
            mask_path = row[self.mask_name]
            if mask_path is None or (isinstance(mask_path, float) and np.isnan(mask_path)):
                continue
            try:
                mask = read_mask(mask_path)
            except Exception as e:
                print(f"Error: failed to read mask {mask_path}: {e}", file=sys.stderr)
                sys.exit(1)
            stem = _build_field_stem(row)
            for cid in get_labels(mask):
                self._flat_index.append((row_idx, int(cid)))
                self._field_stems.append(stem)

    def __len__(self):
        return len(self._flat_index)

    def subsample(self, indices):
        """Keep only the given flat-index positions (ascending order).

        The public sampling seam for inference: meta_rows and feature rows
        are written in flat-index order, so the kept set must stay sorted to
        preserve the metadata ↔ prediction alignment.
        """
        keep = sorted(indices)
        self._flat_index = [self._flat_index[i] for i in keep]
        self._field_stems = [self._field_stems[i] for i in keep]

    def __getitem__(self, idx):
        row_idx, label = self._flat_index[idx]
        try:
            crop_hwc, _, bbox = self.image_dataset.get_cropped_cell(
                row_idx, label, self.mask_name, padding=self.padding)
        except SystemExit as e:
            # microBase cropping hard-exits (sys.exit) on corrupt/missing
            # masks; inside a DataLoader worker that surfaces as a cryptic
            # "worker died" error, so convert it to a clean exception.
            raise ValueError(
                f"Degenerate cell {label} has no pixels in mask (row {row_idx}); "
                f"mask may be corrupted or changed since indexing ({e})"
            ) from e
        crop_hwc = _to_float_max(crop_hwc, self.max_value)
        tensor = _cell_to_tensor(
            crop_hwc, self.channels, self.aug_pipeline,
            self.normalize_method, self.clip_low, self.clip_high, self.with_masking,
            _compute_ref_stats(crop_hwc, self.channels, self.with_masking,
                               self.clip_low, self.clip_high, self.normalize_method)
            if self.fixed_reference else None)
        stem = self._field_stems[idx]
        return tensor, int(label), stem, bbox

    def row_channel_filenames(self, row_idx):
        """Return list of ALL channel TIFF filenames (in channel order) for a row."""
        row = self.image_dataset.metadata.iloc[row_idx]
        if "__file__" in row and row["__file__"]:
            return [os.path.basename(row["__file__"])]
        names = []
        for col in self.image_dataset.intensity_colnames:
            if col in row and row[col]:
                names.append(os.path.basename(row[col]))
        return names

    def row_mask_filename(self, row_idx):
        """Return the full mask filepath for a row (using self.mask_name column)."""
        row = self.image_dataset.metadata.iloc[row_idx]
        if self.mask_name in row and row[self.mask_name]:
            mask_path = row[self.mask_name]
            if isinstance(mask_path, float) and np.isnan(mask_path):
                return ""
            return str(mask_path)
        return ""

    def row_source_path(self, row_idx):
        """Return the source TIFF full path for a row (for directory derivation)."""
        row = self.image_dataset.metadata.iloc[row_idx]
        if "__file__" in row and row["__file__"]:
            return row["__file__"]
        for col in self.image_dataset.intensity_colnames:
            if col in row and row[col]:
                return row[col]
        return ""


def _build_field_stem(row):
    """Build a human-readable field identifier from a metadata row."""
    parts = []
    for key in ("well", "field", "stack", "timepoint"):
        val = row.get(key)
        if val is not None and str(val) != "":
            parts.append(f"{key}={val}")
    if not parts:
        d = row.get("directory", "")
        if d:
            parts.append(str(d))
    return "|".join(parts) if parts else "field"
