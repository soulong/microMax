"""Datasets backed by microBase.CellDataset / microBase.ImageDataset.

Pipeline: load (raw HWC) -> extract mask (pixel != 0) -> augment (ToFloat +
geometric + resize; mask co-transformed) -> normalize (z-score with mask) ->
CHW tensor.

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


# ----------------------------------------------------------------------------
# Shared pipeline helper — used by all datasets.
# ----------------------------------------------------------------------------

def _cell_to_tensor(img_hwc, channels, aug_pipeline,
                    normalize_method, clip_low, clip_high, with_masking):
    """Common single-cell pipeline: channel subset -> mask -> augment -> normalize -> CHW tensor."""
    if channels is not None:
        ch_idx = [c - 1 for c in channels]
        img_hwc = img_hwc[:, :, ch_idx]
    mask = (img_hwc != 0).any(axis=2).astype(np.uint8) if with_masking else None
    img_hwc, mask = apply(aug_pipeline, img_hwc, mask)
    img_hwc = normalize(img_hwc, mask=mask, method=normalize_method,
                        clip_low=clip_low, clip_high=clip_high)
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)).astype(np.float32))


# ----------------------------------------------------------------------------
# Split / subsample helpers — adapted to CellDataset's metadata DataFrame.
# ----------------------------------------------------------------------------

def stratified_split(records, val_ratio, seed):
    """Split records into train/val, stratified by 'label'."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r.get("label"), []).append(r)
    train, val = [], []
    for lab, items in by_label.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_val = int(len(items) * val_ratio)
        if len(items) > 1:
            n_val = max(1, n_val)
        else:
            n_val = 0
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
                 normalize_method, clip_low, clip_high, with_masking):
        self.cell_dataset = cell_dataset
        self.indices = list(indices)
        self.channels = list(channels) if channels is not None else None
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.with_masking = with_masking
        self.aug_pipelines = [build_pipeline(spec) for spec in augmentation_specs]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_idx = self.indices[idx]
        img_hwc = self.cell_dataset.get_cell(cell_idx)
        views = [_cell_to_tensor(img_hwc, self.channels, pipe,
                                  self.normalize_method, self.clip_low,
                                  self.clip_high, self.with_masking)
                 for pipe in self.aug_pipelines]
        return views


# ----------------------------------------------------------------------------
# SingleCellDataset — for train (single view + label).
# ----------------------------------------------------------------------------

class SingleCellDataset(Dataset):
    """Single-view dataset for train. Supports (cell_dataset, cell_idx) pairs
    spanning multiple CellDataset roots."""

    def __init__(self, pairs, label_to_idx,
                 channels=None,
                 augmentation_spec=None,
                 normalize_method="per_channel",
                 clip_low=0.05, clip_high=99.95,
                 with_masking=False,
                 label_column="directory"):
        self.pairs = list(pairs)
        self.label_to_idx = label_to_idx
        self.channels = list(channels) if channels is not None else None
        self.with_masking = with_masking
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.label_column = label_column
        self.aug_pipeline = build_pipeline(augmentation_spec) if augmentation_spec else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cell_ds, cell_idx = self.pairs[idx]
        img_hwc = cell_ds.get_cell(cell_idx)
        tensor = _cell_to_tensor(
            img_hwc, self.channels, self.aug_pipeline,
            self.normalize_method, self.clip_low, self.clip_high, self.with_masking)
        meta = cell_ds.metadata.iloc[cell_idx]
        label = meta.get(self.label_column)
        if label is not None:
            label = os.path.basename(str(label).replace("\\", "/"))
        label_idx = self.label_to_idx.get(label, -1) if label is not None else -1
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
                 with_masking=False, padding=4):
        self.image_dataset = image_dataset
        self.mask_name = mask_name
        self.channels = list(channels) if channels is not None else None
        self.with_masking = with_masking
        self.normalize_method = normalize_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.padding = padding
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
        tensor = _cell_to_tensor(
            crop_hwc, self.channels, self.aug_pipeline,
            self.normalize_method, self.clip_low, self.clip_high, self.with_masking)
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
