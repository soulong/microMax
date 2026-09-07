"""I/O helpers, device selection, random seed, logging, reducer helpers."""

import os
import sys
import pickle
import random
import logging
from collections import defaultdict
import numpy as np
import torch

from microBase import load_yaml, CellDataset  # re-exported from microBase

logger = logging.getLogger("microModel")


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M",
    ))
    root_logger = logging.getLogger("microModel")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("albumentations").setLevel(logging.WARNING)


def add_file_logging(log_dir):
    """Append an additional FileHandler writing to <log_dir>/micromodel.log.

    Called after the run's output_dir is known (pretrain/train/infer/vis).
    Idempotent per path.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "micromodel.log")
    root_logger = logging.getLogger("microModel")
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == os.path.abspath(path):
            return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M",
    ))
    root_logger.addHandler(handler)
    logger.info("Logging to %s", path)


def atomic_torch_save(obj, path):
    """Save via temp file + atomic rename so an interrupted save never
    leaves a corrupt .pt behind."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def select_device():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        logger.info("CUDA device %d: %s", idx, name)
        return torch.device(f"cuda:{idx}")
    logger.info("CUDA not available, using CPU")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def load_label_csv(path):
    """Load a label CSV mapping absolute file path -> label string.

    Relative `filepath` entries are resolved against the CSV's own directory
    (NOT the process CWD), so a CSV using paths relative to a dataset root
    keeps working no matter where training is launched from. Absolute entries
    are used as-is.
    """
    import pandas as pd
    base = os.path.dirname(os.path.abspath(path))
    label_map = {}
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        fp = str(r["filepath"])
        if not os.path.isabs(fp):
            fp = os.path.join(base, fp)
        label_map[os.path.abspath(fp)] = str(r["label"])
    return label_map


def parse_pred_prob(value):
    """Parse the infer.db `pred_prob` column into a single float.

    pred_prob holds the highest per-class probability (REAL), or NULL for
    features-only rows (SSL bundles). Returns 0.0 when absent or unparseable.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def shorten_labels(paths):
    """Strip common leading/trailing path components to produce concise labels."""
    parsed = []
    for p in paths:
        head, _ = os.path.split(p)
        parts = head.replace(os.sep, "/").split("/")
        parts = [x for x in parts if x]
        parsed.append(parts if parts else [""])

    if len(parsed) < 2:
        return ["/".join(parts) if parts != [""] else "unknown" for parts in parsed]

    prefix = parsed[0][:]
    for parts in parsed[1:]:
        i = 0
        while i < len(prefix) and i < len(parts) and prefix[i] == parts[i]:
            i += 1
        prefix = prefix[:i]

    stripped = [parts[len(prefix):] for parts in parsed]
    if stripped[0]:
        suffix = stripped[0][:]
        for parts in stripped[1:]:
            k = 0
            while k < len(suffix) and k < len(parts) and suffix[-(k+1)] == parts[-(k+1)]:
                k += 1
            suffix = suffix[-k:] if k > 0 else []
            if not suffix:
                break
    else:
        suffix = []

    labels = []
    for parts in stripped:
        mid = parts[:len(parts) - len(suffix)] if suffix else parts
        label = "/".join(mid) if mid else "unknown"
        labels.append(label)
    return labels


def resolve_max_value(data_cfg):
    """Resolve and validate data.max_value (required, > 0).

    The max possible intensity of the input dtype (65535 for 16-bit, 255 for
    8-bit). Images are divided by it on load so all view pipelines and the
    fixed-reference stats share one [0, 1] domain. Shared by pretrain, train,
    and infer — a missing or invalid value hard-exits.
    """
    max_value = data_cfg.get("max_value")
    if max_value is None:
        print("Error: data.max_value is required (e.g. 65535 for 16-bit, "
              "255 for 8-bit); images are converted to float [0, 1] by it on load",
              file=sys.stderr)
        sys.exit(1)
    try:
        max_value = float(max_value)
    except (TypeError, ValueError):
        print(f"Error: data.max_value must be a number, got {max_value!r}",
              file=sys.stderr)
        sys.exit(1)
    if max_value <= 0:
        print(f"Error: data.max_value must be > 0, got {max_value}", file=sys.stderr)
        sys.exit(1)
    return max_value


def resolve_output_paths(data_roots, output_dir):
    """Pair each data root with an output dir. If output_dir is None, use the data root itself."""
    roots = [data_roots] if isinstance(data_roots, str) else list(data_roots)
    if output_dir:
        return [(r, output_dir) for r in roots]
    return [(r, r) for r in roots]


def merge_locked_normalize(saved_cfg, config):
    """Bundle wins on mismatch for the normalize.* locked keys.

    Shared by pretrain and train resume (identical behavior in both). Keys
    absent from the run config are adopted silently — a warning there would
    be pure noise.
    """
    for key in ("method", "with_masking", "clip_low", "clip_high", "fixed_reference"):
        sv = saved_cfg.get("normalize", {}).get(key)
        if sv is None:
            continue
        cv = config.get("normalize", {}).get(key)
        if cv is not None and str(cv) != str(sv):
            logger.warning("Locked normalize.%s differs; using bundle value", key)
        config.setdefault("normalize", {})[key] = sv


def resolve_channels(channels, n_avail, root):
    """Resolve data.channels against an indexed dataset.

    None means all available channels; an empty list or out-of-range
    requests hard-exit (shared by pretrain and train).
    """
    if channels is None:
        return list(range(1, n_avail + 1))
    resolved = list(channels)
    if not resolved:
        print("Error: data.channels is an empty list; "
              "use null to select all channels", file=sys.stderr)
        sys.exit(1)
    if min(resolved) < 1:
        print(
            f"Error: requested channels {resolved} must be 1-based positive "
            f"integers (got {min(resolved)})",
            file=sys.stderr,
        )
        sys.exit(1)
    if max(resolved) > n_avail:
        print(
            f"Error: requested channels {resolved} exceed available "
            f"channels {n_avail} in {root}",
            file=sys.stderr,
        )
        sys.exit(1)
    return resolved


def build_cell_datasets(roots, channel_layout, image_pattern):
    """Index one CellDataset per data root, skipping empty roots.

    Returns a list of (root, CellDataset) pairs for the non-empty datasets.
    """
    datasets = []
    for r in roots:
        logger.info("Indexing data from %s ...", r)
        cell_ds = CellDataset(r, channel_layout=channel_layout, image_pattern=image_pattern)
        if len(cell_ds) == 0:
            logger.warning("No TIFF files found in %s", r)
            continue
        datasets.append((r, cell_ds))
    return datasets


def stratified_sample_indices(n, labels, sample_per_class, seed, uniform=False):
    """Select up to sample_per_class indices, stratified by `labels`.

    uniform=True samples without stratification (used for pred_prob). The
    chosen indices are sorted. Shared by vis.show_reduction (reducer fitting)
    and vis_interactive (browser subsampling) — identical sampling semantics.
    """
    rng = np.random.default_rng(seed)
    if uniform:
        return rng.choice(n, size=min(sample_per_class, n), replace=False).tolist()
    class_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        class_indices[lab].append(i)
    selected = []
    for lab, indices in class_indices.items():
        if len(indices) > sample_per_class:
            chosen = rng.choice(indices, size=sample_per_class, replace=False).tolist()
        else:
            chosen = indices
        selected.extend(chosen)
    selected.sort()
    return selected


def copy_config_file(config_path, target_dir):
    """Copy config file to target_dir, falling back to manual read/write on PermissionError."""
    import shutil
    os.makedirs(target_dir, exist_ok=True)
    dst = os.path.join(target_dir, os.path.basename(config_path))
    try:
        shutil.copy2(config_path, dst)
    except PermissionError:
        with open(config_path, "rb") as src_f, open(dst, "wb") as dst_f:
            dst_f.write(src_f.read())
    return dst


def save_reducer(obj, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved reducer to %s", path)


def load_reducer(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded reducer from %s", path)
    return obj


def validate_pca(pca, n_features, name="reduction_pca"):
    n_in = getattr(pca, "n_features_in_", None)
    if n_in is not None and n_in != n_features:
        print(
            f"Error: {name} n_features_in_={n_in} "
            f"does not match input feature dimension {n_features}",
            file=sys.stderr,
        )
        sys.exit(1)
    if hasattr(pca, "n_components"):
        logger.info("%s: %d components, %d features", name, pca.n_components, n_in)


def validate_umap_pipeline(pipeline, n_features, name="reduction_umap"):
    pca_pre = pipeline.get("pca_pre")
    if pca_pre is not None:
        validate_pca(pca_pre, n_features, name=f"{name}.pca_pre")
    else:
        logger.info("%s: no preprocessing PCA", name)
