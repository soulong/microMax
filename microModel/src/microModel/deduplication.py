"""Diversity-preserving curation of pre-cropped single-cell folders.

`micromodel deduplication` embeds every cell with an SSL pretrain bundle's teacher
backbone (the same extraction path as infer) and selects a subset that
preserves latent-space coverage while removing excess redundancy:

    radius-coverage greedy — walking candidates in random order, a cell is
    kept only if its distance to the already-kept set exceeds a radius r;
    otherwise it is a redundant near-duplicate and dropped. Sparse latent
    regions keep everything (every region keeps a floor), dense regions are
    thinned to ~r spacing. Distances live in the whitened-PCA space also used
    for Leiden clustering in vis (rows L2-normalized, so distances are in
    [0, 2]).

One rule serves both use cases:
  - prune:    reference = null, sources = the full pool -> r thins it. The
              radius is either given directly or searched (bisection) to hit
              selection.target_keep.
  - augment:  reference = a previous selection_state.pkl — its kept cells are
              immutable seeds — and sources = new folders. Only cells bringing
              new latent territory (beyond r of the seeds) enter the pool;
              selection.max_add caps the additions.

Outputs, all under output_dir (the source folders are never touched):
  selection_state.pkl  kept cells (features + paths) — the next run's reference
  manifest.csv         per-cell keep/drop decision, distance, label
  curated/<root>/      hardlinked (cross-drive: copied) kept files, directly
                       usable as a pretrain data root; rebuilt each run;
                       <root> is the path relative to the prefix shared by
                       all roots (e.g. opencell_single_cell)
  keep_label.csv       label_csv-format kept list for train (labels known only)
  plot_umap.png        UMAP scatter of keep/drop/seeds (deterministic)
  features/            per-root feature cache keyed by bundle + file list
"""

import os
import sys
import json
import hashlib
import shutil
import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from microBase import CellDataset, MicroMaxError

from .utils import (logger, set_seed, select_device, load_label_csv,
                    save_reducer, load_reducer, validate_pca,
                    resolve_max_value, add_file_logging, copy_config_file,
                    atomic_npz_save, load_npz_cache)
from .dataset import SingleCellDataset
from .backbone import load_ssl_backbone_from_bundle
from .infer import _forward_pass, _validate_channel_count, _resolve_gt

# Output artifact names, all under output_dir.
STATE_NAME = "selection_state.pkl"
MANIFEST_NAME = "manifest.csv"
KEEP_LABEL_NAME = "keep_label.csv"
PLOT_NAME = "plot_umap.png"
CURATED_DIR = "curated"
CACHE_DIR = "features"

# Candidates are scored against the kept set in chunks of this many rows.
CHUNK = 8192

# Rows are L2-normalized, so every distance lives in [0, 2]; a radius of 2.0
# can only ever keep the unavoidable first point of an empty kept set.
MAX_DIST = 2.0

# Per-root config keys that override the global data.* defaults.
_ROOT_OVERRIDE_KEYS = ("image_pattern", "channel_layout", "channels",
                       "max_value", "label_csv", "label_from_dir")


def _error(msg):
    """Raise the standard library error (the CLI boundary prints + exits)."""
    raise MicroMaxError(f"Error: {msg}")


# ----------------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------------

def _parse_roots(data_cfg):
    """Normalize data.root entries into per-root dicts.

    Each entry is either a plain path string (every setting from the global
    data.* defaults) or a dict with a 'path' plus any of the override keys
    (max_value, channels, ...). Paths are made absolute.
    """
    roots = data_cfg["root"]
    if isinstance(roots, str):
        roots = [roots]
    if not isinstance(roots, list) or not roots:
        _error("data.root must be a non-empty list of folders (strings or "
               "dicts with 'path' + overrides)")
    out = []
    for item in roots:
        if isinstance(item, str):
            entry = {"path": item}
        elif isinstance(item, dict):
            entry = dict(item)
            if "path" not in entry:
                _error(f"data.root dict entry needs a 'path' key: {item}")
        else:
            _error(f"data.root entries must be strings or dicts, got: {item!r}")
        for key in _ROOT_OVERRIDE_KEYS:
            if key not in entry:
                entry[key] = data_cfg.get(key)
        entry["path"] = os.path.abspath(entry["path"])
        out.append(entry)
    return out


def _resolve_max_value_entry(entry):
    """Per-root max_value: explicit override wins, else the validated global.

    The global default was already validated via resolve_max_value; an
    override only needs the same > 0 number check.
    """
    mv = entry["max_value"]
    if mv is None:
        _error(f"root {entry['path']}: max_value is required (e.g. 65535 for "
               "16-bit, 255 for 8-bit)")
    try:
        mv = float(mv)
    except (TypeError, ValueError):
        _error(f"root {entry['path']}: max_value must be a number, got {mv!r}")
    if mv <= 0:
        _error(f"root {entry['path']}: max_value must be > 0, got {mv}")
    return mv


def _root_dirname(path):
    """Fallback folder name for one data root: the full sanitized path.

    Only used when a root has no short name (e.g. reference seeds from a
    previous run whose roots are not part of this run).
    """
    s = os.path.abspath(path)
    s = s.replace(":", "").replace("\\", "_").replace("/", "_").strip("_")
    return s.replace(" ", "_")


def _root_names(paths):
    """Short, collision-free folder names for the run's roots.

    Each name is the root's path relative to the prefix shared by ALL roots
    (e.g. common prefix D:\\Model\\sc_dataset -> opencell_single_cell,
    p53_2026-07-10_63x), so curated/ subfolders and cache files stay
    readable. Mixed drives have nothing in common to strip — fall back to
    full sanitized paths. Duplicate short names would silently merge two
    roots' outputs, so they are a hard error.
    """
    abspaths = [os.path.abspath(p) for p in paths]
    try:
        common = os.path.commonpath(abspaths)
    except ValueError:  # roots on different drives — nothing shared
        common = None
    names = []
    for p in abspaths:
        if common:
            rel = os.path.relpath(p, common)
            if rel == ".":  # single root, or the root IS the common prefix
                rel = os.path.basename(p)
        else:
            rel = p
        names.append(rel.replace("\\", "_").replace("/", "_")
                     .replace(":", "").replace(" ", "_"))
    if len(set(names)) != len(names):
        _error(f"roots shorten to duplicate output names {sorted(set(n for n in names if names.count(n) > 1))} "
               "— rearrange or rename the folders")
    return names


# ----------------------------------------------------------------------------
# Feature extraction with per-root caching
# ----------------------------------------------------------------------------

def _cache_path(output_dir, entry, bundle_id, rel_paths, sample_max, seed):
    """Cache file for one root: <output_dir>/features/<root>_<hash>.npz.

    The hash covers everything that changes the features: the bundle identity
    (path + mtime + size), the root, the complete file list, the channel /
    normalization-relevant overrides, the sample size AND the seed driving
    the sample_max sub-sampling (a different seed picks different cells).
    Adding files or touching the bundle therefore invalidates the cache
    automatically; "v" is a layout version that retired older caches when
    raw-case paths were added alongside the normcase ones.
    """
    key = json.dumps({
        "v": 2,
        "bundle": bundle_id,
        "root": os.path.normcase(entry["path"]),
        "files": rel_paths,
        "channels": entry["channels"],
        "channel_layout": entry["channel_layout"],
        "image_pattern": entry["image_pattern"],
        "max_value": entry["max_value"],
        "sample_max": sample_max,
        "seed": seed,
    }, sort_keys=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    name = entry["name"][:80]
    return os.path.join(output_dir, CACHE_DIR, f"{name}_{digest}.npz")


def _extract_root_features(entry, meta, model, device, pool_fn,
                           output_dir, bundle_id, sample_max, seed,
                           dl_cfg):
    """Full cache-aware extraction for one root.

    Returns (abs_paths, raw_paths, feats): normcase absolute paths + their
    raw-case originals + float32 features, rows aligned (the raw case feeds
    the label_from_dir fallback, which must show the folder's real name).
    See module docstring for the cache layout.
    """
    root = entry["path"]
    cell_ds = CellDataset(root, channel_layout=entry["channel_layout"],
                          image_pattern=entry["image_pattern"])
    if len(cell_ds) == 0:
        _error(f"no TIFF files found in {root}")
    _validate_channel_count(len(cell_ds.intensity_colnames),
                            entry["channels"], meta, root)
    md = cell_ds.metadata

    rel_paths = sorted(
        os.path.relpath(p, root).replace("\\", "/") for p in md["path"])

    indices = np.arange(len(md))
    if sample_max is not None and sample_max < len(md):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(md), size=int(sample_max),
                                     replace=False))
        logger.info("Sub-sampled %s to %d / %d cells (sample_max)",
                    root, len(indices), len(md))

    cpath = _cache_path(output_dir, entry, bundle_id, rel_paths, sample_max,
                        seed)

    if os.path.exists(cpath):
        cached = load_npz_cache(cpath)
        if cached:
            paths = [str(p) for p in cached["paths"]]
            raw_paths = [str(p) for p in cached["paths_raw"]]
            feats = cached["feats"]
            logger.info("Feature cache hit: %s (%d cells, dim %d)",
                        os.path.basename(cpath), len(paths), feats.shape[1])
            return paths, raw_paths, feats

    # Bundle-meta normalization + inference augmentation — NOT configurable
    # here, exactly like infer (the embedding must match the bundle).
    required_meta = ("augmentation_infer", "normalize_method",
                     "normalize_with_masking", "clip_low", "clip_high",
                     "normalize_fixed_reference")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        _error(f"bundle meta missing required keys: {missing}")
    dataset = SingleCellDataset(
        [(cell_ds, int(i)) for i in indices],
        {"_": 0}, ["_"] * len(indices),
        channels=entry["channels"],
        augmentation_spec=meta["augmentation_infer"],
        normalize_method=meta["normalize_method"],
        clip_low=meta["clip_low"], clip_high=meta["clip_high"],
        with_masking=meta["normalize_with_masking"],
        fixed_reference=bool(meta["normalize_fixed_reference"]),
        max_value=_resolve_max_value_entry(entry))

    num_workers = dl_cfg.get("num_workers", 4)
    loader_kwargs = dict(batch_size=dl_cfg.get("batch_size", 128),
                         shuffle=False,
                         num_workers=num_workers,
                         persistent_workers=dl_cfg.get(
                             "persistent_workers", True) and num_workers > 0)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_cfg.get("prefetch_factor", 2)
    loader = DataLoader(dataset, **loader_kwargs)

    logger.info("Extracting teacher features for %s (%d cells) ...",
                root, len(indices))
    _, all_feats = _forward_pass(loader, model, device, True, False,
                                 write_pred_class=False, pool_fn=pool_fn)
    feats = torch.cat(all_feats, dim=0).numpy()

    # Normcase absolute paths (dedup / lookups) plus their raw-case originals
    # (the label_from_dir fallback reads the folder's real name), aligned
    # with the feature rows.
    raw_paths = [os.path.abspath(md.iloc[int(i)]["path"]) for i in indices]
    paths = [os.path.normcase(p) for p in raw_paths]

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    atomic_npz_save(cpath, paths=np.array(paths),
                    paths_raw=np.array(raw_paths), feats=feats)
    logger.info("Cached features to %s", cpath)
    return paths, raw_paths, feats


# ----------------------------------------------------------------------------
# Reference state (previous run's kept set)
# ----------------------------------------------------------------------------

def _load_reference(ref_path, bundle_path):
    """Load and sanity-check a previous selection_state.pkl.

    Returns the state dict, or None when ref_path is empty. The reference is
    only comparable if it was built from the SAME bundle — a different bundle
    means a different embedding space, so distances would be meaningless.
    """
    if not ref_path:
        return None
    if not os.path.exists(ref_path):
        _error(f"reference state not found: {ref_path}")
    state = load_reducer(ref_path)
    for key in ("bundle", "pca", "W", "paths", "sources", "labels"):
        if key not in state:
            _error(f"reference state {ref_path} lacks key '{key}' "
                   "(not a deduplication selection_state.pkl?)")
    if os.path.normcase(os.path.abspath(state["bundle"])) != \
            os.path.normcase(os.path.abspath(bundle_path)):
        _error(
            f"reference state was built from bundle {state['bundle']} but "
            f"this run uses {bundle_path} — different embedding spaces are "
            "not comparable; re-run the reference pool with the new bundle")
    if len(state["paths"]) == 0:
        _error(f"reference state {ref_path} contains no kept cells")
    logger.info("Reference: %d seed cells from %s",
                len(state["paths"]), ref_path)
    return state


# ----------------------------------------------------------------------------
# Selection space: whitened PCA + L2 normalization (vis.py convention)
# ----------------------------------------------------------------------------

def _l2_normalize(W):
    """Row-wise L2 normalization (1e-8 guard, same as vis.py)."""
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-8)


def _build_space(feats_all, pca_components, seed=42):
    """Fit the whitened-PCA selection space on the source features.

    Returns (pca, W_sources). Dimensionality is capped by both the sample
    count and the feature dim so tiny runs still work. random_state makes
    randomized SVD deterministic across runs.
    """
    n_white = min(int(pca_components), feats_all.shape[0], feats_all.shape[1])
    if n_white < 2:
        _error(f"selection space needs >= 2 PCA components, got {n_white} "
               f"({feats_all.shape[0]} cells x {feats_all.shape[1]} dims)")
    pca = PCA(n_components=n_white, whiten=True,
              random_state=int(seed)).fit(feats_all)
    W = pca.transform(feats_all).astype(np.float32)
    W = _l2_normalize(W)
    logger.info("Selection space: whitened PCA %dd (%.0f%% variance) + "
                "L2-normalized rows", n_white,
                100 * float(pca.explained_variance_ratio_.sum()))
    return pca, np.ascontiguousarray(W)


def _apply_space(pca, feats):
    """Project features through an existing (reference) PCA + L2 normalize."""
    validate_pca(pca, feats.shape[1], name="reference pca")
    W = pca.transform(feats).astype(np.float32)
    return np.ascontiguousarray(_l2_normalize(W))


# ----------------------------------------------------------------------------
# Radius-coverage greedy selection
# ----------------------------------------------------------------------------

def _radius_pass(W_cand, W_seeds, radius, seed, device):
    """One greedy pass at a fixed radius. Returns (keep_mask, min_dist).

    Candidates are walked in a seeded-random order, chunk by chunk. Inside a
    chunk only candidates beyond the radius of the kept set S compete; they
    are processed most-isolated-first and each is additionally checked
    against the ones accepted earlier in the same chunk (the heavy C x S
    distances stay on the GPU/torch, the per-item checks run in numpy).
    min_dist records the distance to the kept set at decision time (for
    drops it is the chunk-start distance — informational only).
    """
    n = W_cand.shape[0]
    rng = np.random.default_rng(seed)
    keep = np.zeros(n, dtype=bool)
    mind = np.full(n, np.inf, dtype=np.float32)

    C_all = torch.from_numpy(np.ascontiguousarray(W_cand)).to(device)
    if W_seeds is not None and len(W_seeds) > 0:
        S = torch.from_numpy(np.ascontiguousarray(W_seeds)).to(device)
    else:
        S = torch.empty((0, W_cand.shape[1]), dtype=torch.float32,
                        device=device)

    order = rng.permutation(n)
    for start in range(0, n, CHUNK):
        idx = order[start:start + CHUNK]
        C = C_all[idx]
        if S.shape[0] > 0:
            # Squared euclidean on unit rows: d^2 = 2 - 2*cos; min distance
            # <=> max dot product. Monotonic sqrt applied after the min.
            d2 = 2.0 - 2.0 * (C @ S.T)
            d_base = torch.clamp(d2.min(dim=1).values, min=0.0)
            d_base_np = d_base.sqrt().cpu().numpy()
        else:
            # Empty kept set: the first candidate always passes.
            d_base_np = np.full(C.shape[0], np.inf, dtype=np.float32)

        # Only candidates beyond the radius can be kept; process them most
        # isolated first so early accepts are the strongest representatives.
        pass_ids = np.nonzero(d_base_np > radius)[0]
        settled = np.zeros(C.shape[0], dtype=bool)  # mind already recorded
        acc = None  # accepted rows of this chunk, filled below
        acc_n = 0
        if pass_ids.size:
            pass_ids = pass_ids[np.argsort(-d_base_np[pass_ids])]

            # Preallocated buffer for the accepted rows of this chunk — the
            # per-item check is a small BLAS matvec, no GPU sync needed.
            acc = np.empty((C.shape[0], W_cand.shape[1]), dtype=np.float32)
            rows_np = C.cpu().numpy()
            for j in pass_ids:
                d = float(d_base_np[j])
                if acc_n > 0:
                    dots = acc[:acc_n] @ rows_np[j]
                    d_acc = float(np.sqrt(max(0.0, 2.0 - 2.0 * float(dots.max()))))
                    if d_acc < d:
                        d = d_acc
                settled[j] = True
                if d > radius:
                    gidx = int(idx[j])
                    keep[gidx] = True
                    mind[gidx] = d
                    acc[acc_n] = rows_np[j]
                    acc_n += 1
                else:
                    mind[int(idx[j])] = d
        # Candidates at/below the radius are dropped regardless of in-chunk
        # accepts (accepts only ever decrease distances).
        unset = ~settled
        if unset.any():
            umask = np.nonzero(unset)[0]
            mind[idx[umask]] = d_base_np[umask]

        if acc_n > 0:
            S = torch.cat([S, torch.from_numpy(acc[:acc_n]).to(device)], dim=0)

    return keep, mind


def _search_radius(W_cand, W_seeds, target, seed, device):
    """Bisection on the radius to keep ~target candidate cells.

    The kept count is monotonic non-increasing in the radius, so a plain
    bisection over [0, 2] converges; stops early within 1% of the target.
    Returns (radius, keep_mask, min_dist) of the best pass.
    """
    tol = max(1, int(round(0.01 * target)))
    lo, hi = 0.0, MAX_DIST
    best = None  # (|count-target|, radius, keep, mind)
    for _ in range(24):
        r = 0.5 * (lo + hi)
        keep, mind = _radius_pass(W_cand, W_seeds, r, seed, device)
        count = int(keep.sum())
        diff = abs(count - target)
        if best is None or diff < best[0]:
            best = (diff, r, keep, mind)
            logger.info("  radius %.4f -> keep %d (target %d)", r, count,
                        target)
        if diff <= tol:
            break
        # Too many kept -> radius too small -> search the upper half.
        if count > target:
            lo = r
        else:
            hi = r
    return best[1], best[2], best[3]


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def _write_plot(path, W_cand, keep, W_seeds, seed, radius, out_cfg):
    """UMAP scatter of the selection outcome (drop / keep / seeds).

    UMAP is fit on everything plotted — candidates and seeds share the
    selection space — with the run seed, so the picture is deterministic.
    """
    if not out_cfg.get("plot", True):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    X = np.vstack([W_cand, W_seeds]) if len(W_seeds) else W_cand
    logger.info("Fitting UMAP on %d points for %s ...", X.shape[0],
                os.path.basename(path))
    emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    emb_seeds = emb[len(W_cand):]
    emb_cand = emb[:len(W_cand)]
    emb_drop = emb_cand[~keep]
    emb_keep = emb_cand[keep]

    fig, ax = plt.subplots(figsize=(8, 7))
    if len(emb_drop):
        ax.scatter(emb_drop[:, 0], emb_drop[:, 1], s=3, c="lightgray",
                   alpha=0.4, label=f"drop ({len(emb_drop)})")
    if len(emb_keep):
        ax.scatter(emb_keep[:, 0], emb_keep[:, 1], s=5, c="tab:green",
                   alpha=0.7, label=f"keep ({len(emb_keep)})")
    if len(emb_seeds):
        ax.scatter(emb_seeds[:, 0], emb_seeds[:, 1], s=8, c="tab:orange",
                   alpha=0.8, label=f"seeds ({len(emb_seeds)})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"deduplication: radius={radius:.4f}")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _link_kept(kept_paths, kept_sources, name_map, output_dir):
    """Hardlink kept files into curated/<root name>/<relative path>.

    Same-drive links are zero-copy; a cross-drive OSError falls back to a
    real copy. Root names come from name_map (this run's roots + any names
    inherited from the reference state); unknown sources fall back to the
    full sanitized path. The curated tree is fully derived from the kept
    set, so it is wiped and rebuilt on every run.
    """
    curated_root = os.path.join(output_dir, CURATED_DIR)
    if os.path.exists(curated_root):
        logger.info("Rebuilding %s", curated_root)
        shutil.rmtree(curated_root)
    linked = copied = 0
    for path, source in zip(kept_paths, kept_sources):
        name = name_map.get(os.path.normcase(source)) or _root_dirname(source)
        rel = os.path.relpath(path, source)
        dest = os.path.join(curated_root, name, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            continue
        try:
            os.link(path, dest)
            linked += 1
        except OSError:
            shutil.copy2(path, dest)
            copied += 1
    logger.info("curated/: %d hardlinks + %d copies under %s",
                linked, copied, curated_root)
    return curated_root


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_deduplication(config, config_path=None):
    """micromodel deduplication — see the module docstring for the full design."""
    # ---- config -----------------------------------------------------------
    model_path = config["model"]
    if not isinstance(model_path, str) or not model_path:
        _error("config 'model' must be a string path to an SSL bundle")
    if not os.path.exists(model_path):
        _error(f"model not found: {model_path}")
    output_dir = config["output_dir"]
    if not output_dir:
        _error("config 'output_dir' is required")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    if config_path is not None:
        copy_config_file(config_path, output_dir)

    data_cfg = config.get("data", {})
    if "root" not in data_cfg:
        _error("config 'data.root' is required")
    roots = _parse_roots(data_cfg)
    # Short per-root names for curated/ subfolders + cache files: the path
    # relative to the prefix shared by ALL roots (e.g. opencell_single_cell).
    for entry, name in zip(roots, _root_names([e["path"] for e in roots])):
        entry["name"] = name
    # Global default max_value (per-root entries may override it). Only
    # validated when actually present — a config that sets max_value on
    # every root is valid, and roots without any value are rejected with
    # the same message per-root in _resolve_max_value_entry.
    if data_cfg.get("max_value") is not None:
        resolve_max_value(data_cfg)

    sel_cfg = config.get("selection", {})
    radius_cfg = sel_cfg.get("radius")
    target_keep = sel_cfg.get("target_keep")
    max_add = sel_cfg.get("max_add")
    seed = int(sel_cfg.get("seed", 42))
    space_cfg = config.get("space", {})
    dl_cfg = config.get("dataloader", {})
    out_cfg = config.get("output", {})
    sample_max = data_cfg.get("sample_max")

    # ---- bundle (SSL features-only, teacher branch) -----------------------
    logger.info("Loading bundle from %s", model_path)
    device = select_device()
    set_seed(seed)
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    meta = bundle["meta"]
    if "state_dict" not in bundle or "num_classes" in meta:
        _error("deduplication needs an SSL pretrain bundle (state_dict without "
               "num_classes); a train classifier bundle has no shared "
               "embedding space")
    model, feat_dim, pool_fn, meta = load_ssl_backbone_from_bundle(bundle,
                                                                   device)
    logger.info("SSL backbone loaded: feat_dim=%d", feat_dim)
    bundle_id = json.dumps([
        os.path.normcase(os.path.abspath(model_path)),
        os.path.getmtime(model_path),
        os.path.getsize(model_path),
    ])

    # ---- reference (previous run's kept set as immutable seeds) -----------
    ref = _load_reference(config.get("reference"), model_path)
    if ref is not None:
        pca = ref["pca"]
        W_seeds = np.ascontiguousarray(ref["W"], dtype=np.float32)
        seed_paths = list(ref["paths"])
        seed_sources = list(ref["sources"])
        seed_labels = list(ref["labels"])
        if W_seeds.shape[1] != pca.n_components:
            _error("reference state is inconsistent: W dim "
                   f"{W_seeds.shape[1]} != pca components {pca.n_components}")
        if len(seed_paths) != len(seed_sources) or \
                len(seed_paths) != W_seeds.shape[0]:
            _error("reference state is inconsistent: paths/sources/W lengths "
                   "differ")
    else:
        pca = None
        W_seeds = np.empty((0, 0), dtype=np.float32)
        seed_paths, seed_sources, seed_labels = [], [], []
    n_seeds = len(seed_paths)

    # Root-name map for curated/: this run's roots, extended with the names
    # recorded in the reference state so seed files land in their original
    # (short-named) folders even when their root is not part of this run.
    name_map = {os.path.normcase(e["path"]): e["name"] for e in roots}
    if ref is not None:
        for k, v in (ref.get("root_names") or {}).items():
            name_map.setdefault(k, v)

    # ---- feature extraction (cache-aware) ---------------------------------
    all_paths, all_sources, all_labels, feats_list = [], [], [], []
    for entry in roots:
        paths, raw_paths, feats = _extract_root_features(
            entry, meta, model, device, pool_fn, output_dir, bundle_id,
            sample_max, seed, dl_cfg)
        # Label per cell: label_csv map wins, else the parent folder name
        # (label_from_dir — the RAW-case path, so the folder's real name is
        # what shows up, not a lowercased normcase artifact), else unknown.
        # Carried through to manifest and keep_label.csv only — labels never
        # affect the features.
        label_map = {}
        if entry["label_csv"]:
            if not os.path.exists(entry["label_csv"]):
                _error(f"label_csv not found: {entry['label_csv']}")
            label_map = load_label_csv(entry["label_csv"])
        for p, raw in zip(paths, raw_paths):
            all_paths.append(p)
            all_sources.append(entry["path"])
            all_labels.append(_resolve_gt(
                label_map, entry["label_from_dir"], p,
                os.path.dirname(raw).replace("\\", "/")))
        feats_list.append(feats)
        logger.info("%s: %d cells ready", entry["path"], len(paths))

    # Overlapping roots would double-count cells and collide in curated/.
    if len(set(all_paths)) != len(all_paths):
        seen, dupes = set(), []
        for p in all_paths:
            if p in seen and p not in dupes:
                dupes.append(p)
            seen.add(p)
        _error(f"overlapping roots: {len(dupes)} cells appear more than "
               f"once, e.g. {dupes[:5]}")
    feats_all = np.ascontiguousarray(
        np.vstack([f.astype(np.float32, copy=False) for f in feats_list]))
    n_cand = feats_all.shape[0]
    logger.info("Total candidate cells: %d (+ %d reference seeds)",
                n_cand, n_seeds)
    if n_cand == 0:
        _error("no candidate cells found in any root")

    # Free the extractor before the selection math.
    del model, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- selection space ---------------------------------------------------
    if pca is not None:
        if int(space_cfg.get("pca_components", 50)) != pca.n_components:
            logger.warning("space.pca_components ignored: the reference "
                           "state's PCA (%d components) is reused so new "
                           "cells stay comparable with the seeds",
                           pca.n_components)
        W_cand = _apply_space(pca, feats_all)
    else:
        pca, W_cand = _build_space(feats_all,
                                   space_cfg.get("pca_components", 50),
                                   seed=seed)

    # ---- radius resolution + selection ------------------------------------
    if radius_cfg is not None:
        radius = float(radius_cfg)
        logger.info("Selection: fixed radius %.4f", radius)
        keep, mind = _radius_pass(W_cand, W_seeds if n_seeds else None,
                                  radius, seed, device)
    else:
        if n_seeds > 0 and max_add is not None:
            target, what = int(max_add), f"max_add={max_add} new cells"
        elif target_keep is not None:
            target = int(target_keep) - n_seeds
            what = f"target_keep={target_keep} total ({target} from sources)"
        elif max_add is not None:
            target, what = int(max_add), f"max_add={max_add} cells"
        else:
            _error("selection needs one of: radius, target_keep, max_add")
        if target <= 0:
            _error(f"selection target resolves to {target} cells "
                   f"({what}); nothing to select")
        target = min(target, n_cand)
        logger.info("Selection: searching radius for %s", what)
        radius, keep, mind = _search_radius(W_cand,
                                            W_seeds if n_seeds else None,
                                            target, seed, device)
    n_keep = int(keep.sum())
    logger.info("Kept %d / %d candidate cells (+ %d seeds kept as-is), "
                "radius %.4f", n_keep, n_cand, n_seeds, radius)

    # ---- outputs -----------------------------------------------------------
    # state: the kept set (seeds first, then kept candidates) is the next
    # run's reference — features, paths, sources and labels travel together.
    W_keep_cand = W_cand[keep]
    state = {
        "bundle": os.path.abspath(model_path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pca": pca,
        "W": np.vstack([W_seeds, W_keep_cand]) if n_seeds
             else W_keep_cand,
        "paths": seed_paths + [all_paths[i] for i in np.nonzero(keep)[0]],
        "sources": seed_sources + [all_sources[i] for i in np.nonzero(keep)[0]],
        "labels": seed_labels + [all_labels[i] for i in np.nonzero(keep)[0]],
        "radius": radius,
        "seed": seed,
        "space": {"pca_components": int(pca.n_components)},
        "reference": config.get("reference"),
        "root_names": name_map,
    }
    save_reducer(state, os.path.join(output_dir, STATE_NAME))

    # manifest: every candidate cell with its decision (+ the seeds).
    rows = []
    for i in range(n_cand):
        rows.append({
            "filepath": all_paths[i],
            "source": all_sources[i],
            "role": "source",
            "decision": "keep" if keep[i] else "drop",
            "min_dist": float(mind[i]) if np.isfinite(mind[i]) else "",
            "label": all_labels[i] if all_labels[i] is not None else "",
        })
    for i in range(n_seeds):
        rows.append({
            "filepath": seed_paths[i],
            "source": seed_sources[i],
            "role": "seed",
            "decision": "keep",
            "min_dist": "",
            "label": seed_labels[i] if seed_labels[i] is not None else "",
        })
    manifest = pd.DataFrame(rows).sort_values(
        ["source", "filepath"]).reset_index(drop=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    manifest.to_csv(manifest_path, index=False)
    logger.info("Wrote %s (%d rows)", manifest_path, len(manifest))

    # keep_label.csv: label_csv-format kept list for train (only when the
    # kept cells actually carry labels).
    if out_cfg.get("keep_label_csv", True):
        kept_rows = manifest[manifest["decision"] == "keep"]
        labeled = kept_rows[kept_rows["label"].astype(str).str.len() > 0]
        if len(labeled):
            kl_path = os.path.join(output_dir, KEEP_LABEL_NAME)
            labeled[["filepath", "label"]].to_csv(kl_path, index=False)
            logger.info("Wrote %s (%d labeled kept cells)", kl_path,
                        len(labeled))
        else:
            logger.info("No labels known for kept cells; %s skipped",
                        KEEP_LABEL_NAME)

    # curated/: hardlinked kept files (seeds included) as a ready-to-use
    # pretrain root.
    if out_cfg.get("link_dir", True):
        kept_mask_idx = np.nonzero(keep)[0]
        _link_kept(
            [all_paths[i] for i in kept_mask_idx] + seed_paths,
            [all_sources[i] for i in kept_mask_idx] + seed_sources,
            name_map, output_dir)

    # plot: UMAP keep/drop/seed scatter (fit on everything plotted).
    _write_plot(os.path.join(output_dir, PLOT_NAME),
                W_cand, keep, W_seeds, seed, radius, out_cfg)

    # Per-root summary.
    logger.info("---- deduplication summary (radius %.4f) ----", radius)
    for entry in roots:
        m = manifest[(manifest["role"] == "source") &
                     (manifest["source"] == entry["path"])]
        logger.info("  %s: %d kept / %d", entry["path"],
                    int((m["decision"] == "keep").sum()), len(m))
    logger.info("  total: %d kept + %d seeds / %d candidates",
                n_keep, n_seeds, n_cand)
