"""Diversity-preserving curation of pre-cropped single-cell folders or
data.file_list CSVs (implicit roots = the listed files' parent dirs;
data.file_dir is ignored when file_list is set; the CSVs' label column is the
per-cell label).

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

Two config profiles serve the two use cases (deduplication_prune.yml /
deduplication_incremental.yml):
  - prune:    reference = null, sources = the full pool -> r thins it. The
              radius is either given directly or searched (bisection) to hit
              selection.target_keep.
  - augment:  reference = a previous selection_state.pkl — its kept cells are
              immutable seeds — and sources = new folders. Only cells bringing
              new latent territory (beyond r of the seeds) enter the pool;
              selection.max_add caps the additions.

Optional density-adaptive radius (selection.adaptive > 0):
  one global radius treats dense and sparse regions alike, so the densest
  clusters collapse onto a handful of representatives. With adaptive=p the
  per-cell radius is radius * clip((kNN distance / median)^p, 0.5, 2.0) —
  dense regions exclude with a SMALLER radius (near-duplicates survive too,
  finer morphological detail) while sparse regions exclude with a LARGER
  one (they were already covered by few anchors). p = 0 (default) keeps the
  plain uniform radius.

Optional group ratios (selection.group_by + selection.target_ratio):
  group_by clusters (Leiden, auto resolution sweep — default), label, or
  source partitions the candidates; target_ratio then compresses the
  largest:smallest group ratio (e.g. 100:1 -> 10:1) via power-compressed
  per-group quotas — each group runs its own radius search against the
  shared kept set, so global coverage survives while extreme dominance is
  damped. Without target_ratio the groups are reported only (manifest
  column, per-group summary, plot colors).

Outputs, all under output_dir (the source folders are never touched):
  selection_state.pkl  kept cells (features + paths) — the next run's reference
  manifest.csv         per-cell keep/drop decision, distance, label
  curated/<root>/      hardlinked (cross-drive: copied) kept files, directly
                       usable as a pretrain data root; rebuilt each run;
                       <root> is the path relative to the prefix shared by
                       all roots (e.g. opencell_single_cell)
  keep_label.csv       file_list-format kept list for train (labels known only)
  metrics.png          one figure: UMAP keep/drop/seeds + cluster IDs at
                       the cluster centroids + the dropped-vs-kept outcome
                       view + a per-source view + run quality metrics +
                       per-cluster keep/total stats (deterministic)
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
import igraph as ig
import leidenalg
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader

from microBase import CellDataset, MicroMaxError, canonical_directory

from .utils import (logger, set_seed, select_device, load_label_csv,
                    load_file_list,
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
CURATED_CSV_NAME = "curated.csv"
PLOT_NAME = "metrics.png"
CURATED_DIR = "curated"
CACHE_DIR = "features"

# Candidates are scored against the kept set in chunks of this many rows.
CHUNK = 8192

# Morphological grouping (selection.group_by: cluster) — Leiden over a kNN
# graph built exactly like reduction.py's cluster step. The resolution
# ladder is swept; standard (gamma=1) modularity is comparable across
# resolutions, and among partitions scoring within CLUSTER_MOD_TOL of the
# best the COARSEST one wins — splitting a homogeneous population buys
# almost no modularity but would defeat the per-group quotas.
CLUSTER_K = 15
CLUSTER_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)

# Density-adaptive radius (selection.adaptive): the per-cell radius is the
# configured radius scaled by clip((kNN dist / median)^power, LO, HI). K
# sets the density-estimate neighborhood; LO/HI bound how far the effective
# radius may deviate from the configured one in either direction.
ADAPTIVE_K = 10
ADAPTIVE_MIN = 0.5
ADAPTIVE_MAX = 2.0
CLUSTER_MOD_TOL = 0.95

# Quota floor: with target_ratio active every group keeps at least this many
# cells (bounded by the group size), so no morphology is ever wiped out.
MIN_GROUP_KEEP = 2

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
    """Normalize data.file_dir entries into per-root dicts.

    Each entry is either a plain path string (every setting from the global
    data.* defaults) or a dict with a 'path' plus any of the override keys
    (max_value, channels, ...). Paths are made absolute.
    """
    roots = data_cfg["file_dir"]
    if isinstance(roots, str):
        roots = [roots]
    if not isinstance(roots, list) or not roots:
        _error("data.file_dir must be a non-empty list of folders (strings or "
               "dicts with 'path' + overrides)")
    out = []
    for item in roots:
        if isinstance(item, str):
            entry = {"path": item}
        elif isinstance(item, dict):
            entry = dict(item)
            if "path" not in entry:
                _error(f"data.file_dir dict entry needs a 'path' key: {item}")
        else:
            _error(f"data.file_dir entries must be strings or dicts, got: {item!r}")
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


def _extract_root_features(entry, meta, model, device,
                           output_dir, bundle_id, sample_max, seed,
                           dl_cfg, only=None):
    """Full cache-aware extraction for one root.

    Returns (paths, raw_paths, feats): PORTABLE paths + their raw-case
    originals + float32 features, rows aligned. A portable path is
    CWD-relative with forward slashes when the file lives under the process
    CWD, and an absolute forward-slash path otherwise (the canonical_directory
    convention shared with curated.csv / infer.db) — the cache thus survives
    moving the working tree and the dataset together. ``paths`` is the
    normcase identity, ``raw_paths`` keeps the real case (it feeds the
    label_from_dir fallback, which must show the folder's real name).
    Consumers that touch the filesystem resolve with os.path.abspath.
    See module docstring for the cache layout.

    only: optional iterable of normcase absolute paths restricting the
    extraction to that subset of the root (the label file-list mode points
    whole folders' CSVs at these extractors). The subset rides into the
    cache key, so a CSV's features are extracted and cached once no matter
    how many times the same list comes back.
    """
    root = entry["path"]
    cell_ds = CellDataset(root, channel_layout=entry["channel_layout"],
                          image_pattern=entry["image_pattern"])
    if len(cell_ds) == 0:
        _error(f"no TIFF files found in {root}")
    _validate_channel_count(len(cell_ds.intensity_colnames),
                            entry["channels"], meta, root)
    md = cell_ds.metadata
    if only is not None:
        want = set(only)
        md = md[[os.path.normcase(os.path.abspath(p)) in want
                 for p in md["path"]]].reset_index(drop=True)
        if len(md) == 0:
            _error(f"none of the file-list files exist in {root}")

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
                                 write_pred_class=False)
    feats = torch.cat(all_feats, dim=0).numpy()

    # Portable paths (see docstring) plus their raw-case originals (the
    # label_from_dir fallback reads the folder's real name), aligned with
    # the feature rows. The cache stores these; consumers resolve them.
    raw_paths = [canonical_directory(
        os.path.abspath(md.iloc[int(i)]["path"])) for i in indices]
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

def _adaptive_scale(W_cand, power):
    """Per-cell radius scale from local density (selection.adaptive).

    The distance to the ADAPTIVE_K-th nearest candidate is a local density
    probe: dense region -> small kNN distance, sparse -> large. The scale is
    (kNN distance / median)^power, clipped to [ADAPTIVE_MIN, ADAPTIVE_MAX],
    so a dense cell's effective radius shrinks (its near-duplicates survive
    too) while a sparse cell's grows (its region needs fewer keeper anchors).
    power=0 would make every scale 1.0 — the caller treats 0 as "off" and
    never calls this. The clip bounds keep the greedy's guarantees sane:
    nothing is forced way below half the radius (near-duplicate flood) or
    above double (sparse floor wiped out).

    Returns the dimensionless scale array (multiply by the run's radius).
    """
    n = W_cand.shape[0]
    if n <= ADAPTIVE_K:
        return np.ones(n, dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    logger.info("Density probe: %d-cell kNN distances (k=%d) ...",
                n, ADAPTIVE_K)
    nn = NearestNeighbors(n_neighbors=ADAPTIVE_K + 1, algorithm="brute",
                          n_jobs=-1).fit(W_cand)
    # X=None -> query the fit data; sklearn excludes self, so column
    # ADAPTIVE_K (0-based, after self) is the k-th neighbor distance.
    dnn = nn.kneighbors()[0][:, -1].astype(np.float64)
    med = float(np.median(dnn))
    if med <= 0:
        # Degenerate pool (mass duplicate features): keep the plain radius.
        return np.ones(n, dtype=np.float32)
    scale = np.clip((dnn / med) ** float(power), ADAPTIVE_MIN, ADAPTIVE_MAX)
    logger.info("Adaptive radius scales: min %.2f / median %.2f / max %.2f "
                "(power %.2f, k=%d, bounds %.1f-%.1f)",
                scale.min(), np.median(scale), scale.max(), power,
                ADAPTIVE_K, ADAPTIVE_MIN, ADAPTIVE_MAX)
    return scale.astype(np.float32)


def _radius_pass(W_cand, W_seeds, radius, seed, device, adaptive=None):
    """One greedy pass. Returns (keep_mask, min_dist).

    Candidates are walked in a seeded-random order, chunk by chunk. Inside a
    chunk only candidates beyond the radius of the kept set S compete; they
    are processed most-isolated-first and each is additionally checked
    against the ones accepted earlier in the same chunk (the heavy C x S
    distances stay on the GPU/torch, the per-item checks run in numpy).
    min_dist records the distance to the kept set at decision time (for
    drops it is the chunk-start distance — informational only).

    adaptive: optional per-cell radius scale (from _adaptive_scale) — the
    effective radius of candidate j is radius * adaptive[j], so dense
    regions can exclude with a smaller radius and sparse ones with a larger
    one. None keeps one global radius.
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

        # Per-candidate effective radius (global radius, or scaled by the
        # density-adaptive factor of this chunk's rows).
        r_loc = (np.full(C.shape[0], radius, dtype=np.float32)
                 if adaptive is None else
                 np.asarray(radius * adaptive[idx], dtype=np.float32))

        # Only candidates beyond their own radius can be kept; process them
        # most isolated first so early accepts are the strongest
        # representatives.
        pass_ids = np.nonzero(d_base_np > r_loc)[0]
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
                if d > r_loc[j]:
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


def _search_radius(W_cand, W_seeds, target, seed, device, adaptive=None):
    """Bisection on the radius to keep ~target candidate cells.

    The kept count is monotonic non-increasing in the radius, so a plain
    bisection over [0, 2] converges; stops early within 1% of the target.
    Returns (radius, keep_mask, min_dist) of the best pass. The adaptive
    per-cell scales ride along unchanged (they are dimensionless), so the
    searched radius stays the global baseline.
    """
    tol = max(1, int(round(0.01 * target)))
    lo, hi = 0.0, MAX_DIST
    best = None  # (|count-target|, radius, keep, mind)
    for _ in range(24):
        r = 0.5 * (lo + hi)
        keep, mind = _radius_pass(W_cand, W_seeds, r, seed, device,
                                  adaptive=adaptive)
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
# Morphological groups: Leiden clustering + per-group quotas
# ----------------------------------------------------------------------------

def _cluster_groups(W, seed, resolution=None, resolutions=CLUSTER_RESOLUTIONS):
    """Cluster candidates into morphological groups (Leiden on a kNN graph).

    Returns (labels, resolution): one 0-based group id per row plus the
    resolution that produced them. With resolution=None the ladder is swept:
    standard (gamma=1) modularity is comparable across resolutions, and the
    coarsest partition scoring within CLUSTER_MOD_TOL of the best wins —
    fine partitions score marginally higher by shaving off micro-clusters,
    which would only add artificial groups to the quotas. A fixed resolution
    skips the sweep. A pool that stays one community at every swept
    resolution returns all-zeros labels with resolution None.
    """
    n = W.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), None
    n_neighbors = min(CLUSTER_K, n - 1)
    A = kneighbors_graph(W, n_neighbors, mode="connectivity",
                         include_self=False)
    A = A.maximum(A.T).tocoo()  # undirected graph, same construction as reduction
    g = ig.Graph(n=n, edges=list(zip(A.row.tolist(), A.col.tolist())))
    ladder = ([float(resolution)] if resolution is not None
              else [float(r) for r in resolutions])
    swept = []  # (modularity, resolution, n_clusters, membership)
    for res in ladder:
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res, seed=seed)
        memb = np.asarray(part.membership, dtype=int)
        k = int(memb.max()) + 1
        mod = float(g.modularity(memb.tolist())) if k >= 2 else -1.0
        logger.info("  Leiden resolution %.2f -> %d clusters "
                    "(modularity %.4f)", res, k, mod)
        swept.append((mod, res, k, memb))
    best_mod = max(s[0] for s in swept)
    eligible = [s for s in swept if s[0] >= best_mod * CLUSTER_MOD_TOL]
    best = min(eligible, key=lambda s: (s[2], s[1]))
    return best[3], best[1]


def _group_quotas(sizes, total, target_ratio, floor=MIN_GROUP_KEEP):
    """Integer per-group keep quotas compressing max:min to ~target_ratio.

    Power compression: quota_i is proportional to s_i**alpha with alpha
    solved so that (s_max/s_min)**alpha == target_ratio — e.g. a 100:1 pool
    with target_ratio 10 gives alpha 0.5 (sqrt compression), so relative
    proportions survive but extreme dominance is damped. alpha = 1 when the
    pool is already at/below the target ratio (plain proportional shares).

    Quotas sum EXACTLY to min(total, sum(sizes)), respect a per-group floor
    (bounded by the group size — no morphology is wiped out), and never
    exceed a group's size. Returns (quotas, alpha).
    """
    s = np.asarray(sizes, dtype=np.float64)
    k = len(s)
    total = int(min(total, s.sum()))
    smax, smin = float(s.max()), float(s.min())
    alpha = 1.0
    if target_ratio and smax > smin and target_ratio < smax / smin:
        alpha = float(np.log(target_ratio) / np.log(smax / smin))
    cap = s.astype(np.int64)
    lo = np.minimum(floor, cap)
    # Water-filling: split the budget proportionally to s**alpha among the
    # groups not pinned at their floor/cap; pins iterate until everyone is
    # within bounds (each iteration pins at least one group).
    quotas = np.zeros(k, dtype=np.int64)
    free = np.ones(k, dtype=bool)
    for _ in range(k + 1):
        remaining = total - int(quotas.sum())
        if remaining <= 0 or not free.any():
            break
        w = s[free] ** alpha
        q = w / w.sum() * remaining
        fidx = np.nonzero(free)[0]
        under = fidx[q <= lo[free]]
        over = fidx[q >= cap[free]]
        if under.size == 0 and over.size == 0:
            quotas[fidx] = np.rint(q).astype(np.int64)
            break
        quotas[under] = lo[under]
        quotas[over] = cap[over]
        free[under] = False
        free[over] = False
    # Rounding to the exact total: the largest groups absorb the remainder
    # (upward or downward), always within [lo, cap].
    order = np.argsort(-(s ** alpha))
    diff = total - int(quotas.sum())
    i = 0
    while diff != 0 and i < 4 * k:
        j = int(order[i % k])
        step = 1 if diff > 0 else -1
        if lo[j] <= quotas[j] + step <= cap[j]:
            quotas[j] += step
            diff -= step
        i += 1
    return quotas, alpha


def _resolve_groups(group_by, W_cand, all_labels, all_sources, name_map,
                    seed, cluster_resolution):
    """One group name per candidate cell (group_by == none -> None).

    cluster: Leiden on the selection space (auto resolution sweep unless a
    fixed one is given), clusters named cluster_1.. by decreasing size.
    label: the per-cell label (label_csv / label_from_dir), unknown -> "unknown".
    source: the root's short name.
    """
    if group_by == "cluster":
        ids, used = _cluster_groups(W_cand, seed, cluster_resolution)
        counts = np.bincount(ids)
        rank = np.empty(len(counts), dtype=int)
        rank[np.argsort(-counts)] = np.arange(len(counts))
        return np.array([f"cluster_{r + 1}" for r in rank[ids]],
                        dtype=object), used
    if group_by == "label":
        return np.array([l if l else "unknown" for l in all_labels],
                        dtype=object), None
    # group_by == "source"
    return np.array([name_map.get(os.path.normcase(s)) or _root_dirname(s)
                     for s in all_sources], dtype=object), None


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def _selection_metrics(W_cand, keep, W_seeds, radius, device):
    """Exact run-level quality numbers, measured against the FINAL kept set.

    One chunked candidate x kept-union matmul (the kept union = seeds + kept
    candidates) yields, per candidate, the distance to the nearest kept cell
    with and without self:
      coverage   — fraction of ALL candidates within radius/2 and radius of
                   the kept union (kept cells count as covered by
                   themselves). @radius is 100% by construction for a fixed
                   radius; with selection.adaptive it can fall below — that
                   gap is the price of the density adaptation.
      drop dist  — dropped cells' nearest-kept distance (median / mean /
                   p95): how much redundancy every removal had.
      isolation  — kept cells' distance to the nearest OTHER kept cell
                   (median / min): the spacing of the representative grid.
    Returns the dict for the metrics panel; every field is a plain number
    or None (nothing dropped / too few kept to space).
    """
    n = W_cand.shape[0]
    kept_cand = W_cand[keep]
    parts = ([W_seeds] if W_seeds is not None and len(W_seeds) else []) \
        + [kept_cand]
    K = torch.from_numpy(np.ascontiguousarray(np.vstack(parts))).to(device)
    n_seeds = 0 if W_seeds is None else len(W_seeds)
    C = torch.from_numpy(np.ascontiguousarray(W_cand)).to(device)
    rank = np.full(n, -1, dtype=np.int64)   # kept candidate -> its K row
    rank[np.nonzero(keep)[0]] = np.arange(len(kept_cand)) + n_seeds

    d_with_self = np.empty(n, dtype=np.float64)     # coverage distance
    d_without_self = np.full(n, np.nan, dtype=np.float64)  # kept isolation
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        sims = C[idx] @ K.T
        rows_kept = np.nonzero(rank[idx] >= 0)[0]
        if rows_kept.size:
            sims[rows_kept, rank[idx[rows_kept]]] = -np.inf  # hide self
        d_without_self[idx] = torch.sqrt(
            torch.clamp(2.0 - 2.0 * sims.max(dim=1).values,
                        min=0.0)).cpu().numpy()
        if rows_kept.size:   # restore self so coverage counts it as 0
            sims[rows_kept, rank[idx[rows_kept]]] = 1.0
        d_with_self[idx] = torch.sqrt(
            torch.clamp(2.0 - 2.0 * sims.max(dim=1).values,
                        min=0.0)).cpu().numpy()

    n_kept = int(keep.sum())
    drop = d_with_self[~keep]
    iso = d_without_self[keep]
    return {
        "n_cand": n,
        "n_kept": n_kept,
        "n_seeds": n_seeds,
        "cov_half": float((d_with_self <= radius / 2).mean()),
        "cov_full": float((d_with_self <= radius).mean()),
        "drop_med": float(np.median(drop)) if drop.size else None,
        "drop_mean": float(drop.mean()) if drop.size else None,
        "drop_p95": float(np.percentile(drop, 95)) if drop.size else None,
        "iso_med": float(np.median(iso)) if n_kept else None,
        "iso_min": float(iso.min()) if n_kept else None,
    }


def _write_plot(path, W_cand, keep, W_seeds, seed, radius, out_cfg,
                group_of=None, cluster_resolution=None,
                adaptive_power=None, source_of=None, device=None):
    """UMAP scatter of the selection outcome — all views in ONE figure.

    Five panels side by side:
      left   — the classic scatter: gray drop, group-colored (or green)
               keep, orange seeds, each cluster's ID at its centroid;
      middle — the outcome alone: dropped points gray, kept points blue,
               cluster IDs repeated at the centroids;
      third  — the same embedding colored by dataset source (every root's
               short name, kept and dropped alike), cluster IDs repeated;
      fourth — run-level quality metrics (coverage, drop distances, kept
               isolation, per-source keep rates);
      right  — per-cluster keep/total counts plus the totals row.

    UMAP is fit on everything plotted — candidates and seeds share the
    selection space — with the run seed, so the picture is deterministic.
    Cluster IDs come from group_of when groups are active (cluster_N /
    label / source names); otherwise Leiden clusters (the same construction
    as group_by=cluster) are computed here purely for the annotation.
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
    emb = umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(X)
    emb_seeds = emb[len(W_cand):]
    emb_cand = emb[:len(W_cand)]
    emb_drop = emb_cand[~keep]
    emb_keep = emb_cand[keep]

    # Per-candidate cluster names for the annotation: reuse the run's
    # groups when active, else cluster the space here (ranked by size into
    # cluster_1..N, same convention as _resolve_groups).
    if group_of is not None:
        names = group_of
    else:
        ids, _res = _cluster_groups(W_cand, seed, cluster_resolution)
        counts = np.bincount(ids)
        rank = np.empty(len(counts), dtype=int)
        rank[np.argsort(-counts)] = np.arange(len(counts))
        names = np.array([f"cluster_{r + 1}" for r in rank[ids]],
                         dtype=object)
    # Clusters ordered by size (cluster_1 = largest); centroid = mean of the
    # member points in the 2-D embedding. The plot shows the bare number for
    # cluster groups (cluster_3 -> "3"); label/source names pass through.
    order = sorted(set(names.tolist()),
                   key=lambda n: (-int((names == n).sum()), n))
    disp = {n: n[len("cluster_"):] if n.startswith("cluster_") else n
            for n in order}
    centers = {n: emb_cand[names == n].mean(axis=0) for n in order}

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        1, 5, figsize=(28.5, 7),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.72, 0.72]})

    def _cluster_labels(ax):
        """Cluster ID at each centroid, white-haloed so it stays readable
        on top of the point cloud."""
        for n, c in centers.items():
            ax.text(c[0], c[1], disp[n], fontsize=8, ha="center",
                    va="center", fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))

    # ---- left: keep/drop/seeds with group coloring ------------------------
    if len(emb_drop):
        ax1.scatter(emb_drop[:, 0], emb_drop[:, 1], s=3, c="lightgray",
                    alpha=0.4, label=f"drop ({len(emb_drop)})")
    if group_of is not None and keep.any():
        cmap = plt.colormaps["tab20"]
        for gi, name in enumerate(sorted(set(group_of[keep].tolist()))):
            m = keep & (group_of == name)
            ax1.scatter(emb_cand[m, 0], emb_cand[m, 1], s=5,
                        color=cmap(gi % 20), alpha=0.7,
                        label=f"{disp[name]} ({int(m.sum())})"
                        if gi < 15 else None)
    elif len(emb_keep):
        ax1.scatter(emb_keep[:, 0], emb_keep[:, 1], s=5, c="tab:green",
                    alpha=0.7, label=f"keep ({len(emb_keep)})")
    if len(emb_seeds):
        ax1.scatter(emb_seeds[:, 0], emb_seeds[:, 1], s=8, c="tab:orange",
                    alpha=0.8, label=f"seeds ({len(emb_seeds)})")
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.set_title(f"deduplication: radius={radius:.4f}"
                  + (f" (adaptive ^{adaptive_power:g})"
                     if adaptive_power else ""))
    ax1.legend(markerscale=3)
    _cluster_labels(ax1)

    # ---- middle: outcome only — dropped gray vs kept blue -----------------
    if len(emb_drop):
        ax2.scatter(emb_drop[:, 0], emb_drop[:, 1], s=3, c="lightgray",
                    alpha=0.4, label=f"dropped ({len(emb_drop)})")
    if len(emb_keep):
        ax2.scatter(emb_keep[:, 0], emb_keep[:, 1], s=5, c="tab:blue",
                    alpha=0.7, label=f"kept ({len(emb_keep)})")
    ax2.set_xlabel("UMAP-1")
    ax2.set_title("dropped (gray) vs kept (blue)")
    ax2.legend(markerscale=3)
    _cluster_labels(ax2)

    # ---- third: the same embedding colored by dataset source --------------
    if source_of is not None:
        cmap = plt.colormaps["tab20"]
        for si, sname in enumerate(sorted(set(source_of.tolist()))):
            m = source_of == sname
            ax3.scatter(emb_cand[m, 0], emb_cand[m, 1], s=5,
                        color=cmap(si % 20), alpha=0.7,
                        label=f"{sname} ({int(m.sum())})")
        if len(emb_seeds):
            ax3.scatter(emb_seeds[:, 0], emb_seeds[:, 1], s=8,
                        c="tab:orange", alpha=0.8,
                        label=f"seeds ({len(emb_seeds)})")
        ax3.set_xlabel("UMAP-1")
        ax3.set_title("by dataset source")
        ax3.legend(markerscale=3)
    _cluster_labels(ax3)

    # ---- fourth: run-level quality metrics ---------------------------------
    ax4.axis("off")
    ax4.set_title("metrics")
    m = _selection_metrics(W_cand, keep, W_seeds, radius,
                           device if device is not None
                           else torch.device("cpu"))
    rows_m = [
        f"kept {m['n_kept']} / {m['n_cand']}"
        f" ({100.0 * m['n_kept'] / max(1, m['n_cand']):.1f}%)",
        f"seeds {m['n_seeds']}",
        f"compression {m['n_cand'] / max(1, m['n_kept']):.1f}x",
        f"coverage @r/2  {100.0 * m['cov_half']:5.1f}%",
        f"coverage @r    {100.0 * m['cov_full']:5.1f}%",
    ]
    if m["drop_med"] is not None:
        rows_m += ["drop->kept med/avg/p95",
                   f"  {m['drop_med']:.3f} / {m['drop_mean']:.3f}"
                   f" / {m['drop_p95']:.3f}"]
    else:
        rows_m.append("drops none")
    if m["iso_med"] is not None:
        rows_m += ["kept isolation med/min",
                   f"  {m['iso_med']:.3f} / {m['iso_min']:.3f}"]
    if source_of is not None:
        rows_m.append("source keep%")
        line = ""
        for sname in sorted(set(source_of.tolist())):
            sm = source_of == sname
            pct = 100.0 * float((sm & keep).sum()) / max(1, int(sm.sum()))
            item = f"{sname} {pct:.1f}"
            if line and len(line) + 2 + len(item) > 34:
                rows_m.append("  " + line)
                line = item
            else:
                line = f"{line}  {item}" if line else item
        if line:
            rows_m.append("  " + line)
    ax4.text(0.0, 1.0, "\n".join(rows_m), va="top", ha="left",
             family="monospace", fontsize=9, transform=ax4.transAxes)

    # ---- right: per-cluster keep/total stats ------------------------------
    ax5.axis("off")
    ax5.set_title("keep / total per cluster")
    rows = []
    for n in order:
        total = int((names == n).sum())
        kept = int(((names == n) & keep).sum())
        rows.append(f"{disp[n]:<14}{kept:>8}/{total:<8}"
                    f"{100.0 * kept / total if total else 0.0:5.1f}%")
    rows.append("-" * 38)
    rows.append(f"{'total':<14}{int(keep.sum()):>8}/{len(names):<8}"
                f"{100.0 * float(keep.sum()) / max(1, len(names)):5.1f}%")
    # Shrink the font when a run produced many clusters so the panel holds.
    fs = min(10, max(5, int(400 / (len(rows) + 2))))
    ax5.text(0.0, 1.0, "\n".join(rows), va="top", ha="left",
             family="monospace", fontsize=fs, transform=ax5.transAxes)

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
    file_list = data_cfg.get("file_list")
    list_paths = list_labels = None
    if file_list:
        # File-list mode: the CSVs (filepath[, label] columns — e.g. a
        # previous curated.csv or any custom subset) ARE the pool;
        # data.file_dir is ignored. Implicit roots are the listed files'
        # distinct parent directories: per-root dict overrides don't exist,
        # the global data.* defaults hold for every file, and the CSVs'
        # label column (when present) is the per-cell label — label_from_dir
        # does not apply. The per-root "only" set keeps extraction on the
        # listed subset instead of the whole folder.
        if data_cfg.get("file_dir"):
            logger.warning("data.file_list takes precedence: data.file_dir "
                           "is ignored")
        list_paths, list_labels = load_file_list(file_list)
        pseudo = {k: v for k, v in data_cfg.items()
                  if k not in ("file_dir", "label_csv", "file_list")}
        pseudo["file_dir"] = [{"path": d} for d in sorted(
            {os.path.dirname(p) for p in list_paths})]
        roots = _parse_roots(pseudo)
        for entry in roots:
            entry["only"] = {os.path.normcase(p) for p in list_paths
                             if os.path.dirname(p) == entry["path"]}
    else:
        if "file_dir" not in data_cfg:
            _error("config 'data.file_dir' (or 'data.file_list') is required")
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
    # Optional density-adaptive radius: 0/None = off (one global radius);
    # > 0 = the exponent that turns local kNN density into per-cell radius
    # scales (dense region -> smaller radius -> keeps more; sparse -> larger
    # -> keeps fewer).
    adaptive_power = sel_cfg.get("adaptive", 0) or 0
    try:
        adaptive_power = float(adaptive_power)
    except (TypeError, ValueError):
        _error(f"selection.adaptive must be a number, got {adaptive_power!r}")
    if adaptive_power < 0:
        _error(f"selection.adaptive must be >= 0, got {adaptive_power}")
    space_cfg = config.get("space", {})
    dl_cfg = config.get("dataloader", {})
    out_cfg = config.get("output", {})
    sample_max = data_cfg.get("sample_max")

    # Morphological grouping + optional ratio compression. Groups give the
    # summary/plot/manifest a per-group view even without target_ratio; the
    # ratio itself can only be controlled when groups exist.
    group_by = str(sel_cfg.get("group_by", "cluster")).lower()
    if group_by not in ("none", "label", "source", "cluster"):
        _error("selection.group_by must be one of none | label | source | "
               f"cluster, got {group_by!r}")
    target_ratio = sel_cfg.get("target_ratio")
    if target_ratio is not None:
        if group_by == "none":
            _error("selection.target_ratio needs groups to balance — set "
                   "selection.group_by to cluster | label | source")
        try:
            target_ratio = float(target_ratio)
        except (TypeError, ValueError):
            _error(f"selection.target_ratio must be a number, got "
                   f"{target_ratio!r}")
        if target_ratio < 1:
            _error(f"selection.target_ratio must be >= 1, got {target_ratio}")
    cluster_resolution = sel_cfg.get("cluster_resolution")

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
    model, feat_dim, meta = load_ssl_backbone_from_bundle(bundle, device)
    logger.info("SSL backbone loaded: feat_dim=%d", feat_dim)
    # The readout config rides in the cache key: the same bundle yields a
    # DIFFERENT feature space when the pooling changes, so stale caches must
    # not be served across embed_source / mask_weighted changes.
    bundle_id = json.dumps([
        os.path.normcase(os.path.abspath(model_path)),
        os.path.getmtime(model_path),
        os.path.getsize(model_path),
        model.source, model.mask_weighted,
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
            entry, meta, model, device, output_dir, bundle_id,
            sample_max, seed, dl_cfg, only=entry.get("only"))
        # The cache stores PORTABLE paths (CWD-relative — see docstring);
        # the selection pipeline works with absolute filesystem paths.
        paths = [os.path.abspath(p) for p in paths]
        raw_paths = [os.path.abspath(p) for p in raw_paths]
        # Label per cell: the file-list CSVs' label column wins, then the
        # label_csv map, else the parent folder name (label_from_dir — the
        # RAW-case path, so the folder's real name is what shows up, not a
        # lowercased normcase artifact), else unknown. Carried through to
        # manifest and keep_label.csv only — labels never affect the
        # features.
        label_map = {}
        if entry.get("only") is not None:
            label_map = {os.path.normcase(p): l for p, l in
                         zip(list_paths, list_labels) if l}
        elif entry["label_csv"]:
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

    # ---- morphological groups ---------------------------------------------
    # One group name per candidate (or None when grouping is off). Clusters
    # are computed on the candidates only — reference seeds keep their own
    # decisions and never participate in quotas.
    quota_mode = group_by != "none" and target_ratio is not None
    if quota_mode and radius_cfg is not None:
        _error("selection.target_ratio works with target_keep / max_add, "
               "not with a fixed radius — drop 'radius' and let the search "
               "hit the target")
    group_of = None
    cluster_res_used = None
    if group_by != "none":
        group_of, cluster_res_used = _resolve_groups(
            group_by, W_cand, all_labels, all_sources, name_map, seed,
            cluster_resolution)
        logger.info("Groups (%s%s): %s", group_by,
                    f", resolution {cluster_res_used}"
                    if cluster_res_used is not None else "",
                    ", ".join(
                        f"{n}({int((group_of == n).sum())})"
                        for n in sorted(set(group_of.tolist()),
                                        key=lambda n: (-int((group_of == n)
                                                          .sum()), n))))

    # ---- radius resolution + selection ------------------------------------
    # The density-adaptive scales are computed ONCE for the pool (dimensionless,
    # relative to the pool median) and ride along every pass unchanged.
    adaptive = (_adaptive_scale(W_cand, adaptive_power)
                if adaptive_power > 0 else None)
    if radius_cfg is not None:
        radius = float(radius_cfg)
        logger.info("Selection: fixed radius %.4f%s", radius,
                    f" (adaptive ^{adaptive_power:g})"
                    if adaptive is not None else "")
        keep, mind = _radius_pass(W_cand, W_seeds if n_seeds else None,
                                  radius, seed, device, adaptive=adaptive)
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
        if quota_mode:
            # Per-group quotas: power-compressed shares of the target (the
            # largest group walks first and sets the coverage baseline; each
            # smaller group then keeps its own distinct cells against it).
            names = sorted(set(group_of.tolist()),
                           key=lambda n: (-int((group_of == n).sum()), n))
            sizes = [int((group_of == n).sum()) for n in names]
            quotas, alpha = _group_quotas(sizes, target, target_ratio)
            logger.info("Per-group quotas (alpha %.3f for target ratio %s): "
                        "%s", alpha, target_ratio,
                        ", ".join(f"{n}={int(q)}"
                                  for n, q in zip(names, quotas)))
            keep = np.zeros(n_cand, dtype=bool)
            mind = np.full(n_cand, np.inf, dtype=np.float32)
            kept_list = [W_seeds] if n_seeds else []
            radii = {}
            for gi, name in enumerate(names):
                idx = np.nonzero(group_of == name)[0]
                r_g, keep_g, mind_g = _search_radius(
                    W_cand[idx],
                    np.vstack(kept_list) if kept_list else None,
                    int(quotas[gi]), seed + gi, device,
                    adaptive=adaptive[idx] if adaptive is not None else None)
                keep[idx[keep_g]] = True
                mind[idx] = mind_g
                kept_list.append(W_cand[idx[keep_g]])
                radii[name] = r_g
                logger.info("  %s: radius %.4f -> %d / quota %d", name, r_g,
                            int(keep_g.sum()), int(quotas[gi]))
            radius = float(np.average([radii[n] for n in names],
                                      weights=quotas))
        else:
            radius, keep, mind = _search_radius(W_cand,
                                                W_seeds if n_seeds else None,
                                                target, seed, device,
                                                adaptive=adaptive)
    n_keep = int(keep.sum())
    logger.info("Kept %d / %d candidate cells (+ %d seeds kept as-is), "
                "radius %.4f", n_keep, n_cand, n_seeds, radius)

    # ---- outputs -----------------------------------------------------------
    # state: the kept set (seeds first, then kept candidates) is the next
    # run's reference — features, paths, sources and labels travel together.
    W_keep_cand = W_cand[keep]
    kept_idx = np.nonzero(keep)[0]
    kept_group_list = ((["seed"] * n_seeds +
                        [group_of[i] for i in kept_idx])
                       if group_of is not None
                       else [""] * (n_seeds + n_keep))
    state = {
        "bundle": os.path.abspath(model_path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pca": pca,
        "W": np.vstack([W_seeds, W_keep_cand]) if n_seeds
             else W_keep_cand,
        "paths": seed_paths + [all_paths[i] for i in kept_idx],
        "sources": seed_sources + [all_sources[i] for i in kept_idx],
        "labels": seed_labels + [all_labels[i] for i in kept_idx],
        "groups": kept_group_list,
        "group_by": group_by,
        "target_ratio": target_ratio,
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
            "group": group_of[i] if group_of is not None else "",
        })
    for i in range(n_seeds):
        rows.append({
            "filepath": seed_paths[i],
            "source": seed_sources[i],
            "role": "seed",
            "decision": "keep",
            "min_dist": "",
            "label": seed_labels[i] if seed_labels[i] is not None else "",
            "group": "seed",
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

    # curated.csv: the kept set as a plain file list — the data.file_list
    # input for pretrain/train, so the curated/ hardlink tree is optional.
    # Paths are relative to the RUN DIRECTORY (os.getcwd()) with forward
    # slashes: run train/pretrain from the same directory and the list just
    # works. Seeds are part of the kept set and included.
    if out_cfg.get("curated_csv", True):
        base = os.getcwd()
        rows = []
        cross_drive = False
        for fp, src, lbl in zip(
                [all_paths[i] for i in kept_idx] + seed_paths,
                [all_sources[i] for i in kept_idx] + seed_sources,
                [all_labels[i] for i in kept_idx] + seed_labels):
            try:
                rel = os.path.relpath(fp, base)
            except ValueError:  # different drive than the run directory
                rel = fp
                cross_drive = True
            rows.append({"filepath": rel.replace("\\", "/"),
                         "source": name_map.get(os.path.normcase(src))
                                   or _root_dirname(src),
                         "label": lbl if lbl else ""})
        cc_path = os.path.join(output_dir, CURATED_CSV_NAME)
        pd.DataFrame(rows).to_csv(cc_path, index=False)
        logger.info("Wrote %s (%d rows, relative to %s%s)", cc_path,
                    len(rows), base,
                    "; some absolute (cross-drive)" if cross_drive else "")

    # curated/: hardlinked kept files (seeds included) as a ready-to-use
    # pretrain root.
    if out_cfg.get("link_dir", True):
        kept_mask_idx = np.nonzero(keep)[0]
        _link_kept(
            [all_paths[i] for i in kept_mask_idx] + seed_paths,
            [all_sources[i] for i in kept_mask_idx] + seed_sources,
            name_map, output_dir)

    # plot: UMAP keep/drop/seed scatter + outcome view + per-source view +
    # per-cluster keep/total stats (one PNG; fit on everything plotted).
    source_of = np.array(
        [name_map.get(os.path.normcase(s)) or _root_dirname(s)
         for s in all_sources], dtype=object)
    _write_plot(os.path.join(output_dir, PLOT_NAME),
                W_cand, keep, W_seeds, seed, radius, out_cfg,
                group_of=group_of, cluster_resolution=cluster_resolution,
                adaptive_power=adaptive_power if adaptive is not None
                else None,
                source_of=source_of, device=device)

    # Per-root summary.
    logger.info("---- deduplication summary (radius %.4f) ----", radius)
    for entry in roots:
        m = manifest[(manifest["role"] == "source") &
                     (manifest["source"] == entry["path"])]
        logger.info("  %s: %d kept / %d", entry["path"],
                    int((m["decision"] == "keep").sum()), len(m))
    logger.info("  total: %d kept + %d seeds / %d candidates",
                n_keep, n_seeds, n_cand)

    # Per-group summary: the before/after max:min ratio is what
    # target_ratio promises — show it next to the promise.
    if group_of is not None:
        gnames = sorted(set(group_of.tolist()),
                        key=lambda n: (-int((group_of == n).sum()), n))
        kept_counts = {n: int(((group_of == n) & keep).sum())
                       for n in gnames}
        sizes = {n: int((group_of == n).sum()) for n in gnames}
        logger.info("---- per-group summary (group_by=%s, "
                    "target_ratio=%s) ----", group_by, target_ratio)
        for n in gnames:
            logger.info("  %s: %d kept / %d", n, kept_counts[n], sizes[n])
        logger.info("  max:min ratio %.1f -> %.1f",
                    max(sizes.values()) / max(min(sizes.values()), 1),
                    max(kept_counts.values()) /
                    max(min(kept_counts.values()), 1))
