from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from microBase import load_yaml
from microBase.db_contracts import INFER_DB_NAME, MASK_COLUMN_PREFIX

logger = logging.getLogger(__name__)


def default_n_workers() -> int:
    """Default worker count for profiling steps: half the CPU count, min 1."""
    return max(1, (os.cpu_count() or 1) // 2)


class ProjectionMethod(str, Enum):
    max = "max"
    mean = "mean"
    min = "min"


_BASIC_MODES = ("fit", "transform", "fit-transform")

# Canonical pipeline step sections — single source of truth for the order in
# which sections are parsed/serialized (must match _STEP_FUNCTIONS keys in
# pipeline/steps.py).
SECTION_ATTRS = (
    "resize", "zproject", "basic", "tile",
    "segment", "image_profile", "object_profile", "inference",
)


@dataclass
class FilterEntry:
    column: str = ""
    pattern: str = ""


@dataclass
class ResizeConfig:
    """Rescale every image (in place) by an isotropic factor."""

    run: bool = False
    scale_factor: float = 1.0          # > 0; e.g. 0.5 halves both dimensions


@dataclass
class BasicConfig:
    """BaSiC illumination correction (in place). Models live under
    <dataset>/.microprofiler/BaSiC_model/."""

    run: bool = False
    mode: str = "fit-transform"        # fit | transform | fit-transform
    n_image: int = 100                 # images sampled to fit the shading models
    working_size: int = 64             # downscaled width used while fitting
    enable_darkfield: bool = False     # also estimate + remove darkfield offset


@dataclass
class ZProjectConfig:
    """Project each z-stack to a single plane (in place; needs a `stack` column)."""

    run: bool = False
    method: ProjectionMethod = ProjectionMethod.max   # projection method name


@dataclass
class TileConfig:
    """Split each image into tiles (in place; needs a `field` column).

    Only complete tiles are written — right/bottom remainders (and images
    smaller than the tile size) are intentionally dropped.
    """

    run: bool = False
    tile_width: int = 1024
    tile_height: int = 1024


@dataclass
class SegmentEntry:
    """One Cellpose run over the dataset, writing <stem>_cp_masks_<object_name>.png."""

    object_name: str = "cell"          # mask name, used by profiling/inference
    model_name: str = "cpdino"         # cellpose model (cpdino / cpdino-vitb / ...)
    chan1: List[str] = field(default_factory=list)   # cytoplasm channel(s); [] = block skipped (never "all")
    chan2: Optional[List[str]] = None  # nuclear channel(s) for two-channel models
    merge1: str = "mean"               # how chan1 channels combine before the model
    merge2: str = "mean"
    resize_factor: float = 0.5         # images are downscaled by this before segmentation
    diameter: Optional[float] = None   # object diameter in px; null = cellpose auto
    flow_threshold: float = 0.4        # cellpose flow-error threshold
    cellprob_threshold: float = 0.0    # cellpose cell-probability threshold
    edge_pixel_ratio: float = 0.4      # drop masks whose edge-pixel/perimeter ratio exceeds this ([0, 1]; 1 = off)
    gpu_batch_size: int = 32           # images per GPU batch (<= 1 = no batching)
    overwrite_mask: bool = False       # re-segment when a mask file already exists


@dataclass
class SegmentConfig:
    run: bool = False
    configs: List[SegmentEntry] = field(default_factory=list)


@dataclass
class ImageProfileConfig:
    """Per-site whole-image intensity table (`image` in profiler.db)."""

    run: bool = False
    n_workers: int = field(default_factory=default_n_workers)
    image_channels: Optional[List[str]] = None   # channels to profile; null/[] = step skipped (never "all")
    image_thresholds: Optional[Dict[str, float]] = None   # optional per-channel foreground thresholds


@dataclass
class ObjectProfileEntry:
    """Per-object features for one mask type — one table in profiler.db.

    All channel lists mean "these channels"; null/[] skips that feature
    group (never an implicit "all channels").
    """

    mask_name: Optional[str] = None                 # which masks to crop objects from
    parent_mask_name: Optional[str] = None          # parent objects this one lives in (e.g. cell for nuclei)
    output_table_name: Optional[str] = None         # profiler.db table name; null = mask_name
    overwrite_db: bool = False                      # drop + rewrite the table on re-run
    intensity_channels: Optional[List[str]] = None  # per-channel intensity stats; null/[] = block skipped
    radial_channels: Optional[List[str]] = None     # radial distribution profile channels
    radial_bins: int = 4                            # radial bins per profile
    granularity_channels: Optional[List[str]] = None       # granularity (texture spectrum) channels
    granularity_spectrum_length: Optional[int] = None      # spectrum length (resolved default 8)
    granularity_subsample_ratio: Optional[float] = None    # (0, 1] pixel subsample inside the object
    granularity_background_subsample_ratio: Optional[float] = None   # (0, 1] subsample of the background ring
    granularity_background_radius: Optional[int] = None    # background ring width (px) around the object
    glcm_channels: Optional[List[str]] = None       # GLCM texture feature channels
    glcm_distances: Optional[List[int]] = None      # GLCM offsets in px (resolved default [2])
    glcm_levels: Optional[int] = None               # GLCM gray levels (resolved default 256)
    correlation_pairs: Optional[List[List[str]]] = None   # channel pairs for Pearson correlation

    def resolved(self) -> "ResolvedProfiling":
        granularity_spectrum_length = self.granularity_spectrum_length if self.granularity_spectrum_length is not None else 8
        granularity_subsample_ratio = self.granularity_subsample_ratio if self.granularity_subsample_ratio is not None else 0.5
        granularity_background_subsample_ratio = self.granularity_background_subsample_ratio if self.granularity_background_subsample_ratio is not None else 0.25
        granularity_background_radius = self.granularity_background_radius if self.granularity_background_radius is not None else 10

        glcm_distances = [2]
        glcm_levels: int = 256

        # null and [] both resolve to the documented default [2] — an empty
        # distance list has no valid GLCM meaning.
        if self.glcm_distances is not None and self.glcm_distances:
            glcm_distances = self.glcm_distances
        if self.glcm_levels is not None:
            glcm_levels = self.glcm_levels

        return ResolvedProfiling(
            mask_name=self.mask_name,
            parent_mask_name=self.parent_mask_name,
            output_table_name=self.output_table_name,
            intensity_channels=self.intensity_channels,
            radial_channels=self.radial_channels,
            radial_bins=self.radial_bins,
            granularity_channels=self.granularity_channels,
            granularity_spectrum_length=granularity_spectrum_length,
            granularity_subsample_ratio=granularity_subsample_ratio,
            granularity_background_subsample_ratio=granularity_background_subsample_ratio,
            granularity_background_radius=granularity_background_radius,
            glcm_channels=self.glcm_channels,
            glcm_distances=glcm_distances,
            glcm_levels=glcm_levels,
            correlation_pairs=self.correlation_pairs,
        )


@dataclass(frozen=True)
class ResolvedProfiling:
    mask_name: Optional[str] = None
    parent_mask_name: Optional[str] = None
    output_table_name: Optional[str] = None
    intensity_channels: Optional[List[str]] = None
    radial_channels: Optional[List[str]] = None
    radial_bins: int = 4
    granularity_channels: Optional[List[str]] = None
    granularity_spectrum_length: int = 8
    granularity_subsample_ratio: float = 0.5
    granularity_background_subsample_ratio: float = 0.25
    granularity_background_radius: int = 10
    glcm_channels: Optional[List[str]] = None
    glcm_distances: List[int] = field(default_factory=lambda: [2])
    glcm_levels: int = 256
    correlation_pairs: Optional[List[List[str]]] = None


@dataclass
class ObjectProfileConfig:
    run: bool = False
    n_workers: int = field(default_factory=default_n_workers)
    configs: List[ObjectProfileEntry] = field(default_factory=list)


@dataclass
class InferenceReductionConfig:
    """Optional dimensionality reduction / cluster prediction (per block).

    enabled: the Dimension-reduction group — write reduction_<method>
        tables. `reducer` takes one or MORE pre-fitted reducer pickles
        (pca / umap / pacmap / localmap; the type of each is detected when
        the microModel config is built, and they run in listed order);
        with no reducer, `method` lists the methods to fit fresh — an
        EMPTY list means nothing is fitted, null keeps the legacy default
        ([pca, umap] when enabled, [pca] for a cluster-only run). When
        reducers are given, the `method` selection is ignored entirely
        (transform only).
    cluster_enabled + cluster: the Cluster group — predict-only clustering
        from a baseline cluster.pkl (its stored kNN models label every
        object, cluster IDs stay baseline-aligned) writing the find_cluster
        table. Runs only when BOTH are set.
    color_by / sample_per_class / method / cluster_res: no-widget YAML keys,
        round-tripped verbatim. color_by and sample_per_class only shape
        the reducer FIT and the PDF plots (which this pipeline never
        writes) — the reduction_<method> tables always contain EVERY object.
    """

    enabled: bool = False
    reducer: Optional[List[str]] = None
    cluster_enabled: bool = False
    cluster: Optional[str] = None
    method: Optional[List[str]] = None
    color_by: str = "pred_class"
    cluster_res: Optional[List[float]] = None
    sample_per_class: int = 10000


@dataclass
class InferenceEntry:
    """One inference block: a trained microModel bundle applied per object.

    feature/pred_class are output toggles; SSL bundles only support features
    (pred_class is gated in the GUI). pred_prob is always written together
    with pred_class by microModel — no separate toggle. max_value is the
    maximum possible intensity of the input dtype (65535 for 16-bit, 255 for
    8-bit) and is always read from config — it is trusted as-is.
    """

    model: str = ""
    mask_name: Optional[str] = None
    channels: Optional[List[str]] = None
    feature: bool = True
    pred_class: bool = True
    output_db: str = INFER_DB_NAME
    max_value: Optional[float] = None
    reduction: Optional[InferenceReductionConfig] = None


@dataclass
class InferenceConfig:
    run: bool = False
    configs: List[InferenceEntry] = field(default_factory=list)


def resolve_inference_db(entry: InferenceEntry) -> str:
    """Resolve an inference block's output DB file name.

    ``output_db`` may be null in YAML — the default name is ``infer.db``.
    Single source of truth used by the pipeline (steps), the CLI
    completeness check, and the microModel bridge (``db_name`` key).
    """
    return entry.output_db or INFER_DB_NAME


@dataclass
class PipelineConfig:
    image_pattern: Optional[str] = None
    mask_pattern: Optional[str] = None
    image_subdir_pattern: Optional[str] = None

    filter: Optional[List[FilterEntry]] = None

    resize: Optional[ResizeConfig] = None
    zproject: Optional[ZProjectConfig] = None
    basic: Optional[BasicConfig] = None
    tile: Optional[TileConfig] = None

    segment: Optional[SegmentConfig] = None

    image_profile: Optional[ImageProfileConfig] = None
    object_profile: Optional[ObjectProfileConfig] = None

    inference: Optional[InferenceConfig] = None


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict] = None,
) -> PipelineConfig:
    config_dict: Dict = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"config file not found: {path}")
        config_dict = load_yaml(path)
        logger.debug("load_config: loaded %s (%d keys)", path, len(config_dict))
    if overrides:
        logger.debug("load_config: applying %d override keys", len(overrides))
        _deep_merge(config_dict, overrides)
    if not config_dict:
        raise ValueError("no configuration provided. Specify a YAML config file.")
    return _dict_to_config(config_dict)


def _dict_to_config(d: Dict) -> PipelineConfig:
    """Build PipelineConfig from a plain dict with manual validation."""
    cfg = PipelineConfig()
    cfg.image_pattern = d.get("image_pattern")
    cfg.mask_pattern = d.get("mask_pattern")
    cfg.image_subdir_pattern = d.get("image_subdir_pattern")

    if d.get("filter"):
        cfg.filter = [_entry_from_section("filter", FilterEntry, f) for f in d["filter"]]

    for attr in SECTION_ATTRS:
        section = d.get(attr)
        if section:
            setattr(cfg, attr, section_to_dataclass(attr, section))

    return cfg


def _coerce_bool(value: Any, attr: str) -> bool:
    """Strict boolean parse — a config bool must be a real bool.

    Quoted strings like "false" are rejected (no lenient dual-form parsing);
    the user is responsible for writing config values with correct types.
    Without this, a "false" string is truthy in Python and silently ENABLES
    the flag it names.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{attr}' must be a real boolean (true/false), got {value!r}")


def _coerce_bool_fields(attr: str, cls, values: Dict) -> Dict:
    """Strictly validate every bool-typed dataclass field present in values.

    Applied to all sections/entries (not just `run`): a YAML `overwrite_db:
    "false"` must be rejected, not silently treated as enabled.
    """
    for name, f in cls.__dataclass_fields__.items():
        if f.type in (bool, "bool") and name in values and values[name] is not None:
            values[name] = _coerce_bool(values[name], f"{attr}.{name}")
    return values


def _check_keys(attr: str, section: Dict, known: set) -> None:
    """Reject unknown keys in a section dict with a clear message."""
    unknown = set(section) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in '{attr}' section: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )


def _check(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def _check_bare_mask_name(value: Optional[str], context: str) -> None:
    """Mask/object names in configs are BARE (``cell``), never ``mask_cell``.

    The ``mask_`` prefix is microBase's metadata-column convention; config
    values are converted on use. Rejecting the prefixed form here prevents a
    silent "column not found" later (microModel prepends the prefix itself).
    """
    if value and str(value).startswith(MASK_COLUMN_PREFIX):
        raise ValueError(
            f"{context} must be the bare mask/object name without the "
            f"'{MASK_COLUMN_PREFIX}' prefix (got {value!r}; use "
            f"{str(value)[len(MASK_COLUMN_PREFIX):]!r})."
        )


def section_to_dataclass(attr: str, section: Dict) -> Any:
    """Convert a single section dict to its corresponding dataclass instance.

    attr is one of the SECTION_ATTRS. Unknown keys raise ValueError with a
    clear message (no silent dropping); numeric fields are range-validated.
    """
    if attr == "resize":
        cfg = _dataclass_from_section(attr, ResizeConfig, section)
        _check(cfg.scale_factor > 0, f"'resize.scale_factor' must be > 0, got {cfg.scale_factor}")
        return cfg
    if attr == "basic":
        cfg = _dataclass_from_section(attr, BasicConfig, section)
        _check(cfg.mode in _BASIC_MODES,
               f"'basic.mode' must be one of {_BASIC_MODES}, got {cfg.mode!r}")
        _check(cfg.n_image >= 1, f"'basic.n_image' must be >= 1, got {cfg.n_image}")
        _check(cfg.working_size >= 1, f"'basic.working_size' must be >= 1, got {cfg.working_size}")
        return cfg
    if attr == "zproject":
        zpd = section
        if "method" in zpd and isinstance(zpd["method"], str):
            zpd = {**zpd, "method": ProjectionMethod(zpd["method"])}
        return _dataclass_from_section(attr, ZProjectConfig, zpd)
    if attr == "tile":
        cfg = _dataclass_from_section(attr, TileConfig, section)
        _check(cfg.tile_width > 0, f"'tile.tile_width' must be > 0, got {cfg.tile_width}")
        _check(cfg.tile_height > 0, f"'tile.tile_height' must be > 0, got {cfg.tile_height}")
        return cfg
    if attr == "segment":
        _check_keys(attr, section, {"run", "configs"})
        entries = []
        for e in section.get("configs", []):
            entry = _entry_from_section(attr, SegmentEntry, e)
            _check(entry.object_name, "'segment.configs[].object_name' must not be empty")
            _check_bare_mask_name(entry.object_name, "'segment.configs[].object_name'")
            _check(entry.gpu_batch_size >= 1, "'segment.configs[].gpu_batch_size' must be >= 1")
            _check(entry.resize_factor > 0, "'segment.configs[].resize_factor' must be > 0")
            _check(entry.flow_threshold >= 0, "'segment.configs[].flow_threshold' must be >= 0")
            _check(0 <= entry.edge_pixel_ratio <= 1,
                   "'segment.configs[].edge_pixel_ratio' must be in [0, 1]")
            entries.append(entry)
        # Duplicate object names would overwrite each other's mask file — the
        # GUI rejects them; the CLI must too (single validation source).
        names = [e.object_name for e in entries]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(
                f"Duplicate segment.configs[].object_name values: {dups}. "
                f"Each block writes <stem>_cp_masks_<object_name>.png."
            )
        return SegmentConfig(
            run=_coerce_bool(section.get("run", False), "segment.run"),
            configs=entries,
        )
    if attr == "image_profile":
        cfg = _dataclass_from_section(attr, ImageProfileConfig, section)
        if cfg.n_workers is None:
            cfg.n_workers = default_n_workers()
        _check(cfg.n_workers >= 1, f"'image_profile.n_workers' must be >= 1, got {cfg.n_workers}")
        return cfg
    if attr == "object_profile":
        _check_keys(attr, section, {"run", "n_workers", "configs"})
        entries = []
        for e in section.get("configs", []):
            entry = _entry_from_section(attr, ObjectProfileEntry, e)
            _check(entry.mask_name, "'object_profile.configs[].mask_name' must not be empty")
            _check_bare_mask_name(entry.mask_name, "'object_profile.configs[].mask_name'")
            _check_bare_mask_name(entry.parent_mask_name,
                                  "'object_profile.configs[].parent_mask_name'")
            _check(entry.radial_bins >= 1, "'object_profile.configs[].radial_bins' must be >= 1")
            _check(entry.granularity_spectrum_length is None or entry.granularity_spectrum_length >= 1,
                   "'object_profile.configs[].granularity_spectrum_length' must be >= 1")
            _check(entry.glcm_levels is None or entry.glcm_levels >= 2,
                   "'object_profile.configs[].glcm_levels' must be >= 2")
            # Range-validate the granularity/GLCM knobs here rather than per
            # row: the profiler skips failing rows silently, so an
            # out-of-range YAML value would otherwise yield empty tables
            # with no error.
            _check(entry.granularity_subsample_ratio is None or 0 < entry.granularity_subsample_ratio <= 1,
                   "'object_profile.configs[].granularity_subsample_ratio' must be in (0, 1]")
            _check(entry.granularity_background_subsample_ratio is None
                   or 0 < entry.granularity_background_subsample_ratio <= 1,
                   "'object_profile.configs[].granularity_background_subsample_ratio' must be in (0, 1]")
            _check(entry.granularity_background_radius is None or entry.granularity_background_radius >= 1,
                   "'object_profile.configs[].granularity_background_radius' must be >= 1")
            _check(entry.glcm_distances is None
                   or all(d >= 1 for d in entry.glcm_distances),
                   "'object_profile.configs[].glcm_distances' values must be >= 1")
            if entry.parent_mask_name and entry.parent_mask_name == entry.mask_name:
                raise ValueError(
                    "'object_profile.configs[].parent_mask_name' must differ "
                    "from mask_name (a mask cannot be its own parent)."
                )
            entries.append(entry)
        # Duplicate output table names would replace each other (first flush
        # uses if_exists='replace') — validate once here for GUI and CLI.
        table_names = [e.output_table_name or e.mask_name for e in entries]
        dups = sorted({t for t in table_names if t and table_names.count(t) > 1})
        if dups:
            raise ValueError(
                f"Duplicate object_profile output table(s): {dups}. Set a "
                f"distinct output_table_name per block."
            )
        n_workers = section.get("n_workers")
        if n_workers is None:
            n_workers = default_n_workers()
        _check(n_workers >= 1, f"'object_profile.n_workers' must be >= 1, got {n_workers}")
        return ObjectProfileConfig(
            run=_coerce_bool(section.get("run", False), "object_profile.run"),
            n_workers=n_workers,
            configs=entries,
        )
    if attr == "inference":
        _check_keys(attr, section, {"run", "configs"})
        entries = []
        for e in section.get("configs", []):
            entry_dict = dict(e)
            red = entry_dict.pop("reduction", None)
            entry = _entry_from_section(attr, InferenceEntry, entry_dict)
            if entry.model is None or not entry.model:
                raise ValueError("'inference.configs[].model' must not be empty")
            _check_bare_mask_name(entry.mask_name, "'inference.configs[].mask_name'")
            if entry.max_value is not None and entry.max_value <= 0:
                raise ValueError(
                    f"'inference.configs[].max_value' must be > 0, got {entry.max_value}")
            red_obj = (
                _dataclass_from_section(attr, InferenceReductionConfig, red)
                if red else None
            )
            if red_obj is not None and isinstance(red_obj.reducer, str):
                red_obj.reducer = [red_obj.reducer]
            if red_obj is not None and red_obj.method:
                # Reject typos here (not silently filter downstream), so a
                # bad method can never make the completeness check pass
                # without the matching table existing.
                bad = [m for m in red_obj.method
                       if m not in ("pca", "umap", "pacmap", "localmap")]
                if bad:
                    raise ValueError(
                        f"'inference.configs[].reduction.method' has unknown "
                        f"entries {bad}; valid: ['pca', 'umap', 'pacmap', 'localmap']")
            entry.reduction = red_obj
            if red_obj is not None:
                # A checked Dimension-reduction group only runs when reducers
                # or methods are configured; an empty method list means
                # "nothing fitted" and needs no features. The Cluster group
                # needs its cluster.pkl (and the feature BLOB for its PCA
                # reference).
                dr_will_run = bool(
                    red_obj.enabled
                    and (red_obj.reducer
                         or red_obj.method is None
                         or red_obj.method))
                red_runs = bool(dr_will_run
                                or (red_obj.cluster_enabled and red_obj.cluster))
                if red_runs and not entry.feature:
                    raise ValueError(
                        "'inference.configs[].reduction' requires feature=true "
                        "(reduction reads the features BLOB)."
                    )
            entries.append(entry)
        # Duplicate output_db names would overwrite each other's inference
        # rows — reject in both GUI and CLI (single validation source).
        db_names = [resolve_inference_db(e) for e in entries if e.channels]
        dups = sorted({n for n in db_names if db_names.count(n) > 1})
        if dups:
            raise ValueError(
                f"Duplicate inference output_db name(s): {dups}. Rename one "
                f"of the blocks."
            )
        return InferenceConfig(
            run=_coerce_bool(section.get("run", False), "inference.run"),
            configs=entries,
        )
    raise ValueError(f"Unknown config section: {attr!r}")


def _dataclass_from_section(attr: str, cls, section: Dict):
    """Instantiate a dataclass from a section dict, rejecting unknown keys.

    Strict config typing: empty lists normalize to None (``[] in yml -> None``)
    and a ``run`` flag must be a real bool.
    """
    known = set(cls.__dataclass_fields__)
    unknown = set(section) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in '{attr}' section: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    coerced = {
        k: (None if isinstance(v, list) and not v else v)
        for k, v in section.items()
    }
    coerced = _coerce_bool_fields(attr, cls, coerced)
    if "run" in coerced:
        coerced["run"] = _coerce_bool(coerced["run"], f"{attr}.run")
    return cls(**coerced)


def _entry_from_section(attr: str, cls, entry: Dict):
    """Instantiate a block-list entry dataclass, rejecting unknown keys.

    Empty lists normalize to None (``[] in yml -> None``), so optional
    channel lists are either a real list or None — never a stale [].
    """
    known = set(cls.__dataclass_fields__)
    unknown = set(entry) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in '{attr}.configs' entry: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    coerced = {
        k: (None if isinstance(v, list) and not v else v)
        for k, v in entry.items()
    }
    return cls(**_coerce_bool_fields(f"{attr}.configs", cls, coerced))


def config_to_dict(cfg: PipelineConfig) -> Dict:
    """Serialize PipelineConfig back to a plain dict (for session.yml)."""
    import dataclasses
    result: Dict = {}
    if cfg.image_pattern is not None:
        result["image_pattern"] = cfg.image_pattern
    if cfg.mask_pattern is not None:
        result["mask_pattern"] = cfg.mask_pattern
    if cfg.image_subdir_pattern is not None:
        result["image_subdir_pattern"] = cfg.image_subdir_pattern

    # `filter` is always written (even as []): SessionFile.save deep-merges,
    # so an absent key would leave a stale GUI-written filter in session.yml.
    result["filter"] = [dataclasses.asdict(f) for f in cfg.filter] if cfg.filter else []

    for attr in SECTION_ATTRS:
        val = getattr(cfg, attr)
        if val is not None:
            d = dataclasses.asdict(val)
            if attr == "zproject" and "method" in d:
                d["method"] = d["method"].value if hasattr(d["method"], "value") else d["method"]
            result[attr] = d
    return result


def _deep_merge(base: Dict, overrides: Dict) -> None:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
