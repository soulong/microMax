from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from microBase import load_yaml

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
    run: bool = False
    scale_factor: float = 1.0


@dataclass
class BasicConfig:
    run: bool = False
    mode: str = "fit-transform"
    n_image: int = 100
    working_size: int = 64
    enable_darkfield: bool = False


@dataclass
class ZProjectConfig:
    run: bool = False
    method: ProjectionMethod = ProjectionMethod.max


@dataclass
class TileConfig:
    run: bool = False
    tile_width: int = 1024
    tile_height: int = 1024


@dataclass
class SegmentEntry:
    object_name: str = "cell"
    model_name: str = "cpdino"
    chan1: List[str] = field(default_factory=list)
    chan2: Optional[List[str]] = None
    merge1: str = "mean"
    merge2: str = "mean"
    resize_factor: float = 0.5
    diameter: Optional[float] = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu_batch_size: int = 32
    overwrite_mask: bool = False


@dataclass
class SegmentConfig:
    run: bool = False
    configs: List[SegmentEntry] = field(default_factory=list)


@dataclass
class ImageProfileConfig:
    run: bool = False
    n_workers: int = field(default_factory=default_n_workers)
    image_channels: Optional[List[str]] = None
    image_thresholds: Optional[Dict[str, float]] = None


@dataclass
class ObjectProfileEntry:
    mask_name: Optional[str] = None
    parent_mask_name: Optional[str] = None
    output_table_name: Optional[str] = None
    overwrite_db: bool = False
    intensity_channels: Optional[List[str]] = None
    radial_channels: Optional[List[str]] = None
    radial_bins: int = 4
    gran_channels: Optional[List[str]] = None
    gran_spectrum_length: Optional[int] = None
    gran_subsample_ratio: Optional[float] = None
    gran_background_subsample_ratio: Optional[float] = None
    gran_background_radius: Optional[int] = None
    glcm_channels: Optional[List[str]] = None
    glcm_distances: Optional[List[int]] = None
    glcm_levels: Optional[int] = None
    correlation_pairs: Optional[List[List[str]]] = None

    def resolved(self) -> "ResolvedProfiling":
        gran_spectrum_length = self.gran_spectrum_length if self.gran_spectrum_length is not None else 8
        gran_subsample_ratio = self.gran_subsample_ratio if self.gran_subsample_ratio is not None else 0.5
        gran_background_subsample_ratio = self.gran_background_subsample_ratio if self.gran_background_subsample_ratio is not None else 0.25
        gran_background_radius = self.gran_background_radius if self.gran_background_radius is not None else 10

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
            granularity_channels=self.gran_channels,
            gran_spectrum_length=gran_spectrum_length,
            gran_subsample_ratio=gran_subsample_ratio,
            gran_background_subsample_ratio=gran_background_subsample_ratio,
            gran_background_radius=gran_background_radius,
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
    gran_spectrum_length: int = 8
    gran_subsample_ratio: float = 0.5
    gran_background_subsample_ratio: float = 0.25
    gran_background_radius: int = 10
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
    """Optional dimensionality reduction after inference (per inference block).

    method lists the DR methods to run (subset of pca/umap/pacmap/localmap;
    null = microModel default [pca, umap]). cluster_k lists KMeans cluster
    counts for the find_cluster table (null/[] = no cluster finding).
    reduction_<method> take pre-fitted reducer pickles (transform directly,
    no refit); cluster takes a pre-fitted cluster.pkl whose stored KMeans
    models predict every stored k (no refit, cluster_k ignored).
    """

    enabled: bool = False
    method: Optional[List[str]] = None
    color_by: str = "pred_class"
    cluster: Optional[str] = None
    cluster_k: Optional[List[int]] = None
    sample_per_class: int = 10000
    reduction_pca: Optional[str] = None
    reduction_umap: Optional[str] = None
    reduction_pacmap: Optional[str] = None
    reduction_localmap: Optional[str] = None


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
    output_db: str = "infer.db"
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
    return entry.output_db or "infer.db"


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
    """Strict boolean parse — a YAML 'run' flag must be a real bool.

    Quoted strings like "false" are rejected (no lenient dual-form parsing);
    the user is responsible for writing config values with correct types.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{attr}' must be a real boolean (true/false), got {value!r}")


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
            _check(entry.gpu_batch_size >= 1, "'segment.configs[].gpu_batch_size' must be >= 1")
            _check(entry.resize_factor > 0, "'segment.configs[].resize_factor' must be > 0")
            _check(entry.flow_threshold >= 0, "'segment.configs[].flow_threshold' must be >= 0")
            entries.append(entry)
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
            _check(entry.radial_bins >= 1, "'object_profile.configs[].radial_bins' must be >= 1")
            _check(entry.gran_spectrum_length is None or entry.gran_spectrum_length >= 1,
                   "'object_profile.configs[].gran_spectrum_length' must be >= 1")
            _check(entry.glcm_levels is None or entry.glcm_levels >= 2,
                   "'object_profile.configs[].glcm_levels' must be >= 2")
            entries.append(entry)
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
            if entry.max_value is not None and entry.max_value <= 0:
                raise ValueError(
                    f"'inference.configs[].max_value' must be > 0, got {entry.max_value}")
            red_obj = (
                _dataclass_from_section(attr, InferenceReductionConfig, red)
                if red else None
            )
            entry.reduction = red_obj
            entries.append(entry)
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
    return cls(**{
        k: (None if isinstance(v, list) and not v else v)
        for k, v in entry.items()
    })


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
