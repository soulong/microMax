from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from microBase import load_yaml

logger = logging.getLogger(__name__)


class ProjectionMethod(str, Enum):
    max = "max"
    mean = "mean"
    min = "min"


@dataclass
class FilterEntry:
    column: str = ""
    pattern: str = ""


@dataclass
class ResizeConfig:
    run: bool = True
    scale_factor: float = 1.0


@dataclass
class BasicConfig:
    run: bool = True
    mode: str = "fit-transform"
    n_image: int = 100
    working_size: int = 64
    enable_darkfield: bool = False


@dataclass
class ZProjectConfig:
    run: bool = True
    method: ProjectionMethod = ProjectionMethod.max


@dataclass
class TileConfig:
    run: bool = True
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
    run: bool = True
    configs: List[SegmentEntry] = field(default_factory=list)


@dataclass
class ImageProfileConfig:
    run: bool = True
    n_workers: int = field(
        default_factory=lambda: max(1, (os.cpu_count() or 1) // 2)
    )
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

        if self.glcm_distances:
            glcm_distances = self.glcm_distances
        if self.glcm_levels is not None:
            glcm_levels = self.glcm_levels

        radial_bins = self.radial_bins

        return ResolvedProfiling(
            mask_name=self.mask_name,
            parent_mask_name=self.parent_mask_name,
            output_table_name=self.output_table_name,
            intensity_channels=self.intensity_channels,
            radial_channels=self.radial_channels,
            radial_bins=radial_bins,
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
    run: bool = True
    n_workers: int = field(
        default_factory=lambda: max(1, (os.cpu_count() or 1) // 2)
    )
    configs: List[ObjectProfileEntry] = field(default_factory=list)


@dataclass
class PipelineConfig:
    image_pattern: Optional[str] = None
    mask_pattern: Optional[str] = None
    image_subdir_pattern: Optional[str] = None

    filter: Optional[List[FilterEntry]] = None

    resize: Optional[ResizeConfig] = None
    basic: Optional[BasicConfig] = None
    zproject: Optional[ZProjectConfig] = None
    tile: Optional[TileConfig] = None

    segment: Optional[SegmentConfig] = None

    image_profile: Optional[ImageProfileConfig] = None
    object_profile: Optional[ObjectProfileConfig] = None


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict] = None,
) -> PipelineConfig:
    config_dict: Dict = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            print(f"Error: config file not found: {path}")
            import sys
            sys.exit(1)
        config_dict = load_yaml(path)
        logger.debug("load_config: loaded %s (%d keys)", path, len(config_dict))
    if overrides:
        logger.debug("load_config: applying %d override keys", len(overrides))
        _deep_merge(config_dict, overrides)
    if not config_dict:
        print("Error: no configuration provided. Specify a YAML config file.")
        import sys
        sys.exit(1)
    return _dict_to_config(config_dict)


def _dict_to_config(d: Dict) -> PipelineConfig:
    """Build PipelineConfig from a plain dict with manual validation."""
    cfg = PipelineConfig()
    cfg.image_pattern = d.get("image_pattern")
    cfg.mask_pattern = d.get("mask_pattern")
    cfg.image_subdir_pattern = d.get("image_subdir_pattern")

    if d.get("filter"):
        cfg.filter = [FilterEntry(**f) for f in d["filter"]]

    for attr in ("resize", "basic", "zproject", "tile",
                 "segment", "image_profile", "object_profile"):
        section = d.get(attr)
        if section:
            setattr(cfg, attr, section_to_dataclass(attr, section))

    return cfg


def section_to_dataclass(attr: str, section: Dict) -> Any:
    """Convert a single section dict to its corresponding dataclass instance.

    attr is one of: resize, basic, zproject, tile, segment,
    image_profile, object_profile.
    """
    if attr == "resize":
        return ResizeConfig(**section)
    if attr == "basic":
        return BasicConfig(**section)
    if attr == "zproject":
        zpd = section
        if "method" in zpd and isinstance(zpd["method"], str):
            zpd = {**zpd, "method": ProjectionMethod(zpd["method"])}
        return ZProjectConfig(**zpd)
    if attr == "tile":
        return TileConfig(**section)
    if attr == "segment":
        entries = [SegmentEntry(**e) for e in section.get("configs", [])]
        return SegmentConfig(run=section.get("run", True), configs=entries)
    if attr == "image_profile":
        return ImageProfileConfig(**section)
    if attr == "object_profile":
        entries = [ObjectProfileEntry(**e) for e in section.get("configs", [])]
        n_workers = section.get("n_workers")
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) // 2)
        return ObjectProfileConfig(
            run=section.get("run", True), n_workers=n_workers, configs=entries)
    raise ValueError(f"Unknown config section: {attr!r}")


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

    if cfg.filter:
        result["filter"] = [dataclasses.asdict(f) for f in cfg.filter]

    for attr in ("resize", "basic", "zproject", "tile",
                 "segment", "image_profile", "object_profile"):
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
