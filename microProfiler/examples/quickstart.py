#!/usr/bin/env python3
"""Quick-start example for the microProfiler library."""

from pathlib import Path

import re

from microProfiler import ImageDataset
from microProfiler.config import ObjectProfileEntry
from microProfiler.preprocessing.resizer import resize_dataset
from microProfiler.preprocessing.basic_correction import apply_basic
from microProfiler.preprocessing.z_projection import z_project_dataset
from microProfiler.preprocessing.tile_splitter import tile_dataset
from microProfiler.segmentation.cellpose import segment_dataset
from microProfiler.profiling.image_profiler import profile_images
from microProfiler.profiling.object_profiler import profile_objects

DATA_DIR = Path(r"/path/to/unified/images")
OUTPUT_DB = DATA_DIR / "result.db"

IMAGE_PATTERN = re.compile(
    r"(?P<well>[A-Z]\d+)_f(?P<field>\d+)_z(?P<stack>\d+)_t(?P<timepoint>\d+)_ch(?P<channel>\d+)\.tiff$"
)
MASK_PATTERN = re.compile(
    r"(?P<well>[A-Z]\d+)_f(?P<field>\d+)_z(?P<stack>\d+)_t(?P<timepoint>\d+)_ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png$"
)

ds = ImageDataset(DATA_DIR, image_pattern=IMAGE_PATTERN, mask_pattern=MASK_PATTERN)

ds = resize_dataset(ds, scale_factor=0.5)
ds = apply_basic(ds, mode="fit-transform")
ds = z_project_dataset(ds, method="max")
print(ds)

ds = tile_dataset(ds, tile_w=1024, tile_h=1024)
print(ds)

ds = segment_dataset(ds, object_name="cell", chan1=ds.intensity_colnames[:1])
print(ds)

profile_images(ds, channels=ds.intensity_colnames, db_path=OUTPUT_DB)

profile_objects(
    ds,
    obj_config=ObjectProfileEntry(
        mask_name="cell",
        intensity_channels=ds.intensity_colnames,
        radial_channels=ds.intensity_colnames,
        radial_bins=4,
        gran_channels=ds.intensity_colnames,
        glcm_channels=ds.intensity_colnames,
        glcm_distances=[2],
        correlation_pairs=[[ds.intensity_colnames[0], ds.intensity_colnames[1]]]
        if len(ds.intensity_colnames) >= 2 else None,
    ),
    db_path=OUTPUT_DB,
)

print(f"\nDone! Results written to: {OUTPUT_DB}")


# Equivalent config-driven pipeline (see examples/pipeline_config.yml)
if __name__ == "__main__":
    from microProfiler.config import load_config
    from microProfiler.pipeline import run_pipeline

    cfg = load_config("examples/pipeline_config.yml")
    ds, applied_steps = run_pipeline(cfg, dataset_dir=DATA_DIR)
