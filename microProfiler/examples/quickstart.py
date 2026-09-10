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
OUTPUT_DB = DATA_DIR / "profiler.db"
MODEL_PATH = Path(r"/path/to/model.pt")
INFER_DB = DATA_DIR / "infer.db"

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)(?:sk|t)(?P<timepoint>\d+).*\.tiff"
)
MASK_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)(?:sk|t)(?P<timepoint>\d+).*_cp_masks_(?P<mask_name>.+)\.png"
)

ds = ImageDataset(DATA_DIR, image_pattern=IMAGE_PATTERN, mask_pattern=MASK_PATTERN)

ds = resize_dataset(ds, scale_factor=0.5)
ds = z_project_dataset(ds, method="max")
ds = apply_basic(ds, mode="fit-transform")
ds = tile_dataset(ds, tile_width=1024, tile_height=1024)
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
        granularity_channels=ds.intensity_colnames,
        glcm_channels=ds.intensity_colnames,
        glcm_distances=[2],
        correlation_pairs=[[ds.intensity_colnames[0], ds.intensity_colnames[1]]]
        if len(ds.intensity_colnames) >= 2 else None,
    ),
    mask_name="cell",
    db_path=OUTPUT_DB,
)

print(f"\nDone! Results written to: {OUTPUT_DB}")

# --- Inference (requires a trained microModel .pt bundle) ---
# The model path is faked here — replace MODEL_PATH with a real trained
# bundle to run. microModel is imported lazily (the suite's convention:
# microProfiler runs fully without microModel installed).
try:
    from microModel.infer import run_inference
except ImportError:
    run_inference = None
mm_cfg = {
    "mode": "whole_image",
    "model": str(MODEL_PATH),
    "output_dir": None,
    "dataloader": {"num_workers": 4, "prefetch_factor": 2, "persistent_workers": False},
    "data": {
        "root": [str(DATA_DIR)],
        "channels": None,
        "channel_layout": None,
        "image_pattern": IMAGE_PATTERN.pattern,
        "mask_pattern": MASK_PATTERN.pattern,
        "image_subdir_pattern": None,
        "mask_name": "cell",
        "max_value": 65535.0,
        "label_from_dir": False,
        "label_csv": None,
        "sample_max": None,
        "sample_by": "per_dataset",
    },
    "inference": {
        "pred_class": True,
        "feature": True,
        "db_name": str(INFER_DB),
        "batch_size": 128,
    },
}
if run_inference is None:
    print("microModel not installed — skipping inference (install microModel to run it).")
else:
    run_inference(mm_cfg)
    print(f"Inference results written to: {INFER_DB}")


# Equivalent config-driven pipeline (see examples/pipeline_config.yml)
if __name__ == "__main__":
    from microProfiler.config import load_config, InferenceConfig, InferenceEntry
    from microProfiler.pipeline import run_pipeline

    cfg = load_config("examples/pipeline_config.yml")
    cfg.inference = InferenceConfig(
        run=True,
        configs=[
            InferenceEntry(
                model=str(MODEL_PATH),
                mask_name="cell",
                max_value=65535.0,
            )
        ],
    )
    ds, applied_steps = run_pipeline(cfg, root_dir=DATA_DIR)
