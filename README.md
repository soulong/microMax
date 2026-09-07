# microMax — Microscopy Suite

`microMax` is a four-package Python suite for end-to-end microscopy image
analysis: preprocessing, cell segmentation, feature profiling, interactive
viewing/annotation, self-supervised pretraining, classification, and feature
extraction.

```
microMax/
├── microBase/       shared foundation (no internal deps)
├── microProfiler/   preprocessing + segmentation + profiling pipeline (CLI + Qt GUI)
├── microVis/        interactive Qt viewer + annotation + export (GUI)
└── microModel/      SSL pretrain / train / infer (CLI + Flask viewer)
```

| Package | Version | Console script |
|---|---|---|
| microBase | 0.9.1 | — (library) |
| microProfiler | 1.6.3 | `microprofiler` |
| microVis | 1.2.1 | `microvis` |
| microModel | 0.9.0 | `micromodel` |

- `microBase` is the only shared dependency — the three consumers never
  import each other (the one documented exception: microProfiler lazily
  imports microModel for its optional per-object inference step).
- The three tools talk to each other through on-disk artifacts: TIFFs,
  masks, `result.db`, `infer.db`, and per-dataset `session.yml`.

---

## Installation

### Environment

The suite is installed into a conda environment named `micro`.

```
conda create -n micro python=3.12
conda activate micro
```

install a CUDA-enabled PyTorch build and make sure your NVIDIA drivers are up to date.

```
# for windows
pip install torch --index-url https://download.pytorch.org/whl/cu130

# for linux 
pip install torch torchvision
```

### Install Cellpose and dependent DINOv3

`microProfiler` segmentation is built on [Cellpose](https://github.com/MouseLand/cellpose).
The DINOv3-based models (`cpdino`, `cpdino-vitb`) require extra DINOv3 (Cellpose v4.2+, June 2026)

```
pip install cellpose[gui]
python -m pip install git+https://github.com/facebookresearch/dinov3
```

### Installation

Install the meta packages in dependency order:

```
git clone https://github.com/soulong/microMax.git
cd microMax

python -m pip install -e microBase
python -m pip install -e microModel
python -m pip install -e microProfiler
python -m pip install -e microVis
```


Notes:

- Cellpose model weights download automatically on first use (from HuggingFace).
- All Cellpose models are trained on data licensed **CC-BY-NC** — check the license before commercial use.
- See the Cellpose docs (https://cellpose.readthedocs.io) for GPU setup, fine-tuning, and troubleshooting.

---

## Typical usage

### microProfiler — profiling pipeline

**GUI** (recommended for interactive work):

```
microprofiler
```

Five-page flow: **Input → Pre-process → Segment → Profile → Inference**.

1. **Input** — Browse to a dataset folder, then press **Load Dataset**.
   The image/mask filename patterns are matched against your file naming.
   Optional filters narrow the dataset by metadata column (e.g. well).
2. **Pre-process** (optional) — enable any of: resize, Z-projection (needs a
   `stack` metadata column), BaSiC illumination correction, tiling (needs a
   `field` column). Run with **Run Preprocessing**. Steps run only when
   enabled — nothing runs by default.
3. **Segment** — configure one or more Cellpose runs (object name, model,
   channels, diameter, thresholds), then **Run Segmentation**. Masks are
   written next to the images as `<stem>_cp_masks_<name>.png`.
4. **Profile** — choose image-level intensity features and per-object
   features (shape, intensity, radial profiles, granularity, GLCM,
   correlations), then **Run Profiling**. Results go into
   `<dataset>/result.db` (tables `image`, plus one table per object type).
   The image-profiling step requires at least one selected channel — an
   empty selection skips the step.
5. **Inference** (optional, needs `microModel` installed) — add one block
   per trained model bundle, select the mask and channels, then **Run
   Inference**. Writes `<dataset>/<output_db>` (e.g. `infer.db`) with
   predictions/features, plus optional PCA/UMAP reduction tables.

**CLI** (same pipeline, headless — useful for batch/plate processing):

```
microprofiler run --config pipeline_config.yml --dataset-dir D:\data\plate1
```

The CLI config mirrors the GUI exactly: every step section has a `run`
flag (default `false`), so a minimal YAML never silently runs a destructive
in-place step.

### microVis — interactive viewer & annotation

```
microvis            # then select a dataset folder in the UI
microvis D:\data\plate1
```

Typical flow:

1. **Select Dataset Directory** (Browse), then **Load Dataset** to scan the
   images and load their profiling results from `result.db`.
2. Inspect images: thumbnail grid, well-plate grid, filters on well/field/
   stack/timepoint and any extra metadata columns, per-channel color and
   contrast controls, full-resolution zoom, pixel readout.
3. Pick a mask from the **Select object** dropdown (Object Overlay panel) to
   see cell outlines and hover over individual objects.
4. **Annotate**: drag objects onto class boxes to label them. Labels can be
   written back to `result.db`.
5. **Export**: exports the selected class's objects as multi-channel cell
   TIFFs (ImageJ-compatible, one file per cell) plus a `{mask_name}.csv`
   manifest — ready to feed `microModel` training.

### microModel — SSL pretrain / train / infer

All commands are config-driven (`micromodel <subcommand> --config <file>`):

```
micromodel pretrain --config configs/pretrain_dinov3_phase1.yml  # SSL backbone (DINOv3)
micromodel vis-augment --config configs/pretrain_dinov3_phase1.yml # preview the augmentation views
micromodel train --config configs/train_from_pretrain.yml   # classifier (from SSL backbone or scratch)
micromodel infer --config configs/infer_whole_image.yml     # predictions + features -> infer.db
micromodel vis-reduction --config configs/infer_whole_image.yml
micromodel vis-reduction-interactive --config configs/infer_whole_image.yml --port 5000
```

The end-to-end flow: **pretrain → train → infer**, with the interactive
PCA/UMAP viewer (`vis-reduction-interactive`, served at
`http://127.0.0.1:5000`) for clicking through individual cells. Training
inputs can be the exported cells from microVis, or whole images with their
segmentation masks.

---

## Important things to know for accurate use

### Filename patterns are regexes — get them right

`image_pattern` / `mask_pattern` are regular expressions that must match
your microscope's naming convention, with named capture groups such as
`row`, `col`, `field`, `stack`, `channel`, `timepoint` (Operetta-style
defaults are pre-filled in the GUI). Structural columns absent from the
pattern are simply missing — they are never defaulted. Pattern edits require
**Load Dataset** again to take effect.

### Preprocessing overwrites your source files

Resize, Z-projection, BaSiC correction, and tiling modify TIFFs **in place**.
The suite remembers which steps already ran (per-dataset `applied_steps`) and
skips them on re-runs — you cannot accidentally double-apply a step. If an
in-place step is interrupted, restore the original raw files before
re-running. Tiling writes only complete tiles: right/bottom remainders (and
images smaller than the tile size) are intentionally dropped — pick tile
sizes that divide your image dimensions, or that data is lost.

### Upgrading from the old pre-processing order

Older versions ran BaSiC **before** Z-projection. If a dataset was already
processed under the old order and only `basic` was recorded in
`applied_steps`, upgrading will run Z-projection on the already-BaSiC-corrected
z-stacks and skip BaSiC — a different result from a fresh run of both steps
in the new order. To get the new-order behavior cleanly on such a dataset,
restore the original raw files and re-run preprocessing from scratch.

### session.yml — per-dataset state

Each dataset keeps a `session.yml` (patterns, step parameters, applied steps,
channel colors). It is read once on the first Browse of a folder; after that,
your GUI edits are preserved. Action buttons (Load Dataset, Run, Apply,
Export, …) save current state back to it. The full **Reset** button returns
the GUI to its initial state.

### No database migrations — delete to redo

`result.db` (profiling) and `infer.db` (predictions/features) have no schema
versioning. To re-process or to change a table's structure, delete the DB
file (or the affected table) and re-run. The microProfiler CLI also skips
datasets it considers complete — delete `result.db` to force reprocessing.

### Segmentation & profiling are safe to re-run

Unlike preprocessing, segmentation and profiling never modify source images.
Re-running them is intended (e.g. after tuning model parameters or adding a
new object type); use the per-entry **Overwrite mask** / **Overwrite DB**
toggles to control whether existing outputs are replaced.

### Metadata is text — always

Everything captured from filenames is treated as plain text end-to-end
(`"01"` stays `"01"`, no numeric coercion), so metadata is preserved
verbatim from extraction to the profiling DB, CSV exports, and `infer.db`.

### microModel: config vs. bundle metadata

- The model input size is defined by the augmentation steps in the config —
  there is no separate input-size parameter.
- Inference always uses the **normalization and augmentation settings baked
  into the model bundle** at training time — do not expect config files to
  override them.
- Whole-image inference can use one-channel-per-file even when the model was
  trained on multi-channel cell images.
- If a dataset was processed and then deleted/recreated, stale rows in
  `infer.db` can make the interactive viewer fail for individual cells —
  re-run inference to refresh the DB.

---

## Artifact quick-reference

| Artifact | Location | Produced by |
|---|---|---|
| Profiling DB | `<dataset>/result.db` | microProfiler (image + object tables) |
| Inference DB | `<dataset>/infer.db` or `{output_dir}/infer.db` | microModel (CLI); microProfiler inference step (always `<dataset>/<output_db>`) |
| Per-dataset state | `<dataset>/session.yml` | microProfiler / microVis |
| Cellpose masks | `<stem>_cp_masks_<name>.png` next to images | microProfiler |
| BaSiC shading models | `<dataset>/.microprofiler/BaSiC_model/` | microProfiler |
| Exported cells | `<dataset>/objects_exported/<class>/` + `{mask_name}.csv` | microVis |
| SSL bundles | `runs/model.pt`, `model_<epoch>.pt` | microModel pretrain |
| Train bundles | `runs/model.pt`, `model_<epoch>.pt` | microModel train |
| Reports & logs | `runs/*.pdf`, `runs/*.txt`, `runs/micromodel.log` | microModel |
