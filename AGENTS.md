# AGENTS.md — Microscopy Suite

High-level map for the four-package suite under `microMax/`. Read it before
adding, removing, or changing features — it explains what each package does,
how they depend on each other, and where the seams live.

> This file deliberately keeps ONLY the overall design framework. Specific
> parameters, rules, constraints, inputs, and outputs change constantly as
> the code evolves — always refer to the current code for details.

```
microMax/
├── microBase/        ← shared foundation (no internal deps)
├── microProfiler/    ← preprocessing + segmentation + profiling pipeline (CLI + Qt GUI)
├── microVis/         ← interactive Qt viewer + annotation + export (GUI)
└── microModel/       ← SSL pretrain / train / infer (CLI + Flask viewer)
```

***

## 1. Running environment

| Item      | Value               |
| --------- | ------------------- |
| OS        | Windows, PowerShell |
| Python    | `>=3.10`            |
| Conda env | `micro`             |

Install the packages bottom-up with editable installs:
`microBase` → `microProfiler` → `microVis` → `microModel`.

Console scripts after install:

* `microprofiler` → `microProfiler.__main__:main`

* `microvis` → `microVis.__main__:main`

* `micromodel` → `microModel.cli.main:main`

***

## 2. Inter-package dependency graph

* `microBase` is the only package the three consumers (`microProfiler`,
  `microVis`, `microModel`) import. `microModel` additionally depends on
  `lightly` (SSL heads/losses), `timm` (backbones), `pacmap`
  (PaCMAP/LocalMAP dimensionality reduction) and `leidenalg`/`igraph`
  (Leiden cluster finding on a kNN graph).

* The three consumers never import each other directly, with ONE exception:
  `microProfiler` lazily imports `microModel` in its inference step.
  microProfiler runs fully without microModel installed; requesting
  inference without it is a hard error.

* The tools talk to each other through **on-disk artifacts**: TIFF images,
  segmentation masks, `result.db` (profiling, owned by microProfiler, read
  by microVis), `infer.db` (predictions/features, owned by microModel), and
  per-dataset `session.yml` (deep-merged by microProfiler and microVis so
  each tool owns its own keys; microModel does NOT use session.yml — it
  reads patterns from its own config files).

* `microProfiler` produces the dataset (TIFFs, masks, `result.db`,
  `session.yml`) that `microVis` and `microModel` consume. The tools may
  also be used independently: microModel accepts pre-cropped single-cell
  folders (e.g. exported by microVis) or whole-image datasets with masks.

***

## 3. Development rules (must follow)

1. **Never touch** **`microProfiler/preprocessing/basic/`** — the BaSiC algorithm
   there is a verbatim vendored third-party implementation. Do not read it,
   do not modify it. The only interface is `basic_correction.py`.
2. **No backward compatibility** — when changing/adding/removing a feature,
   ignore old data and old configs. Breaking them is fine.
3. **Non-public repo, keep logic simple** — no over-complicated conditionals
   or compatibility branches; handle only the regular cases.
4. **Comments are required** — every piece of code, class, and function must
   carry the necessary comments when written.
5. **Ignore type hints** — completely ignore Python type annotations/typing;
   just make the code correct.

***

## 4. microBase — shared foundation

**Path:** `microMax/microBase/` · **Layout:** flat, 10 modules under
`src/microBase/` (no subpackages).

Module map (overview):

| Module             | Purpose                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `__init__.py`      | Re-exports public names; sets env vars disabling albumentations update/telemetry         |
| `schema.py`        | Classifies regex-captured columns (structural vs extra); derives `well` from `row`+`col` |
| `io.py`            | TIFF/mask readers, normalized to `(H, W, C)`; no writers                                 |
| `cells.py`         | Pure functions for labeled masks: single-cell cropping, per-object edge-pixel ratio        |
| `config.py`        | YAML load/save + per-dataset `SessionFile` (deep-merge into `session.yml`)               |
| `augment.py`       | Registry-driven augmentation pipeline over AlbumentationsX                               |
| `normalize.py`     | Per-channel percentile clip + min-max rescale + optional z-score                         |
| `image_dataset.py` | Whole-image loader: regex metadata, masks, LRU cache, cell cropping                      |
| `cell_dataset.py`  | Pre-cropped single-cell TIFF loader (one TIFF per cell)                                  |
| `patterns.py`      | Default regex patterns for common microscope file layouts                                |

Key concepts:

* `ImageDataset` pivots metadata from filename-regex captures (one row per
  site), supports one-channel-per-file and multi-channel-per-file (CHW/HWC)
  layouts, and offers per-site image loading and per-mask cell cropping. Its
  LRU cache is thread-safe and picklable (so torch DataLoader workers can
  spawn it on Windows). Missing image files hard-exit; missing mask paths are
  skipped.

* `CellDataset` loads pre-cropped single cells with channels multiplexed
  inside each file.

* `SessionFile` lives at `<dataset>/session.yml` and deep-merges updates so
  each tool only owns its own keys.

* All metadata is TEXT end-to-end (regex captures used verbatim); structural
  columns absent from the pattern are simply missing, never defaulted.

**Adding a new shared capability:** add a flat module under `src/microBase/`,
re-export its public names in `__init__.py`/`__all__`, add a test under
`tests/`, and bump the version when the surface changed. Put something in
microBase only if at least two consumers need it.

***

## 5. microProfiler — pipeline + Qt GUI

**Path:** `microMax/microProfiler/` · **Entry:** `microprofiler`
(CLI: `run`; GUI: no args).

Package layout (overview):

* `config.py` — `PipelineConfig` dataclasses + strict YAML validation.

* `pipeline/` — step orchestration (`run_pipeline`/`run_step`) +
  `_micromodel_bridge.py` (the only file that imports microModel, for the
  inference step).

* `preprocessing/` — resize, zproject, BaSiC correction (via
  `basic_correction.py`; the algorithm itself lives in the untouchable
  vendored `basic/` folder), tile splitter.

* `segmentation/` — Cellpose-based segmentation (one mask file per object
  type).

* `profiling/` — image-level and object-level feature extraction
  (shape/intensity/radial/granularity/GLCM/correlation) + batch writer.

* `io/` — SQLite wrapper (`Database`) for `result.db`.

* `gui/` — QApplication bootstrap, main window, `PipelineController`,
  step panels (one per pipeline step), background workers, progress.

* `cli.py` — headless run, fully equivalent to the GUI.

The pipeline:

* Steps run in fixed order:
  `resize → zproject → basic → tile → segment → image_profile →
  object_profile → inference`. Step keys ARE the config section names.

* The four preprocessing steps modify source TIFFs **in place** and are gated
  by `applied_steps` in `session.yml` (already-applied steps are skipped).
  segment/profile/inference are non-destructive and always re-runnable.

* GUI and CLI build the same `PipelineConfig` and drive the same
  `run_pipeline` loop, so behaviour is identical. Every section has a `run`
  flag defaulting to `false` — a minimal YAML never silently runs a
  destructive step.

* Config is YAML + strict validation (unknown keys raise an error listing the
  valid keys). Metadata-gated steps require their metadata column (zproject
  needs `stack`, tile needs `field`).

* Progress flows through a `ProgressCollector` (tqdm for CLI, status
  bars/signals for GUI). Workers are cancellable at checkpoints.

* The optional inference step lazily imports microModel to run per-object
  inference and optional DR reduction / cluster prediction (one reducer
  pickle of any DR method, or a baseline cluster.pkl that kNN-predicts the
  find_cluster table), writing a per-block DB under the dataset dir.

* Outputs: in-place processed TIFFs, `<stem>_cp_masks_<obj>.png` masks,
  `result.db` (image + per-object tables), `<dataset>/<output_db>`
  (inference), and `session.yml` (applied steps + patterns).

**Adding a new pipeline step:** add a step module exposing
`step_dataset(ds, **kwargs, progress=...) -> ImageDataset`; add an `XxxConfig`
dataclass + `PipelineConfig` field + section registration in `config.py`; add
a `_run_xxx` helper in `pipeline/steps.py` registered in `_STEP_FUNCTIONS`
(the key is the config section name — preprocessing steps that modify files in
place also join `PREPROC_STEPS`); add a `XxxStepPanel` under `gui/panels/`
(subclass `BaseStepPanel`; block-list panels subclass `BlockContainerPanel`),
register the panel and add a page if it is a new top-level page. GUI Run
buttons all drive `run_pipeline` with a section-restricted config.

***

## 6. microVis — interactive Qt viewer

**Path:** `microMax/microVis/` · **Entry:** `microvis` (GUI only).

Package layout (overview):

* `io/data_module.py` — `DataModule`, the single facade over
  `microBase.ImageDataset` + `result.db`.

* `widgets/` — image display (thumbnail grid + full-res view), channel
  controls, image filters, well-grid canvas, label annotation panel, pixel
  info, data view.

* `processing/` — multi-channel compositing, contrast, mask overlay.

* `worker.py` — background `QRunnable` workers (thumbnail, full-res, crop,
  object export, dataset load).

* `main_window.py` — `MainWindow`, all signal wiring and the label-annotation
  state.

Data flow:

* All GUI modules go through the `DataModule` facade — widgets never
  instantiate `ImageDataset` directly.

* MainWindow owns one `DataModule`, one `QThreadPool`, and the annotation
  state. Heavy work runs in `QRunnable` workers that emit Qt signals back;
  results are guarded by a generation counter so stale results are silently
  discarded.

* `session.yml` is read once on the first Browse of a directory; action
  buttons (Load Dataset, Export, ...) persist GUI state back to it. Channel
  colors are restored from the GUI on reload, not re-read from `session.yml`.

* Object identity throughout display/annotation/export uses the metadata row
  index (`row_idx`), which works for any metadata combination (including
  non-standard datasets without well/field/stack/timepoint).

**Adding a new widget:** add the class under `widgets/`, instantiate it in
`MainWindow.__init__` and wire its signals to private `_on_*` handlers; access
data through `self._dm`; wrap heavy work in a `QRunnable` under `worker.py`.

***

## 7. microModel — SSL pretrain + train + infer

**Path:** `microMax/microModel/` · **Entry:** `micromodel` (CLI subcommands:
`pretrain`, `train`, `infer`, `vis-augment`, `vis-reduction`,
`vis-reduction-interactive`, `vis-attention`).

Additional deps: `lightly` (SSL projection heads/losses), `timm` (backbones +
ViT rebuild), `pacmap` (PaCMAP/LocalMAP DR).

Package layout (overview):

* `cli/main.py` — argparse dispatch to the subcommands.

* `pretrain.py` / `train.py` / `infer.py` — the three stage entries.

* `backbone.py` — backbone builders (incl. DINOv3 ViT), loss, head,
  model + bundle loaders.

* `dataset.py` — PyTorch Datasets: SSL multi-view, single-cell, whole-image.

* `models/` — SSL method implementations: DINOv3 (lightly + own
  DINO/iBOT/KoLeo/Gram components), behind a small registry.

* `vis.py` — augmentation preview, multi-method DR reduction plots
  (pca/umap/pacmap/localmap) + optional Leiden cluster finding (kNN graph,
  one partition per resolution; a baseline cluster.pkl kNN-predicts new
  datasets).

* `vis_interactive.py` — Flask server for interactive point inspection.

* `utils.py` — logging, seed/device, label resolution, reducer pickling.

Design:

* All subcommands are config-driven (YAML under `configs/`).

* Preprocessing reuses microBase `augment`/`normalize`. Three dataset types:
  SSL multi-view (pretrain), single-cell (train), whole-image + mask (infer).
  Input image size is defined entirely by the augmentation steps — there is
  no separate input-size parameter.

* **Teacher-branch extraction:** all downstream feature consumers (infer,
  train transfer, pretrain UMAP check, attention diagnostics) extract the
  TEACHER backbone by default — the EMA/Polyak-averaged branch official
  DINO-family evaluation uses (`extract_backbone_state_dict` prefix table:
  dinov3 → `teacher_backbone.vit.` / `student_backbone.vit.`).

* Pretrain is a generic SSL loop. Every method's `train_step` returns
  `(loss, {component_name: value})` — the loop aggregates, logs (CSV/TB),
  and persists component/monitor histories generically, so adding a loss or
  head never touches the loop. It dispatches by method through the registry;
  saves model bundles atomically and supports resume (`continue` = exact
  extension, `transfer` = fresh run with pretrained weights).

* Train builds a classifier on top of a backbone and saves train bundles +
  plots/reports. Two config profiles: `train_from_scratch.yml` (random or
  ImageNet init) and `train_from_pretrain.yml` (resume.ssl_model backbone
  transfer, `freeze_backbone: true` = head-only linear probe). Labels come
  from `label_csv` (`[filepath, label]`, relative paths resolved against the
  CSV's own directory). The loss follows the label form — no config key:
  plain single labels train FocalLoss, `;`-joined categories train
  multi-label BCELoss; `model.focal_gamma` is the focal exponent in BOTH
  losses (0 = plain CE / plain BCE). `model.label_smoothing` softens targets
  in both modes. Unlabeled records are dropped with a warning; in multi-label
  mode unlisted classes count as negatives (closed-world).

* Infer loads a bundle, dispatches on bundle type (classify vs
  features-only) and mode (single\_cell vs whole\_image), and writes
  `infer.db` (inference + optional reduction tables). Both modes share one
  layout with no threshold: `pred_class`/`pred_prob` hold the
  highest-probability class and its probability, and the full per-class
  vector is stored in fixed-order `prob_<class>` columns (single-label probs
  are a softmax distribution, multi-label independent per-class sigmoids).
  Probability-descending ordering is a display concern of
  vis-reduction-interactive, never baked into the DB.

* Bundles carry their meta (channels, normalization, augmentation);
  inference always uses the settings baked into the bundle at training time.

* The only interactive surface is the Flask viewer
  (`http://127.0.0.1:5000`) for clicking through individual cells in the
  reduction scatter.

**Adding a new SSL method:** create `models/<method>.py` with a model class,
`build_<method>`, and `train_step`; register it in `models/__init__.py`
(`_SSL_REGISTRY`, `get_train_step`, `get_criterion`); add a
`configs/pretrain_<method>.yml` and a `<method>:` config block.

**Adding a new augmentation:** register it in `microBase.augment` and
reference it by name in the config lists — no microModel change needed.

***

## 8. When you change something

* Keep this file in sync with the code's **overall design** (architecture,
  module layout, data flow, dependency graph). Remember it describes only the
  design framework — don't stuff new detailed rules/parameters here.

* Added a new module → update the relevant package's overview section.

* Added a console subcommand → update the CLI entries + `pyproject.toml`
  `[project.scripts]`.

