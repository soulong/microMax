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
  segmentation masks, `profiler.db` (profiling, owned by microProfiler, read
  by microVis), `infer.db` (predictions/features, owned by microModel), and
  per-dataset `session.yml` (deep-merged by microProfiler and microVis so
  each tool owns its own keys; microModel does NOT use session.yml — it
  reads patterns from its own config files).

* `microProfiler` produces the dataset (TIFFs, masks, `profiler.db`,
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

**Path:** `microMax/microBase/` · **Layout:** flat modules under
`src/microBase/` (no subpackages).

Module map (overview):

| Module             | Purpose                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `__init__.py`      | Re-exports public names; sets env vars disabling albumentations update/telemetry         |
| `schema.py`        | Classifies regex-captured columns (structural vs extra); derives `well` from `row`+`col` |
| `io.py`            | TIFF/mask readers, normalized to `(H, W, C)`; raises `ImageReadError`/`DatasetError`; no writers |
| `cells.py`         | Pure functions for labeled masks: single-cell cropping, per-object edge-pixel ratio        |
| `config.py`        | YAML load/save (atomic) + per-dataset `SessionFile` (deep-merge into `session.yml`)       |
| `augment.py`       | AlbumentationsX pipeline builder (class name -> kwargs, unknown names/kwargs raise)       |
| `normalize.py`     | Per-channel percentile clip + min-max rescale + optional z-score                         |
| `image_dataset.py` | Whole-image loader: regex metadata, masks, LRU cache, cell cropping                      |
| `cell_dataset.py`  | Pre-cropped single-cell TIFF loader (one TIFF per cell)                                  |
| `patterns.py`      | Default regex patterns for common microscope file layouts                                |
| `db_contracts.py`  | Shared profiler.db / infer.db table+column names and SQL helpers                          |
| `errors.py`        | Exception hierarchy (`MicroMaxError`, `ImageReadError`, `ConfigError`, ...)                |

Key concepts:

* `ImageDataset` pivots metadata from filename-regex captures (one row per
  site), supports one-channel-per-file and multi-channel-per-file (CHW/HWC)
  layouts, and offers per-site image loading and per-mask cell cropping. Its
  LRU cache is thread-safe and picklable (so torch DataLoader workers can
  spawn it on Windows). Missing/unreadable files raise `ImageReadError`
  (pipeline quarantine), layout/config mistakes raise `DatasetError` /
  `ConfigError`, and auto-detection skips unreadable candidates. Missing mask
  paths are skipped.

* `CellDataset` loads pre-cropped single cells with channels multiplexed
  inside each file.

* `SessionFile` lives at `<dataset>/session.yml`, deep-merges updates so each
  tool only owns its own keys, and writes atomically (temp + replace).

* All metadata is TEXT end-to-end (regex captures used verbatim); structural
  columns absent from the pattern are simply missing, never defaulted.

* `db_contracts` is the single source of truth for the cross-package DB
  schema (inference/reduction/find_cluster tables, column prefixes, DR method
  names, `directory` canonicalization) and for SQL identifier quoting.

* Library code never calls `sys.exit`: everything raises a `MicroMaxError`
  subclass and only the CLI/GUI boundaries decide how to report it.

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

* `io/` — SQLite wrapper (`Database`) for `profiler.db`.

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

* A missing or unreadable image file **quarantines its metadata row**: every
  file the row references (all channel files + masks) is deleted and the row
  is skipped, so the step and the whole run continue instead of hard-exiting.
  Layout/config mistakes still abort — only genuine I/O/decode failures are
  treated as broken rows. Inference (microModel) is out of scope: earlier
  steps normally remove the row from disk before it runs.

* GUI and CLI build the same `PipelineConfig` and drive the same
  `run_pipeline` loop, so behaviour is identical. Every section has a `run`
  flag defaulting to `false` — a minimal YAML never silently runs a
  destructive step. The GUI has a single Input dir (no separate output dir):
  every artifact is written next to the sources.

* Config is YAML + strict validation (unknown keys raise an error listing the
  valid keys). Metadata-gated steps require their metadata column (zproject
  needs `stack`, tile needs `field`).

* Progress flows through a `ProgressCollector` (tqdm for CLI, status
  bars/signals for GUI). Workers are cancellable at checkpoints.

* The optional inference step lazily imports microModel to run per-object
  inference and optional DR reduction / cluster prediction (one or more
  pre-fitted reducer pickles of any DR mix, and/or a baseline cluster.pkl
  that kNN-predicts the find_cluster table), writing a per-block DB under
  the dataset dir.

* Outputs: in-place processed TIFFs, `<stem>_cp_masks_<obj>.png` masks,
  `profiler.db` (image + per-object tables), `<dataset>/<output_db>`
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
  `microBase.ImageDataset` + `profiler.db`.

* `io/infer_db.py` — `InferDB`, read-only reader for a microModel `infer.db`
  (`inference` + `reduction_<method>` + `find_cluster`).

* `io/profiler_db.py` — `ProfilerDB`, read-only reader for a microProfiler
  `profiler.db` (one reader per open DB tab).

* `widgets/` — image display (thumbnail grid + full-res view), channel
  controls, image filters, well-grid canvas, label annotation panel, pixel
  info, data view, profiler/infer DB plot tabs.

* `processing/` — multi-channel compositing, contrast, mask overlay; `plotting.py`
  builds facet-aware matplotlib figures (boxplot / barplot mean±SD / scatter)
  and exports vector PDFs with editable Type-42 text.

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

* The Image panel's filters are labeled with their metadata column names
  (`field`/`stack`/`timepoint` + extra columns). Both "Color by" dropdowns
  (well grid and Object Overlay) accept profiler tables, merged Excel metadata
  and every loaded infer DB column (`<db-stem>/<column>`, directory-scoped
  with fallback); infer columns also provide object counts when no profiler
  table does.

* The Data page selects a dataset directory via a line edit (type/browse/drop)
  and opens read-only plot tabs: `Select Profiler DB` (one or more `profiler.db`
  files → an object table plotted as boxplot / barplot mean±SD / scatter with
  X/Y/color/size/facets and palette; X/Y accept every meta + measurement
  variable, categoricals plotted on level ticks) and `Select Infer DB` (one or
  more microModel `infer.db` files → DR-method scatter colored/sized by one
  variable each, filtered to the current dataset). Each selected DB gets its
  own tab (re-selecting activates it). Every tab has a free-form pandas-
  expression filter applied before plotting. Plot controls sit in a left column
  with the interactive canvas on the right (hover shows the point's values,
  minus the reduction coordinates on infer tabs); all plots export vector PDFs
  with editable text. PyGwalker is not used.

* Excel plate metadata (`Select Metadata`) is merged by `well` into every open
  plot tab on **Merge** (no DB write) and un-merged on **Clear**. **Write to
  DB** persists it additively — missing columns are added with `ALTER TABLE`
  and rows updated by `well` — in every loaded profiler DB (all tables with a
  `well` column) and every loaded infer DB (`inference` table), preserving
  primary keys and BLOB features.

**Adding a new widget:** add the class under `widgets/`, instantiate it in
`MainWindow.__init__` and wire its signals to private `_on_*` handlers; access
data through `self._dm`; wrap heavy work in a `QRunnable` under `worker.py`.

***

## 7. microModel — SSL pretrain + train + infer

**Path:** `microMax/microModel/` · **Entry:** `micromodel` (CLI subcommands:
`pretrain`, `train`, `infer`, `deduplication`, `label`, `augment-vis`,
`reduction`, `reduction-vis`, `attention-vis`).

Additional deps: `lightly` (SSL projection heads/losses), `timm` (backbones +
ViT rebuild), `pacmap` (PaCMAP/LocalMAP DR).

Package layout (overview):

* `cli/main.py` — argparse dispatch to the subcommands.

* `pretrain.py` / `train.py` / `infer.py` / `deduplication.py` / `label.py` —
  the five stage entries.

* `backbone.py` — backbone builders (incl. DINOv3 ViT), loss, head,
  model + bundle loaders.

* `dataset.py` — PyTorch Datasets: SSL multi-view, single-cell, whole-image.

* `models/` — SSL method implementations: DINOv3 (lightly + own
  DINO/iBOT/KoLeo/Gram components), behind a small registry.

* `reduction.py` — DR reduction + Leiden cluster finding over infer.db
  features (pca/umap/pacmap/localmap; a baseline cluster.pkl kNN-predicts
  new datasets). Implements the `reduction` command.

* `reduction_vis.py` — Flask server for interactive point inspection (the
  `reduction-vis` command).

* `label.py` — Flask server for interactive multi-label annotation (the
  `label` command).

* `augment_vis.py` — augmentation preview (the `augment-vis` command).

* `attention_vis.py` — offline attention/patch-similarity PDF from a
  trained SSL bundle (the `attention-vis` command).

* `plots.py` — training/pretrain monitoring plots shared by the loops and
  the commands above.

* `monitor.py` — MetricsTracker (metrics.csv + TensorBoard sinks) and the
  DINOv3 training-quality diagnostics (head-collapse / gram-split metrics,
  patch-similarity + CLS-attention maps) used by the pretrain loop and
  attention-vis.

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
  reduction-vis, never baked into the DB.

* Bundles carry their meta (channels, normalization, augmentation);
  inference always uses the settings baked into the bundle at training time.

* Deduplication (`deduplication.py`) prunes redundancy / picks diverse new data over
  pre-cropped single-cell folders. Teacher-branch features (cached per folder,
  keyed by bundle + file list) are projected into a whitened-PCA +
  L2-normalized space (same convention as vis clustering) and selected by
  radius-coverage greedy: keep a cell only if it lies beyond a radius of the
  already-kept set — sparse regions keep a floor, dense regions thin out.
  A previous `selection_state.pkl` seeds an incremental run, so new datasets
  contribute only latent diversity. Outputs (manifest, curated/ hardlinks,
  keep_label.csv, plot) never touch the source folders.

* Label (`label.py`) is an interactive multi-label labeling web app
  over pre-cropped single-cell folders (a project-level single-label
  mode makes positives mutually exclusive). An SSL or train bundle embeds all
  cells (deduplication extraction + cache) into a whitened-PCA space where a
  kNN suggest engine scores every label from the user's positive /
  explicit-negative exemplars ("never labeled" is never a negative); a
  classify bundle's per-class probabilities are a second suggestion source.
  Queues: diverse cold-start (farthest-point), auto-label confirm,
  per-label ranked, uncertain band, and a review queue whose leave-one-out
  kNN consistency check ranks already-decided cells whose embedding
  contradicts their label (suspected mislabels) for re-labeling. Cells are
  displayed through the bundle's inference preprocessing (uniform square
  model input, percentile normalization ignoring the zero background).
  Labels support drag reordering and confirmed deletion; every write hits
  SQLite immediately, and re-running a save_dir resumes the project (a
  changed model bundle logs a warning but keeps the human decisions).   `label_export.csv` (`;`-joined multi-labels) feeds train directly —
  exportable at any time with only a subset annotated. State lives in an
  append-friendly SQLite DB (`label.db` single-label mode, `label_multiple.db`
  multi-label mode; label registry, current decisions, full decision log) so
  re-annotating, adding labels or datasets never destroys prior work;
  `label_export.csv` (`;`-joined multi-labels) feeds train directly.

* Interactive surfaces are the two Flask viewers: reduction-vis
  (`http://127.0.0.1:5000`) for clicking through cells in the reduction
  scatter, and label (same port convention) for labeling them.

**Adding a new SSL method:** create `models/<method>.py` with a model class,
`build_<method>`, `train_step`, and a criterion factory; register the module in
`models/__init__.py` (`_SSL_REGISTRY`) and add its criterion branch in
`get_criterion`; add a `configs/pretrain_<method>.yml` and a `<method>:`
config block.

**Adding a new augmentation:** reference its AlbumentationsX class name in the
config lists — no microBase/microModel change needed.

***

## 8. When you change something

* Keep this file in sync with the code's **overall design** (architecture,
  module layout, data flow, dependency graph). Remember it describes only the
  design framework — don't stuff new detailed rules/parameters here.

* Added a new module → update the relevant package's overview section.

* Added a console subcommand → update the CLI entries + `pyproject.toml`
  `[project.scripts]`.

