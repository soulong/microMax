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
| `db_merge.py`      | Mask-aware merge of profiler/infer DBs into fused per-object tables (+ merge-DB writer)    |
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
  names, every path-like column stored portable-first: CWD-relative when
  possible, absolute fallback — `directory` always, and infer.db's
  `mask_filename` likewise; `filename` holds bare file names and full paths
  are re-joined via `resolve_directory`) and for SQL identifier quoting.

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
  step panels (one per pipeline step), background workers, progress;
  `ui_spec.py` holds the shared layout tokens (page margins, button
  heights, compact-width caps) so GUI geometry is tuned in one place.
  Repeatable work (segmentation objects / object profiling / inference) is
  a flat step panel hosting one independent `block-card` box per block;
  Image Profiling uses the same block-container layout with one fixed
  block, so both profiling titles and boxes sit at the same level.

* `cli.py` — headless run, fully equivalent to the GUI.

* `user_defaults.py` — per-user GUI defaults in the shared file `~/.micromax`
  (microProfiler's `microprofiler` section; key-merged, atomic): the
  inference step records the last run's model / reducer(s) / cluster.pkl
  there and the GUI panel pre-fills from it — a remembered reducer/cluster
  path IS the run flag (the old DR/Cluster group checkboxes are gone: an
  empty path means the stage is skipped). The main
  window also stores/restores its size in the `window` sub-section. A
  missing file is ignored at startup; after the first run the file is
  (re)created and completed with every known key (window, model,
  reducer(s), cluster) so it can be hand-edited. This file is GUI
  preferences only — the pipeline run config is a separate YAML passed to
  the CLI.

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
  the dataset dir and remembering the used model / reducer(s) / cluster.pkl
  in `~/.micromax` (user level — the next run defaults to them). Each
  finished block is then AUTO-MERGED with the
  profiler object table of the SAME mask (via microBase `db_merge`;
  sources untouched) into `<dataset>/merge_<mask>.db`. Merges match object
  tables to masks by the table-name convention (the table must be named
  after its mask, i.e. a custom `output_table_name` must equal the mask
  name); written merge outputs carry a `mask` column instead.

* Outputs: in-place processed TIFFs, `<stem>_cp_masks_<obj>.png` masks,
  `profiler.db` (image + per-object tables), `<dataset>/<output_db>`
  (inference), `<dataset>/merge_<mask>.db` (per-mask auto-merge), and
  `session.yml` (applied steps + patterns). The terminal log is mirrored to
  `<dataset>/microProfiler.log` (one file per dataset, INFO; GUI and CLI).

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

**Path:** `microMax/microVis/` · **Entry:** `microvis` (GUI only). Launch logs progress to the terminal (INFO only); each dataset directory keeps its own `microVis.log` (also INFO).

Package layout (overview):

* `io/data_module.py` — `DataModule`, the single facade over
  `microBase.ImageDataset` + `profiler.db`.

* `io/merged_data.py` — `MergedData`, a thin GUI wrapper over microBase
  `db_merge`: the integrated per-object table from any mix of profiler.db
  (object tables) and infer.db files, fused ONLY within the same mask
  (identity columns well/label/directory/...; colliding columns prefixed
  `<db-stem>/`; different masks stack, tagged in a `mask` column; infer
  rows without mask info join the profiler's mask with a log hint). Also
  the Excel-metadata merge and the dataset directory scoping.

* `widgets/` — image display (thumbnail grid + full-res view), channel
  controls, image filters, well-grid canvas, label annotation panel, pixel
  info, data view, and `data_plot.py` — THE single plot view over the merged
  table (all pickers editable, left-click a scatter point shows the cropped
  single cell in a near-cursor popup, empty click hides it); `ui_spec.py`
  centralizes the control-pane geometry tokens and small widget builders
  (form rows, small buttons, checkbox strips) shared by the panel widgets.

* `processing/` — multi-channel compositing, contrast, mask overlay; `plotting.py`
  builds facet-aware matplotlib figures (scatter / line mean±SEM / boxplot /
  barplot mean±SEM) and exports vector PDFs with editable Type-42 text.

* `worker.py` — background `QRunnable` workers (thumbnail, full-res, crop,
  object export, dataset load).

* `main_window.py` — `MainWindow`, all signal wiring and the label-annotation
  state.

* `user_defaults.py` — per-user GUI defaults in the shared file `~/.micromax`
  (microVis's `microvis` section; key-merged, atomic): the main window
  stores/restores its size there. A missing file is ignored at startup and
  (re)created after the first run.

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
  (well grid and Object Overlay) accept the integrated merged-table columns
  (`merge/<column>`, directory-scoped with fallback). The raw profiler-table
  columns are listed only while NO merged table exists — once any DB is
  selected the integrated table REPLACES them (no duplicate entries); merged
  columns also provide object counts when no profiler table does.

* The Data page selects a dataset directory via a line edit (type/browse/drop);
  **Browse... → Load Dataset** sit on that row with **Reset** at the far right,
  and the whole top control area is a titleless rounded box (as is the
  plot-control column, whose width and x match the Image page's control boxes).
  The DB **Select DB** button accepts any number of profiler.db AND infer.db
  files of
  that dataset (a single selection is used as-is); the fused sources are
  shown next to the button as `a.db + b.db (+ metadata) -> merge`. The
  selection works WITHOUT a loaded dataset too — the merge, the plot and
  the Excel-metadata merge are dataset-independent; only the point-click
  cell popup (needs images) and the well-grid color-by merge/* entries
  (needs the DataModule) wait for a dataset, and Write to DB then asks for
  a save location instead of writing next to the dataset. Their
  object rows are
  fused into ONE integrated table per mask (microBase `db_merge`: outer
  merge on the identity columns — well, label, directory, ... — so
  profiler measurements and infer predictions/coordinates meet in a single
  row per object; colliding columns are prefixed `<db-stem>/`; different
  masks stack with a `mask` tag; infer rows without mask info join the
  profiler's mask, logged as a hint). One plot area renders that table:
  scatter / smooth
  line (scipy-default spline fit; categorical x → per-group means ± SEM on
  shared level positions, continuous x → a numeric-axis fit of the raw y
  per categorical color; the Show points toggle also gates its raw dots) /
  boxplot / barplot mean±SEM with
  X/Y/color/size/facets and palette; X/Y accept every merged column,
  categoricals plotted on level ticks. Facet panels SHARE both axes and place categories/groups at their
  GLOBAL level positions (a category missing from one panel still occupies
  its slot there), so panels compare like with like. Every picker combo is editable — type to filter long column lists.
  A free-form pandas-expression filter (directly below Chart) is applied
  before plotting. Plot
  controls sit in a left column with the interactive canvas on the right;
  hover shows a point's values and LEFT-CLICKING a scatter point shows the
  corresponding cropped single cell in a near-cursor popup (nearest point
  wins on overlap; clicking empty space hides it). With "Normalize cell
  image" checked (the checkbox block under Show points on boxplot) the
  popup renders each channel normalized to the CELL itself (per-channel
  Low/High percentiles over the cell's own nonzero pixels + gamma; tuning
  the controls keeps the popup open and re-renders live) for texture
  compares independent of absolute intensity; unchecked it falls back to
  the Image page's rendering. Channel colors follow the Image page either
  way. All plots export vector
  PDFs with editable text. PyGwalker is not used.

* Excel plate metadata (**Select Metadata**) is merged by `well`
  into the integrated table on **Merge** (in memory — no DB write) and
  un-merged on **Clear**. **Write to DB** writes the integrated table
  (profiler + infer + merged metadata columns) into a NEW database next to
  the dataset — the small edit after the button controls the file name
  (default merge.db, table `merged`); the source DBs are never modified.

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

* `pretrain.py` / `train.py` / `infer.py` / `deduplication.py` / `label/` —
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

* `label/` — subpackage for the interactive multi-label annotation web app
  (the `label` command): `db.py` (single append-friendly SQLite project DB
  + undo), `engines.py` (exemplar-kNN suggester + mislabel review),
  `features.py` (cached classify-bundle extraction), `cluster.py`
  (Leiden cluster assignment + medoids over the shared space),
  `imaging.py`
  (percentile PNG render + LRU), `server.py` (Flask APIs) and `ui.py`
  (embedded single-page UI).

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
  no separate input-size parameter. pretrain/train also accept
  `data.file_list` (one or more CSVs with a `filepath` column, e.g.
  deduplication's curated.csv) as the data source, taking precedence over
  `data.file_dir`; train's labels then come from the CSV's `label` column only,
  and relative paths resolve against the process CWD.

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
  from the data itself, no config key: `data.file_list` mode takes each
  CSV's own `label` column, `data.file_dir` mode takes each file's parent
  folder name (a file directly in the root is unlabeled and dropped). The
  loss follows the label form — no config key:
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
  Every row carries a `ground_truth` column = the file's parent folder
  name (the one label convention; a `data.label_csv` key no longer exists),
  and a `mask_name` column holding the BARE mask name (no
  `mask_` prefix — that form is only the internal metadata column name):
  the segmented objects' mask in whole-image mode (config value, or the
  first available mask when null), and an optional config label for
  single-cell mode (null yml -> NULL). This column is the per-mask
  grouping key for the downstream merges; a NULL joins the profiler DB's
  mask with a log hint. Whole-image rows' `mask_filename` follows the same
  portable-first convention as `directory` (canonical_directory:
  CWD-relative when possible, absolute fallback — the bare filename is its
  last segment), so consumers resolve it with `resolve_directory` exactly
  like `directory`; legacy absolute values pass through. Intensity
  `filename` is always bare names (single-cell: the bare TIFF name;
  whole-image: a JSON array of bare channel names). Probability-descending
  ordering is a display
  concern of reduction-vis, never baked into the DB.

* Bundles carry their meta (channels, normalization, augmentation);
  inference always uses the settings baked into the bundle at training time.

* Deduplication (`deduplication.py`) prunes redundancy / picks diverse new data over
  pre-cropped single-cell folders or data.file_list CSVs (implicit roots =
  the listed files' parent dirs; data.file_dir is ignored when file_list is
  set; the CSV's label column is the per-cell label). Two config profiles:
  `deduplication_prune.yml` (reference: null, sources compete) and
  `deduplication_incremental.yml` (a previous selection_state.pkl seeds the
  run; max_add caps the additions). Teacher-branch features (cached per folder,
  keyed by bundle + file list) are projected into a whitened-PCA +
  L2-normalized space (same convention as vis clustering) and selected by
  radius-coverage greedy: keep a cell only if it lies beyond a radius of the
  already-kept set — sparse regions keep a floor, dense regions thin out
  (`selection.adaptive` > 0 makes the radius density-adaptive per cell:
  dense regions exclude with a smaller radius and keep more, sparse ones
  with a larger radius and keep fewer).
  A previous `selection_state.pkl` seeds an incremental run, so new datasets
  contribute only latent diversity. Optional group ratio control
  (`selection.group_by`: cluster / label / source + `selection.target_ratio`)
  power-compresses per-group keep quotas so the largest:smallest group ratio
  lands near the target (e.g. 100:1 -> 10:1); clusters use kNN Leiden with an
  automatic resolution sweep (coarsest partition within 95% of the best
  modularity), quotas apply to the new candidates only in incremental runs,
  and without target_ratio groups are reported only (manifest column,
  per-group summary, plot colors). Outputs (manifest, curated.csv file list
  (CWD-relative paths — the data.file_list input for pretrain/train),
  curated/ hardlinks, keep_label.csv, plot) never touch the source folders.

* Label (the `label/` subpackage: `db.py` storage, `engines.py` suggesters,
  `features.py` extraction, `cluster.py` Leiden clustering over the shared
  space, `imaging.py` render + cache, `server.py` Flask
  APIs, `ui.py` embedded page) is an interactive multi-label labeling web
  app over pre-cropped single-cell folders or data.file_list CSVs (filepath
  column, e.g. deduplication's curated.csv — implicit roots are the listed
  files' parent dirs, data.file_dir is ignored when file_list is set; the CSV's
  label column is the preset). The UI is a GRID BATCH WORKTABLE over ONE
  unified queue — no queue-mode switch: pick an image source, click a
  sidebar label, and the top-bar scope radio picks the members
  (undecided = cells without a decision for the label, ranked by the
  label's score — the main labeling queue; with / without / neg / union =
  the already-decided cells for verification). Every scope renders as one
  page of thumbnails (K = page size, pager
  at the bottom), the user selects images and checks target labels, and
  Apply + / Apply − (explicit negatives) / Remove write selected label(s)
  to selected image(s) in ONE server action (single transaction, one
  undoable op); shift+clicking a thumbnail records an instant explicit
  negative for the checked targets; annotated cells always show their
  label dots bottom-right. A sort radio picks the ranking (best first /
  uncertain first / the leave-one-out "suspicious" mislabel check, which
  needs a model and the Positive scope), plus a Shuffle for the undecided
  queue. Double-click a thumbnail for a zoomed
  overlay where labels
  are toggled directly (chips or keys cycle undecided -> positive ->
  explicit negative, ←/→ walk the queue, suspicious cells show the
  contradicting evidence cell side by side).
  Clicking a sidebar label loads its queue and makes it the only
  checked target; the selection survives page flips so wrong cells can be
  collected across pages and removed in one go.
  An SSL or train bundle embeds all cells (deduplication extraction + cache)
  into a whitened-PCA space where a kNN suggest engine scores every label
  from the user's positive / explicit-negative exemplars ("never labeled"
  is never a negative): score = top-k mean similarity to the positives
  minus `recommend.neg_weight` × top-k mean similarity to the explicit
  negatives. That ONE transparent score drives the Collect order, the
  thumbnail badge and the decided scopes' certainty, and every write
  invalidates
  the cached scores so the ranking follows the latest exemplars. A
  classify bundle's per-class probabilities are a second suggestion
  source. The top-bar Refresh-model button (manual by design) fits a
  per-label logistic scorer for every label holding at least four
  positives AND four explicit negatives — a refreshed label's score
  everywhere becomes that model's P(positive), so negatives define a real
  decision boundary instead of only nudging a similarity; new writes mark
  the models stale (status next to the button) but they keep scoring
  until the user refreshes again, and a mode switch drops them.
  Cluster-assisted bulk labeling (the sidebar's Clusters tab, configured
  by `cluster.target` = a target cluster count, or an explicit
  `cluster.res`): `cluster.py` partitions the shared L2-normalized space
  with the SAME Leiden/kNN-graph clustering as the reduction command
  (resolution ladder + bisection lands nearest the target count), computes
  each cluster's medoid (most typical member) and caches the assignment to
  `features/clusters_*.npz` keyed on bundle identity + cell list + PCA
  width — decision-independent, shared by both modes. Clicking a cluster
  card narrows the queue to that cluster (orthogonal to source/scope, with
  a Medoid ↓ sort = similarity to the medoid first), and Cluster Apply + /
  − / Remove writes the checked targets to the whole cluster ∩ current
  scope through the SAME `_in_scope` membership the queue uses — one
  transaction, one undoable op — so tens of thousands of cells cost ~one
  decision per cluster, then Uncertain/Suspicious sorts clean up cluster
  boundaries per cell. A classify bundle additionally feeds the queue's
  Predicted dropdown (keep cells whose argmax prediction is the picked
  class) and the Prob ↓ sort (P(class) first — argmax confidence while
  All), the fast verify loop after "Create labels from model classes".
  The pager takes a page number + Enter to jump; page flips keep the
  selection.
  Every user action (batch apply, single write, undo) is one
  `op_id` in the append-only decision log, and Ctrl+Z /api/undo reverts
  the newest op exactly (repeated undos walk backward, never redo). Cells render through
  the bundle's inference preprocessing (uniform square model input,
  percentile normalization ignoring the zero background) behind a small
  render LRU; display-only contrast/gamma/size controls adjust the render.
  Labels support drag reordering and confirmed deletion; every write hits
  SQLite immediately, and re-running a save_dir resumes the project (a
  changed model bundle logs a warning but keeps the human decisions).
  Each mode exports its own train-ready CSV after every write
  (`label_export.csv` multi / `label_export_single.csv` single; `;`-joined
  multi-labels). Every path the label project stores (cells, sources, model
  meta, feature caches, exports) is PORTABLE — CWD-relative forward-slash
  like `curated.csv`, absolute only outside the CWD. State lives in TWO
  fully independent append-friendly
  SQLite DBs — `label_multiple.db` (multi) and `label_single.db` (single),
  switched by the top-bar Mode dropdown (Multi-label / Single-label) and
  restored from the `label_mode.txt` pointer file. The two stores never
  mix: separate label registries, decisions, undo history and exports;
  single mode additionally enforces exclusivity (a new positive clears the
  cell's other positives, keep-set semantics, one undoable op). Everything
  decision-independent — the feature cache, the embedding space,
  the suggest engine and the display pipeline — is built once and shared by
  both modes (a switch re-registers cells into the other DB and
  invalidates the score caches once; sub-second). Legacy projects migrate
  on first startup: `annotations.db` -> `label_multiple.db`,
  `annotations_single.db` -> `label_single.db`, and the v0.21 unified
  `label.db` routes by the mode recorded in its meta (set aside as
  `label.db.old` when the target name is taken) so re-annotating, adding
  labels or datasets never destroys prior work.

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

## 8. GUI style conventions (microProfiler + microVis)

The two desktop GUIs share ONE visual language. Whenever a similar
function/region exists in both, it must use the same wording, font size and
control style — not necessarily the same layout.

Palette (microVis is the canonical base):

| Token | Value | Used for |
| ----- | ----- | -------- |
| window | `#1e1e2e` | main window, input/thumbnail/log backgrounds |
| surface | `#252536` | global `QWidget`, pages, inset block cards |
| raised | `#2d2d44` | group boxes/cards, popup menus, hover |
| border | `#333333` | every 1px border, separator, splitter |
| text / muted / placeholder | `#e0e0e0` / `#888888` / `#666666` | |
| accent / accent-hover | `#5a8a9a` / `#6a9aaa` | selection, focus, progress |
| danger / success | `#f04770` / `#06d6a0` | errors / success |

Typography scale: **title 11pt / panel content 9pt / auxiliary 8pt**.
Control panes (microVis side panels, microProfiler step cards) carry a local
9pt content stylesheet; QGroupBox titles stay 11pt; hints, status lines,
mini buttons, checkbox strips and thumbnail captions stay 8pt.

Control rules:

* `resources/style.qss` exists in each package and the two copies must stay
  byte-identical (no shared resource file). App-specific classes
  (microProfiler sidebar footer, microVis nav-tab) are part of the common
  union stylesheet.
* Flat boxes: a functional group of controls sits in a rounded `#2d2d44`
  surface with NO border (QGroupBox, or a titleless `QWidget[class="panel-box"]`
  for the microVis well-grid / Data-page control areas). Inner blocks are
  transparent; only input/select widgets carry a surface of their own.
* One box level only: in microProfiler the block-container step panels
  (Segment / Image Profiling / Object Profiling / Inference) are
  `QGroupBox[class="flat"]`; each block is its own
  `QWidget[class="block-card"]` box, so adding a block adds an independent
  box instead of nesting another frame. Image Profiling uses the same
  container with exactly one fixed block. The Input-page Filter is a
  regular rounded card.
* Inputs/selects are neutral `#1e1e2e` blocks with NO visible border (a
  transparent 1px border turns accent-colored on focus). There is exactly
  ONE button style, and the microVis Data-page `Browse...` button is the
  reference: a TRANSPARENT block with a thin `#444455` border and NO fill —
  inside a box the box colour shows through, on a page the background
  colour shows through; hover highlights the border (`#5a8a9a`); the text
  is accent-colored. No primary/secondary/danger variants; only the length
  follows the caption. Every button shares the same height.
* Similar functions use the same wording: path/file pickers read `Browse...`
  (the Data-page `Select DB` / `Select Metadata` actions keep their action
  names); the dataset row is `Dataset` + `Browse...` / `Load Dataset` /
  `Reset` (`Load Dataset` after `Browse...`, `Reset` right-aligned, the path
  box taking 2/3 of the free width). Reset/remove/clear actions sit
  right-aligned with a gap from the controls before them — the one
  deliberate exception is microProfiler's Filter-panel `Clear Filter`,
  which sits left-aligned directly after `+ Add Filter`.
* All data-entry spin boxes hide their up/down arrows (one QSS rule in the
  shared file); users type the values directly.
* Geometry tokens live in each package's `ui_spec.py`; the stylesheet owns
  look-and-feel. Pattern regex fields use the same monospace 9pt style and
  the same labels (`Image pattern:` / `Mask pattern:` / `Image subdir:`) in
  both apps.
* Window size is remembered per app in the shared GUI config `~/.micromax`
  (`microprofiler:` / `microvis:` sections, each with
  `window: {width, height}`): restore on start, save on close (normal size
  when maximized); a missing file is ignored at startup and recreated with
  all known keys after the first run. This file is GUI preferences only, not
  a pipeline config.

***

## 9. When you change something

* Keep this file in sync with the code's **overall design** (architecture,
  module layout, data flow, dependency graph). Remember it describes only the
  design framework — don't stuff new detailed rules/parameters here.

* Added a new module → update the relevant package's overview section.

* Added a console subcommand → update the CLI entries + `pyproject.toml`
  `[project.scripts]`.

