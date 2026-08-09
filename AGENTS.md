# AGENTS.md — Microscopy Suite

Single map for the four-package suite under `microMax/`. Read it before
adding, removing, or changing features — it explains what each package does,
how they depend on each other, what every module's interface is, and where
the seams live.

```
microMax/
├── microBase/        ← shared foundation (no internal deps)
├── microProfiler/    ← depends on microBase
├── microVis/         ← depends on microBase
└── microModel/       ← depends on microBase (+ lightly for SSL heads/losses)
```

---

## 1. Running environment

| Item | Value |
|---|---|
| OS | Windows 10/11, PowerShell |
| Python | `>=3.10` (target 3.12 in CI) |
| Conda env | `micro` (located at `C:\Users\haohe\miniforge3\envs\micro`) |
| Python exe | `C:\Users\haohe\miniforge3\envs\micro\python.exe` |
| Script dir | `C:\Users\haohe\miniforge3\envs\micro\Scripts\` (not on PATH by default) |

Install order (bottom-up, editable installs):

```powershell
C:\Users\haohe\miniforge3\envs\micro\python.exe -m pip install -e c:\Users\haohe\GitHub\microMax\microBase
C:\Users\haohe\miniforge3\envs\micro\python.exe -m pip install -e c:\Users\haohe\GitHub\microMax\microProfiler
C:\Users\haohe\miniforge3\envs\micro\python.exe -m pip install -e c:\Users\haohe\GitHub\microMax\microVis
C:\Users\haohe\miniforge3\envs\micro\python.exe -m pip install -e c:\Users\haohe\GitHub\microMax\microModel
```

Console scripts (after install):

- `microprofiler` → `microProfiler.__main__:main`
- `microvis`       → `microVis.__main__:main`
- `micromodel`     → `microModel.cli.main:main`

Heavy external stack used across the suite: `numpy`, `pandas`, `tifffile`,
`Pillow`, `pyyaml`, `albumentationsx`, `torch`, `timm`, `cellpose`, `jax`,
`PySide6`, `matplotlib`, `flask`, `umap-learn`, `pydantic`, `scikit-learn`,
`opencv-python`, `tqdm`, `natsort`, `openpyxl`, `lightly`. (`microBase` itself
only declares: `numpy`, `pandas`, `tifffile`, `Pillow`, `pyyaml`,
`albumentationsx`, `natsort`.)

---

## 2. Inter-package dependency graph

```
                       ┌──────────────────────────────────┐
                       │            microBase              │
                       │  ImageDataset, CellDataset, io,   │
                       │  cells, config, schema, augment,  │
                       │  normalize                        │
                       └──────────────┬────────────────────┘
                                      │ (editable install)
            ┌─────────────────────────┼───────────────────────────┐
            │                         │                           │
            ▼                         ▼                           ▼
   ┌─────────────────┐       ┌─────────────────┐        ┌─────────────────┐
   │  microProfiler  │       │     microVis    │        │    microModel   │
   │  CLI + Qt GUI   │       │     Qt GUI      │        │  CLI + Flask    │
   │  pipeline + DB  │       │  viewer + DB    │        │  pretrain/      │
   └────────┬────────┘       └────────┬────────┘        │  train/         │
            │                         │                 │  infer          │
              │  writes result.db       │  reads result.db│  writes model.pt (pretrain)
              │  writes session.yml     │  writes session.yml│  writes model.pt (train)
             │                         │                 │  writes infer.db (infer)
             │                         │                 │
             │  inference step (lazy):  │                 │
             │  writes <dataset>/<output_db>             │
             │  via microModel functions                 │
             │                         │                 │
            └─────── shared on-disk artifacts ──────────┘
                                                  └─ no session.yml (config-only)
```

Rules:

- `microBase` is the only package the three consumers (`microProfiler`,
  `microVis`, `microModel`) import. `microModel` additionally depends on
  `lightly` for SSL projection heads, losses, and masked ViT.
- The three consumers never import each other directly, with ONE exception:
  `microProfiler` lazily imports `microModel` (its `inference` pipeline step
  and the GUI Inference panel's bundle-meta reader). The import happens only
  when inference is actually requested; microProfiler runs fully without
  microModel installed, and requesting inference without it is a hard error
  (CLI `print + sys.exit(1)` before the batch loop; GUI popup, app keeps
  working). `microModel` is self-contained — its model/dataset helpers
  (`build_backbone`, `FocalLoss`, `ClassificationHead`, `Model`, datasets,
  utils) are originals owned by `microModel`; only `load_yaml`,
  `CellDataset`/`ImageDataset`, `build_pipeline`/`apply`, `normalize`,
  `read_mask`, and `cells.get_labels` are imported from microBase.
- `microProfiler` produces the dataset (TIFFs, masks, `result.db`,
  `session.yml`). `microVis` and `microModel` consume it. They may be used
  independently — `microModel` accepts pre-cropped single-cell folders (e.g.
  exported by `microVis`) as training input, and whole-image datasets (with
  segmentation masks) for inference/extraction.
- Per-dataset state lives at `<dataset>/session.yml` and is deep-merged by
  microProfiler and microVis, so each tool only owns its own keys.
  `microModel` does NOT read or write `session.yml` — it obtains
  `image_pattern`/`mask_pattern` from its own config files.

### Database ownership

| DB file | Owner (writer) | Readers |
|---|---|---|
| `<dataset>/result.db` | microProfiler (image/object profiling tables) | microVis (also writes merged + label tables back) |
| `<dataset>/infer.db` (default) or `{output_dir}/infer.db` | microModel (metadata / prediction / feature / reduction tables) | microModel (self, for `vis-reduction*`) |
| `<dataset>/<output_db>` (e.g. `infer.db`; name set per inference block) | microProfiler inference step via microModel `run_inference`/`show_reduction` (tables `inference`, `reduction_pca`, `reduction_umap`, `reduction_pca_variance` — identical to the microModel CLI format) | microModel `vis-reduction*` (self) |

### Session keys

`SessionFile` deep-merges these sections into `<dataset>/session.yml`.
`session.yml` is read once on first Browse of a directory; action buttons
write current GUI state. Writers below are action buttons (including "Load
Dataset", which is distinct from "Browse").

| Section | Writer (action buttons) | Reader (first Browse only) |
|---|---|---|
| `image_pattern`, `mask_pattern`, `image_subdir_pattern` | microProfiler (Load Dataset via `set_patterns`, Run/Apply via `_save_session_yml`), microVis (Load Dataset, Export, Write-to-DB, Write Label to DB) | microProfiler, microVis |
| `applied_steps` | microProfiler (Run/Apply via `_save_session_yml`) | microProfiler (skip logic) |
| `channel_colors` (`{ch: {color, vmin, vmax}}`) | microVis (Load Dataset, Export, Write-to-DB, Write Label to DB, Auto, ImageControls Reset) | microVis (first load only; never re-read on reload) |
| step params (`resize`, `basic`, `zproject`, `tile`, `segment`, `image_profile`, `object_profile`, `inference`) | microProfiler (Run/Apply via `_save_session_yml`) | microProfiler |
| `filter` | microProfiler (Run/Apply via `_save_session_yml`; always written, even `[]`, so cleared filters never resurrect) | microProfiler |

---

## 3. Cross-cutting design rules

1. **Minimal interface, deep module.** `microBase` exposes 22 public names
   from a flat 10-module layout. Consumers reach for `microBase.X` first; a
   new function goes into `microBase` only if at least two consumers need it.
2. **No guessing, hard-exit on bad input.** CLI error paths are
   `print(message, file=sys.stderr); sys.exit(1)`; GUI paths are Qt signals
   that surface a message dialog. No silent fallbacks, no try/except that
   swallows errors. The one exception is `microProfiler` CLI batch mode: a
   per-dataset failure is logged and the batch continues (intentional, so one
   bad dataset doesn't abort a plate scan) — including `SystemExit` from
   microBase's hard-exits. `MetadataValidationError` still hard-exits.
3. **Explicit over implicit.** `channel_layout` is a constructor argument
   (`None` / `"CHW"` / `"HWC"`), never inferred from file contents.
   `image_pattern` is a regex the user supplies, not a heuristic.
   `channel_layout=None` has two modes: if `image_pattern` has a `channel`
   group, each file is one channel (column name = `ch` + raw capture when the
   capture starts with a digit, e.g. `ch1`, `ch01`; non-numeric captures like
   `blue` stay as-is); if no `channel` group, each file is the single
   implicit channel `ch1` (supports non-standard single-channel datasets like
   HPA).
4. **Array layout invariant.** All multi-channel arrays in the public API are
   `(H, W, C)`. All masks are `(H, W)` integer arrays. Cropped cells are
   `(h, w, C)` with background zeroed.
5. **1-based channel indexing** in the public API (`read_tiff_channels`,
   `CellDataset.get_cell`, `ImageDataset.get_imageset`). Intensity column
   names are `ch` + the raw regex-captured channel string when the capture
   starts with a digit (`ch1`, `ch2`, `ch01`); non-numeric captures (e.g.
   `blue`) are used verbatim with no `ch` prefix. No numeric coercion — the
   capture is used verbatim.
6. **Config is YAML + manual validation.** `PipelineConfig` is a dataclass
   (no pydantic at the top level). `pydantic` IS used internally by the BaSiC
   algorithm (`preprocessing/basic/`) for its own config validation. Unknown
   config keys raise `ValueError` with a message listing the valid keys (the
   CLI wraps config loading in `print + sys.exit(1)`).
7. **Naming conventions are enforced everywhere:**
   - `timepoint` (never `tp`)
   - `vis` (never `viz`) — `vis_interactive.py`, `vis.py`, `microVis`
   - `normalize` (never `normalization`)
   - `eval_*` prefix for objects rebuilt from a trained model
   - `logger` (never `_log`)
   - `label` for the integer label of a single cell inside a mask
   - `output_db` (never `db_name`/`database`) — the per-inference-block DB
     file name. Exception: microModel's inference config key is `db_name`
     (its CLI/YAML schema), which microProfiler translates to `output_db`.
     microProfiler's profiling-DB parameter is `result_db`.
   - `max_value` — the trusted input-dtype maximum for inference (read from
     config, never derived silently)
8. **Type hints on public interfaces only.** No `TypeVar`/generic gymnastics.
   `Protocol` is used for the `IControllerView` view-interface seam in
   `microProfiler/gui/interfaces.py`. Internal helpers may be untyped.
9. **Per-module try/except + print + sys.exit.** No central exception
   hierarchy. The one shared exception is `MetadataValidationError` in
   `microProfiler.pipeline` (raised in `_run_zproject`/`_run_tile`, caught in
   `cli.main`). `derive_well` raises `ValueError` (not SystemExit) on
   non-numeric row/col — GUI dataset-load workers catch `Exception`; a
   `SystemExit` there would strand the modal dialog. All GUI worker threads
   that call microBase (dataset load, preview, microVis loaders) also catch
   `SystemExit` explicitly, since microBase's error paths are `sys.exit`.
10. **No schema versioning for DBs.** `result.db` and `infer.db` have no
    `_meta` table and no migration path — old/dirty DBs must be deleted
    manually.
11. **`NO_ALBUMENTATIONS_UPDATE=1`** and **`ALBUMENTATIONS_NO_TELEMETRY=1`**
    are set in `microBase/__init__.py` (via `os.environ.setdefault`,
    respecting any pre-existing value) before any sub-module import.
12. **Input image size is defined entirely by augmentation steps**
    (`LongestMaxSize + PadIfNeeded + Resize`). There is no separate
    `model.input_size` parameter or `resize_to_target` function.
13. **All metadata is TEXT from extraction to DB storage.**
    `normalize_capture` is a verbatim pass-through — no leading-zero
    stripping, no int coercion. `'01'` stays `'01'`, `'blue'` stays `'blue'`.
    `derive_well` is the only exception: it computes well labels (e.g. `'A1'`)
    from `row`+`col` by doing its own `int()` internally. Structural columns
    (`well`, `field`, `stack`, `timepoint`) absent from the regex are simply
    missing — they are NOT defaulted to `0`. Consumers tolerate missing
    columns (microVis hides empty filters; microModel writes only the
    captured columns). Extra regex-captured columns (anything not structural,
    not `ext`/`tile`/`mask_name`, not a path/mask/intensity column) are
    propagated as dynamic TEXT columns through microVis CSV exports and
    microModel `infer.db`.
14. **Metadata-gated pipeline steps.** Z-projection requires `stack`; tiling
    requires `field`. An enabled step without its column raises
    `MetadataValidationError` before any work begins (CLI: `print + sys.exit(1)`;
    GUI: worker error popup).
15. **Channel fallback.** If `image_pattern` has no `channel` group and
    `channel_layout=None`, `ImageDataset` assigns the implicit intensity
    column `ch1` to every image group. If a `channel` group is present, the
    `ch` prefix is added only when the capture starts with a digit; non-
    numeric captures are used verbatim.
16. **session.yml read-once, write-on-action.** `session.yml` is read only on
    the FIRST browse of a given dataset directory (to pre-fill GUI fields);
    subsequent browses of the same directory preserve the user's GUI edits.
    "Load Dataset" is an action button in both microProfiler and microVis: it
    writes the user's current patterns to `session.yml` before scanning data.
    Other action buttons (Run, Apply, Export, Auto, Write-to-DB, ImageControls
    Reset, Write Label to DB) also persist GUI state. The only exempt buttons
    are "Browse" (selects directory + reads session.yml only) and full
    "Reset" (Data tab). For channel colors specifically: on reload, the
    user's current GUI colors are preserved for channels that still exist,
    defaults are used for new channels, and removed channels are dropped —
    `session.yml` `channel_colors` are never re-read on reload. Note: the
    read-once guard compares against the CURRENT directory only — Browse
    A → Browse B → Browse A re-reads A's session.yml.
17. **Modal dataset loading.** Both microProfiler and microVis construct
    `ImageDataset`/`DataModule` on a background `QThread` while showing a
    modal `QProgressDialog` ("Loading dataset...") to prevent UI freeze on
    large datasets. The worker emits `finished`/`error` signals that trigger
    phase-2 UI population or an error dialog. The dialog is indeterminate
    (`setRange(0, 0)`) with no cancel button.
18. **Worker-thread SystemExit containment.** microBase hard-exits
    (`sys.exit`) on bad dataset state. Every GUI worker that calls microBase
    must catch `SystemExit` and route it through the error signal — otherwise
    the modal dialog never closes, the wait cursor stays, or the app dies.

---

## 4. microBase — shared foundation

**Path:** `microMax/microBase/`
**Version:** `0.7.0`  •  **Layout:** flat, 10 modules under `src/microBase/`
**Dependencies:** `numpy`, `pandas`, `tifffile`, `Pillow`, `pyyaml`,
`albumentationsx`, `natsort`.

### 4.1 Module map

| Module | Purpose | Public API |
|---|---|---|
| `__init__.py` | Re-exports 22 public names; sets `NO_ALBUMENTATIONS_UPDATE=1` + `ALBUMENTATIONS_NO_TELEMETRY=1`; defines `__version__`. | all of the below |
| `schema.py` | Classify regex-captured columns; derive `well` from `row+col`; `normalize_capture` is a verbatim pass-through. `derive_well` raises `ValueError` on non-numeric col. | `MetadataSchema`, `derive_well`, `normalize_capture` |
| `io.py` | TIFF/mask readers/writers; normalizes to `(H, W, C)`. Accepts `channel_layout` of `None` / `"CHW"` / `"HWC"`. | `read_tiff`, `read_tiff_channels`, `read_mask`, `write_tiff`, `write_mask` |
| `cells.py` | Pure functions for cropping single cells from labeled masks. `crop_cell` hard-exits on `label < 1` and on zero-pixel labels. | `get_labels`, `crop_cell`, `crop_all_cells` |
| `config.py` | YAML load/save + per-dataset `SessionFile`. `load_yaml` recursively normalizes string spellings of `null`/`None`/`none` (case-insensitive) to Python `None`. | `load_yaml`, `save_yaml`, `SessionFile` |
| `augment.py` | Registry-driven augmentation pipeline on top of AlbumentationsX (import name `albumentations`). | `build_pipeline`, `apply` |
| `normalize.py` | Per-channel percentile clip + min-max rescale to `[0, 1]`, then z-score (per-channel / pooled / none). | `normalize` |
| `image_dataset.py` | Whole-image loader with regex metadata, masks, LRU cache, cell cropping. | `ImageDataset` |
| `cell_dataset.py` | Pre-cropped single-cell TIFF loader (one TIFF per cell). | `CellDataset` |
| `patterns.py` | Default regex patterns for common microscopy file layouts (Operetta-style naming with optional leading zeros). Used by microProfiler and microVis as GUI defaults. | `DEFAULT_IMAGE_PATTERN`, `DEFAULT_MASK_PATTERN`, `DEFAULT_IMAGE_SUBDIR_PATTERN` |

### 4.2 Internal dependency table

| Module | Depends on (internal) |
|---|---|
| `__init__.py` | all other modules |
| `cell_dataset.py` | `.io`, `.schema` |
| `image_dataset.py` | `.io`, `.cells`, `.schema` |
| `augment.py`, `cells.py`, `config.py`, `io.py`, `normalize.py`, `patterns.py`, `schema.py` | none |

### 4.3 Key interfaces

#### `ImageDataset` — whole-image loader

```python
ImageDataset(root, image_pattern, mask_pattern=None,
             channel_layout=None,            # None | "CHW" | "HWC"
             image_subdir_pattern=None,      # glob, e.g. "Images/"
             filters=None)                   # dict[str, str] column -> regex
```

- `channel_layout=None` ⇒ one-channel-per-file. Two sub-modes:
  - `image_pattern` HAS a `channel` group → each file is one channel;
    intensity column = `ch` + raw capture when the capture starts with a
    digit (e.g. `ch1`, `ch01`); non-numeric captures stay as-is (e.g.
    `blue` stays `blue`).
  - `image_pattern` has NO `channel` group → each file is the single
    implicit channel `ch1` (non-standard single-channel datasets like HPA).
- `channel_layout="CHW"`/`"HWC"` ⇒ multi-channel-per-file; `image_pattern`
  MUST NOT have a `channel` group.
- Properties: `metadata`, `intensity_colnames`, `mask_colnames`, `img_shape`,
  `img_dtype`, `schema`, `captured_fields` (set of regex-captured metadata
  column names including `row`/`col` before well-derivation; `schema.
  captured_fields` excludes `row`/`col` — the two differ by design).
- Methods: `build_metadata()`, `get_imageset(row_idx, masks=None) -> (img_HWC, mask_dict)`,
  `image_path(row_idx, channel)`, `filter_metadata(column, pattern)`,
  `get_cropped_cell(row_idx, label, mask_name, padding=4)`,
  `get_cropped_cells(row_idx, mask_name, padding=4, labels=None)`.
- LRU cache size 8, keyed by `row_idx`. On cache miss, all available masks
  are loaded (not just the requested subset) to keep the cache complete. The
  cache is thread-safe (lock-guarded) and **picklable** (lock recreated on
  unpickle, cached entries dropped on pickle), so `ImageDataset` — and thus
  `WholeImageCellDataset` — can be spawned by torch `DataLoader` worker
  processes on Windows.
- **Sorting.** `metadata` rows are natsorted by structural columns in
  microscopy-natural priority: `row → col → well → field → stack → timepoint`
  (columns absent from the regex are skipped; `well` covers explicit-`well`
  datasets, row+col-derived wells sort via row/col before derivation). If
  none of these columns exist, the row order is the deterministic sorted-key
  order of the shared merge keys. `intensity_colnames` and `mask_colnames`
  are natsorted; `schema.extra_cols` is natsorted.
- **Missing files hard-exit.** A row whose image path is NaN (mask-only row
  or deleted file) makes `get_imageset`/`image_path` hard-exit with a clear
  message. Missing MASK paths are skipped silently (mask columns stay NaN;
  e.g. Cellpose found zero cells → no mask saved).

#### `CellDataset` — single-cell loader

```python
CellDataset(root, channel_layout="CHW", image_pattern=None)
```

- One TIFF per cell, channels multiplexed inside the file.
- `channel_layout`: `None` (each TIFF is 2D single-channel, returned as
  `(H, W, 1)`), `"CHW"` (pages along C axis), or `"HWC"` (last axis is C).
- If `image_pattern` has a `channel` group, it is silently skipped.
- Properties: `metadata`, `intensity_colnames`, `img_shape`, `img_dtype`, `schema`.
- **Sorting.** Files are iterated in `natsorted` order (`cell_1.tiff` →
  `cell_2.tiff` → `cell_10.tiff`), which determines `metadata` row order.
- Methods: `build_metadata()`, `get_cell(idx) -> (H, W, C)`.

#### `crop_cell(volume, mask, label, padding=4)`

- Input: `volume (H, W, C)`, `mask (H, W) int`, `label int`.
- Output: `(crop, cell_mask, bbox)` where `crop (h, w, C)` has background
  zeroed, `cell_mask (h, w) bool`, `bbox (x, y, w, h)`.
- Hard-exits if `label < 1` or the cell has zero pixels in the mask.

#### `normalize(image, mask=None, method="per_channel", clip_low=0.05, clip_high=99.95)`

- Clipping is always per-channel and always applied: each channel is
  percentile-clipped and **min-max rescaled to `[0, 1]`**
  (`(clipped - lo)/(hi - lo)`); degenerate channels (empty region or `hi <= lo`)
  become zeros.
- `method`: `"per_channel"` (per-channel z-score of the rescaled values) |
  `"global"` (single pooled z-score across all channels) | `"null"`/`None`
  (rescale only, no z-score).
- Returns `(H, W, C) float32`. Background (mask falsy) is zeroed; stats are
  computed over foreground pixels only when a mask is given.

#### `build_pipeline(spec)` / `apply(pipeline, image, mask=None)`

- `spec`: list of single-key dicts using albumentations class names, e.g.
  `[{"HorizontalFlip": {"p": 0.5}}, {"Rotate": {"angle_range": 180, "border_mode": 0, "p": 0.5}}]`.
  This suite runs on AlbumentationsX (import name `albumentations`), which
  renamed several classic kwargs (`limit` -> `angle_range`, `scale_limit` ->
  `scale_range`, `brightness_limit`/`contrast_limit` ->
  `brightness_range`/`contrast_range`, `gamma_limit` -> `gamma_range`,
  `blur_limit` -> `blur_range`). Unknown kwargs are silently dropped by
  AlbumentationsX, so `build_pipeline` hard-exits on them.
- Returns a `Compose` or `None` if `spec` is empty.
- `apply` returns `(image, mask)`; mask may be `None`.
- Resizing is handled by `LongestMaxSize`, `PadIfNeeded`, and `Resize` steps
  in the augmentation spec — there is no separate `resize_to_target` function.

#### `SessionFile(dataset_dir)`

- Lives at `<dataset>/session.yml`.
- Methods: `exists()`, `load()`, `save(updates)` (deep-merge; non-dict values
  replace), `get_patterns()`, `set_patterns(...)` (None args skipped),
  `get_applied_steps()`, `set_applied_steps(steps)`, `get_channel_colors()`,
  `set_channel_colors(colors)`. Null-valued keys are treated as absent by the
  getters.

#### Default patterns (`patterns.py`)

Three module-level constants providing default regex patterns for the
Operetta naming convention with optional leading zeros (e.g.
`r1c1f01p01-ch1sk1`). Both microProfiler and microVis import these as GUI
defaults at startup; users override them by typing custom patterns in the GUI
or setting them in `session.yml`.

- `DEFAULT_IMAGE_PATTERN` — matches `r{row}c{col}f{field}p{stack}-ch{channel}(sk|t){timepoint}*.tiff`. The `0?` prefix on `field`/`stack`/`channel`/`timepoint` makes a single leading zero optional, while `row`/`col` capture all digits verbatim.
- `DEFAULT_MASK_PATTERN` — same layout plus `_cp_masks_{mask_name}.png`.
- `DEFAULT_IMAGE_SUBDIR_PATTERN` — `"images"`.

### 4.4 Adding a new shared capability

1. Decide: is it needed by ≥2 consumers? If yes → `microBase`. If no → keep
   it inside the consumer.
2. Add a new flat module under `src/microBase/` (no subpackages).
3. Re-export its public names in `__init__.py` and add them to `__all__`.
4. Write a test under `tests/test_<module>.py` (tests are microBase-only —
   the consumers have no test suites).
5. Bump version in `src/microBase/__init__.py` if the surface changed.

---

## 5. microProfiler — pipeline + Qt GUI

**Path:** `microMax/microProfiler/`
**Version:** `1.5.0`  •  **Entry:** `microprofiler` (CLI: `run`; GUI: no args)
**Deps on microBase:** `ImageDataset`, `SessionFile`, `load_yaml`,
`read_tiff`, `write_tiff`, `read_mask`, `write_mask`,
`DEFAULT_IMAGE_PATTERN`, `DEFAULT_MASK_PATTERN`, `DEFAULT_IMAGE_SUBDIR_PATTERN`.

### 5.1 Package layout

```
src/microProfiler/
├── __init__.py            __version__, re-exports ImageDataset, Database
├── __main__.py            dispatches CLI vs GUI
├── cli.py                 `microprofiler run` subcommand
├── config.py              PipelineConfig dataclasses + YAML load/save + validation
├── pipeline.py            run_step / run_pipeline orchestration + inference step
├── logging_utils.py       logger factory
├── progress_collector.py  pub-sub progress (CLI tqdm + GUI signals)
├── io/
│   ├── __init__.py        re-exports microBase io + read_image/write_image/
│   │                      read_image_shape + clone_dataset/rebuild_dataset
│   └── database.py        thread-safe SQLite wrapper (WAL mode)
├── preprocessing/
│   ├── resizer.py         scale-factor resize (in-place)
│   ├── basic_correction.py  BaSiC illumination correction (fit/transform)
│   ├── z_projection.py    max/mean/min Z collapse (deletes sources BEFORE
│   │                      writing the projection — see §5.2)
│   ├── tile_splitter.py   split into tiles, encode tile idx into `field`
│   └── basic/             BaSiC algorithm (jax + dct + metrics)
├── segmentation/
│   └── cellpose.py        Cellpose-SAM, saves <stem>_cp_masks_<obj>.png
├── profiling/
│   ├── batch_writer.py    batched DataFrame -> SQLite
│   ├── image_profiler.py  whole-image intensity features
│   ├── object_profiler.py per-object shape/intensity/radial/granularity/GLCM/correlation
│   └── extras.py          regionprops feature factories
├── gui/
│   ├── app.py             QApplication bootstrap
│   ├── main_window.py     top-level QMainWindow, 5-page nav
│   ├── pipeline_controller.py  mediates view <-> worker
│   ├── dataset_service.py random row pick + single image load
│   ├── state.py           PipelineState (current + original dataset)
│   ├── sidebar.py         nav + global controls
│   ├── dpi.py             high-DPI scaling helper
│   ├── image_widgets.py   QGraphicsView viewer + ChannelTile
│   ├── interfaces.py      IControllerView Protocol
│   ├── panels/            BaseStepPanel + BlockContainerPanel + 9 panels
│   │                      (resize, basic, zproject, tile, segment,
│   │                      image_profile, object_profile, inference, filter)
│   └── workers/           PipelineWorker + PreviewWorker + DatasetLoadWorker (QThread)
└── resources/             icon.ico, style.qss
```

### 5.2 The pipeline

```
                ┌─────────┐
                │  Input  │  ImageDataset(root, image_pattern, mask_pattern,
                └────┬────┘            image_subdir_pattern, filters)
                     │
                     ▼
   ┌──────────────────────────────────────────────────────────┐
   │  resize → basic → zproject → tile → segment → profile → infer │
   └──────────────────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   source TIFFs              result.db
   (overwritten in-place      session.yml (applied_steps)
   by resize/basic/zproject) <dataset>/.microprofiler/BaSiC_model/
                              <stem>_cp_masks_<obj>.png
                              <dataset>/<output_db> (inference, e.g. infer.db)
                              <dataset>/reducer_{pca,umap}.pkl (fitted reduction)
```

- **Section `run` defaults are `False`** everywhere (CLI config dataclasses
  and the GUI checkboxes agree). A step runs only when its config sets
  `run: true` / its checkbox is checked — a minimal YAML never silently runs
  a destructive in-place step.
- The four preprocessing steps (`resize`, `basic`, `zproject`, `tile`) are
  gated by `SessionFile.get_applied_steps()`; rerunning skips already-applied
  steps. `segment`, `profile`, and `infer` always run when enabled (not
  gated) — they are non-destructive and intended to re-run. **`applied_steps`
  never shrinks**: the GUI persists the union of the previous applied steps
  and the steps that actually executed, so unchecking a checkbox can never
  cause a destructive re-run (double resize/zproject/tile) of
  already-processed files. One exception: a **fit-only BaSiC run**
  (`mode: "fit"`, the GUI "Fit Model" button) is never gated — it writes
  shading models without touching images.
- **Tiling design note.** `tile_splitter` writes only COMPLETE tiles;
  right/bottom remainder regions (and images smaller than the tile size,
  which produce zero tiles) are intentionally dropped, and
  `delete_original=True` deletes the sources afterwards. Choose tile sizes
  that divide the image dimensions, or the remainder is lost.
- **Z-projection deletes sources BEFORE writing the projection**
  (`delete_original=True`, then write). This order is intentional: multiple
  input stacks can map to the same projection filename, so write-then-delete
  would delete freshly written projections. A failed write therefore loses
  the stack — keep raw backups of the originals.
- **Profiling failure semantics.** Per-row and per-channel feature failures
  are logged and skipped by design (one bad row/channel never aborts the
  run). A run-level profiling failure is re-raised: batches flushed before
  the failure are already committed, but the dataset is NOT treated as
  complete — the CLI batch loop logs the dataset failure and continues, the
  GUI worker surfaces the error dialog, and `cli._is_dataset_complete`
  additionally requires the `image` table to contain one row per (filtered)
  image. Object tables have no row-count guard — after a failed object run,
  delete `result.db` (or the affected table) before re-running.
- **Config-driven overwrite flags (no CLI flags).** There are no
  `--overwrite-db` / `--overwrite-mask` CLI flags — specific parameters are
  passed via config or GUI:
  - `segment.configs[].overwrite_mask` (default `false`) — when `True`,
    `segment_dataset` re-runs segmentation even if `<stem>_cp_masks_<obj>.png`
    already exists.
  - `object_profile.configs[].overwrite_db` (default `false`) — when `True`,
    `_run_profile` drops the existing `<output_table_name>` table before
    profiling; when `False`, the `BatchWriter` replaces the table on first
    write anyway. The GUI additionally drops all expected profiling tables
    after a confirmation dialog when rerunning profiling — more destructive
    than the CLI's `overwrite_db` semantics, by design.
- **Inference step (optional, needs microModel).** The `infer` step runs
  per-object inference with trained microModel bundles (whole-image mode, one
  block per bundle) and is fully driven by the `inference` config section.
  It is non-destructive, always runs when enabled (NOT gated by
  `applied_steps`, though a COMPLETED inference is recorded in
  `applied_steps` as `infer` — never on cancel, never via the checkbox
  state), and runs LAST in `run_pipeline` (so Run All includes it).
  microModel is imported lazily inside `pipeline._run_inference` (and the
  GUI panel's bundle-meta reader); a missing install is a hard error only
  when inference is actually requested — CLI: `print + sys.exit(1)` pre-check
  before the batch loop; GUI: popup via the worker error signal, app keeps
  working. Each block calls `microModel.infer.run_inference` (writes
  `<dataset>/<output_db>` with the microModel CLI table format —
  `output_dir` is always null so the DB lands under the dataset dir) and,
  when its `reduction` is enabled, `microModel.vis.show_reduction(
  save_plots=False, raise_on_error=True)` (writes `reduction_pca` /
  `reduction_umap` / `reduction_pca_variance` + fitted
  `reducer_{pca,umap}.pkl` under the dataset dir; pre-fitted reducers
  transform directly). microModel's `print + sys.exit` error paths are
  captured and re-raised as `RuntimeError` (`_call_micromodel`), so a failure
  surfaces as a GUI error popup and a CLI dataset-level failure (batch
  continues); a `SystemExit` while the run was cancelled is re-raised as
  `InterruptedError` instead. GUI inference uses
  `dataloader.num_workers: 4` / `prefetch_factor: 2` /
  `persistent_workers: false` — the DataLoader worker processes are spawned
  via the picklable `ImageDataset` cache (§4.3) and torn down after each
  inference pass, while tqdm/log output still comes from the pipeline worker
  thread. `_call_micromodel` tees microModel's stderr to the launching
  terminal (tqdm renders exactly like `micromodel infer`) and to the progress
  collector: tqdm lines become live status-bar bars, microModel's INFO logs
  become status lines. `max_value` is REQUIRED per block and always trusted
  as configured — the GUI derives the dtype max only to default new blocks
  and to show a NON-blocking mismatch warning on dataset load.
  `feature`/`pred_class`/`pred_prob` are per-block toggles; SSL bundles (no
  `num_classes` in meta, read on model selection) force features-only in the
  GUI. `pred_prob` requires `pred_class`. Duplicate `output_db` names block
  Run. No plots are produced for reduction — microProfiler is a data-only
  analysis suite.
- **Inference channels are explicit and ordered.** Each inference block's
  channel selection is a row of plain `QCheckBox` widgets that defaults to
  NOTHING checked; a block with zero checked channels is blocked at Run.
  The left-to-right order is the model's input channel order — `channels` in
  the config is an ORDERED list of names, mapped to 1-based indices in that
  order for microModel. ◀/▶ buttons reorder. Reordered order + checked state
  survive dataset reloads and session restores.
- **Model info row.** Each inference block shows an always-visible info row
  between the model selection and the mask row, populated from the bundle
  meta when a model is selected (Browse, config restore, or block copy):
  `Model type: SL - Classify` (train bundle, `num_classes` in meta) /
  `Model type: SSL - Features` (pretrain bundle), and
  `Input Channel Number: {in_chans}` (`—` until a bundle is read).
- **Segment Object name defaults to `cell`** and is cited downstream. A
  fresh Segmentation block starts with `Object name = cell`, and a
  config/session restore never clears it. The object-profiling and inference
  mask combos are populated in deterministic order — segment Object names
  first — and auto-select the first entry when the current selection is
  empty or stale, so a fresh Object Profiling / Inference block cites `cell`
  automatically. Deliberate selections and config-restored mask names are
  never overwritten.
- **Dataset-level skip (CLI only).** `cli.main` calls `_is_dataset_complete`
  before `run_pipeline`: if `result.db` exists and already has all expected
  profiling tables — and the `image` table's row count matches the
  (filtered) dataset size — the dataset is skipped. To re-process a skipped
  dataset, delete `result.db` (or the specific tables). There is no flag to
  bypass this check or to force re-running already-applied preprocessing
  steps — delete `session.yml` or edit `applied_steps` manually.
- The pipeline is fully driven by `PipelineConfig` (a dataclass). The GUI
  builds the same config from panel state, so CLI and GUI are identical in
  behaviour, with two GUI-only conveniences: (a) BaSiC "Run" with
  pre-fitted models downgrades `fit-transform` to `transform`-only; (b)
  profiling reruns confirm-dropping existing tables. "Fit Model" is a
  GUI-only fit-only invocation of the basic step.
- **Pre-flight metadata validation.** `zproject` requires a `stack` column
  and `tile` requires a `field` column in the dataset metadata. If an
  enabled step's required column is absent, `MetadataValidationError` is
  raised in `pipeline._run_zproject`/`_run_tile` before any work begins. The
  CLI catches it and hard-exits; the GUI worker emits `error`, surfaced as a
  `QMessageBox.critical` pop-up.
- **Cooperative cancel.** `PipelineWorker`/`PreviewWorker` observe a cancel
  flag at every progress checkpoint and abort with `InterruptedError`; the
  pipeline/GUI threads are never terminated mid-execution. Cancelling just
  stops the run at the next checkpoint. Cancelling mid-run still unions the
  checked steps into `applied_steps` (the finished path persists the
  checkbox state) — intentionally, so a later re-run never re-applies an
  in-place step onto half-processed files. After an interrupted in-place
  step, restore the original raw data and clear `applied_steps`
  (`session.yml`) before re-running. Inference is the exception: `infer` is
  recorded only when it actually completes (never on cancel, never via the
  checked fallback).

### 5.3 Config schema (`PipelineConfig`)

```yaml
image_pattern:        r"..."
mask_pattern:         r"...(?P<mask_name>\w+)\.png"
image_subdir_pattern: "Images/"
filter:               [{column: well, pattern: "A[1-6]"}]   # always written to session.yml, even []

resize:      {run: false, scale_factor: 0.5}
basic:       {run: false, mode: fit-transform, n_image: 100,
              working_size: 64, enable_darkfield: false}
zproject:    {run: false, method: max}        # max|mean|min
tile:        {run: false, tile_width: 1024, tile_height: 1024}

segment:
  run: false
  configs:
    - object_name: cell
      model_name: cpdino
      chan1: [ch1]            # GUI: all channels unchecked by default; Run blocked
                              # with no selection. CLI: absent/empty falls back
                              # to the first dataset channel.
      chan2: [ch2]            # optional, null or list; unchecked by default (GUI)
      merge1: mean            # mean|sum|max
      merge2: mean
      resize_factor: 0.5
      diameter: null          # null => auto
      flow_threshold: 0.4
      cellprob_threshold: 0.0
      gpu_batch_size: 32
      overwrite_mask: false   # true => re-run even if mask file exists

image_profile:
  run: false
  n_workers: auto             # default: max(1, cpu_count // 2)
  image_channels: [ch1, ch2]  # null/absent/[] => step SKIPPED entirely
  image_thresholds: {ch1: 1000, ch2: 500}

object_profile:
  run: false
  n_workers: auto             # default: max(1, cpu_count // 2)
  configs:
    - mask_name: cell
      parent_mask_name: null
      output_table_name: cell
      overwrite_db: false     # true => DROP TABLE before profiling
      intensity_channels: [ch1, ch2]
      radial_channels: [ch1]
      radial_bins: 4
      gran_channels: [ch1]
      gran_spectrum_length: 8
      gran_subsample_ratio: null     # null => 0.5
      gran_background_subsample_ratio: null  # null => 0.25
      gran_background_radius: null   # null => 10
      glcm_channels: [ch1]
      glcm_distances: [2]
      glcm_levels: 256
      correlation_pairs: [[ch1, ch2]]

inference:                     # optional; needs microModel installed
  run: false
  configs:
    - model: D:\models\runs\model.pt   # SSL bundle OR train bundle
      mask_name: cell                  # dataset mask column (prefix stripped)
      channels: null                   # ORDERED names; top-to-bottom = model input order; null => all
      feature: true                    # write features BLOB
      pred_class: true                 # write pred_class (SSL bundles: ignored)
      pred_prob: true                  # write pred_prob (requires pred_class)
      output_db: infer.db              # DB file name under the dataset dir
      max_value: 65535                 # REQUIRED; always trusted as configured
      reduction:                       # null/absent => no dimension reduction
        enabled: true                  # true => BOTH pca (var_threshold) + umap
        var_threshold: 0.95            # no GUI widget; preserved verbatim on round-trip
        color_by: pred_class           # pred_class|directory|pred_prob|ground_truth
        sample_per_class: 10000
        reducer_pca: null              # provided => transform-only (no refit)
        reducer_umap: null             # provided => transform-only (no refit)
```

Unknown keys anywhere in the config raise `ValueError` listing the valid
keys (the CLI turns it into `print + sys.exit(1)`).

### 5.4 Outputs

| Artifact | Location | Writer |
|---|---|---|
| Resized/corrected/projected/tiled TIFFs | in-place over source files | `preprocessing.*` |
| BaSiC shading models | `<root>/.microprofiler/BaSiC_model/<chan>.{pkl,tiff}` | `basic_correction.fit_models` |
| Cellpose masks | `<src_stem>_cp_masks_<obj>.png` (uint16) | `segmentation.cellpose` |
| Image profiling table | `<dataset>/result.db` table `image` | `image_profiler` |
| Object profiling tables | `<dataset>/result.db` table `<output_table_name>` (e.g. `cell`). Columns: `label` (int object ID), `shape_*` descriptors, `intensity_*` per channel, `parent_*` optional. | `object_profiler` |
| Inference DBs | `<dataset>/<output_db>` (e.g. `infer.db`) — one per inference block, microModel CLI table format: `inference` (+ `reduction_pca`, `reduction_umap`, `reduction_pca_variance` when reduction is enabled) | microModel `run_inference` / `show_reduction` via `pipeline._run_inference` |
| Fitted reducers | `<dataset>/reducer_{pca,umap}.pkl` (when reduction fits) | microModel `show_reduction` |
| Applied steps + patterns | `<dataset>/session.yml` | `cli.main` / `pipeline_controller._save_session_yml` |

The `directory` column in profiling tables uses the full parent directory path
(normalized with forward slashes) via `_resolve_source_directory()`, matching
microModel's `directory` column when both tools process the same dataset.

### 5.5 CLI vs GUI usage

| Module | CLI | GUI |
|---|:--:|:--:|
| `__main__`, `logging_utils`, `progress_collector`, `config`, `pipeline`, `io/*` | ✓ | ✓ |
| `preprocessing/*`, `segmentation/*`, `profiling/*` | ✓ | ✓ |
| `cli` | ✓ | — |
| `gui/*` | — | ✓ |

### 5.6 Adding a new pipeline step

1. Add a new module under `preprocessing/` (or `segmentation/`/`profiling/`
   as appropriate). Public function signature:
   `step_dataset(ds, **kwargs, progress=NullProgressCollector()) -> ImageDataset`.
2. Add a `XxxConfig` dataclass in `config.py` and a field on `PipelineConfig`.
3. Register a step function in `pipeline._STEP_FUNCTIONS` and add a
   `_run_xxx` helper.
4. Add a `XxxStepPanel(BaseStepPanel)` under `gui/panels/` with a `_FIELD_MAP`
   entry per config field. The base class handles serialization automatically
   (`to_config`/`from_config`); block-list panels (segment/profile/inference)
   override `build_config_section` and use the structured
   `load_config_section` list-of-dicts format via `BlockContainerPanel`.
5. Add the panel to `gui/panels/__init__.py` and `gui/__init__.py`, and a
   page to `Sidebar.PAGES` if it is a new top-level page.
6. Add an entry to `PipelineController._STEP_MAPPING` so the GUI can run it.

---

## 6. microVis — interactive Qt viewer

**Path:** `microMax/microVis/`
**Version:** `1.2.0`  •  **Entry:** `microvis` (GUI only — no CLI subcommands)
**Deps on microBase:** `ImageDataset` (via `DataModule`), `SessionFile`,
`DEFAULT_IMAGE_PATTERN`, `DEFAULT_MASK_PATTERN`, `DEFAULT_IMAGE_SUBDIR_PATTERN`.

### 6.1 Package layout

```
src/microVis/
├── __init__.py            __version__
├── __main__.py            --version / --help, else launch GUI
├── app.py                 QApplication bootstrap (Fusion palette, style.qss)
├── _settings.py           plate formats, channel colors, dtype max table, colormap lookup
├── log_utils.py           %TEMP%/microVis.log file logger
├── main_window.py         central QMainWindow, all signal wiring
├── worker.py              QRunnable workers (image/full-res/crop/export) + _DatasetLoadWorker
├── io/
│   └── data_module.py     DataModule: wraps ImageDataset + result.db
├── processing/
│   ├── compositing.py     multi-channel -> 8-bit RGB composite
│   ├── contrast.py        gamma / histogram equalization / invert
│   └── overlay.py         polygon extraction from label masks
├── widgets/
│   ├── _event_filter.py   NoScroll* widgets + RotatedLabel
│   ├── channel_controls.py  per-channel enable/color/vmin/vmax row
│   ├── data_view.py       Data tab: dataset browse, patterns, table preview, reset
│   ├── image_controls.py  sidebar: filters, channels, contrast, export
│   ├── image_display.py   thumbnail grid + _ThumbnailView + polygon overlay
│   ├── label_annotation.py  drag-and-drop object classification panel
│   ├── pixel_info.py      bottom pixel intensity readout
│   ├── well_grid_canvas.py    matplotlib well-plate scatter
│   └── well_grid_controls.py  plate format / color-by / cmap controls
└── resources/             icon.{ico,png}, style.qss
```

### 6.2 Data flow

```
ImageDataset (microBase) ─┐
                          ├─► DataModule ─► MainWindow ─► ImageWorker ─► _ThumbnailView
result.db (sqlite)       ─┘                     │
                                                ├─► FullResWorker (on zoom)
                                                ├─► CropWorker (on object drag)
                                                └─► ObjectExportWorker (on export)

session.yml ←─ MainWindow (patterns + channel_colors)
```

- `DataModule` is the single facade over `microBase.ImageDataset` and
  `result.db`. All GUI modules go through it.
- `MainWindow` owns one `DataModule`, one `QThreadPool`, and the label
  annotation state.
- Workers are `QRunnable` (kept in an `_active_workers` registry until their
  finished/error signal fires, then `deleteLater`); they emit Qt signals
  back to `MainWindow`.
- Worker results are guarded by a generation counter (`_gen`) so stale
  results from a previous dataset or after a reset are silently discarded.
- **session.yml read-once, write-on-action** (see §3.16). "Select Dataset
  Directory" (Browse) only selects the directory + reads `session.yml`
  (first browse of a given directory only); it does NOT scan data.
  Re-selecting the same directory preserves GUI edits. Browsing a different
  directory clears old display state (images, grid, caches, label panel).
  "Load Dataset" is an action button that writes patterns + channel_colors
  to `session.yml` before starting the data scan. Export, Write-to-DB, Write
  Label to DB, Auto, and ImageControls Reset also persist GUI state. On
  reload, channel colors are preserved from the GUI (not re-read from
  `session.yml`).

### 6.3 Inputs / outputs

| Artifact | Direction | Notes |
|---|---|---|
| `<dataset>/session.yml` | read + write | Read once on first Browse of a directory; written by action buttons — Load Dataset, Export, Write-to-DB, Write Label to DB, Auto, ImageControls Reset. See §3.16. |
| `<dataset>/result.db` | read + write | Reads profiling tables; writes merged metadata tables (`write_merged_table`) and label annotation tables (`write_label_table`, default name `{mask}_label`, columns `well, field, stack, timepoint, label, class`). The DB is NOT auto-attached — click "Select DB" to load it. |
| TIFF/PNG images | read | via `DataModule.get_imageset(row_idx)` |
| `%TEMP%/microVis.log` | write | append-only log |
| `<export_dir>/objects_exported/` | write | `ObjectExportWorker` writes one multi-channel ImageJ TIFF per cell + `{mask_name}.csv` manifest (appended on repeat exports) |

### 6.4 Object export schema

Each exported cell is saved as:

```
<save_dir>[/<class_name>][/<well>/]{mask_file_stem}_{label}.tiff
```

`{mask_file_stem}` is the stem of the mask file (taken from the
`mask_{name}` metadata column for the row), guaranteeing uniqueness across
both standard plate datasets (e.g. `r1c1f1p1-ch1sk1_cp_masks_cell`) and
non-standard ones (e.g. HPA `000a6c98-..._mask_cell`) where
`well`/`field`/`stack`/`timepoint` are absent. Falls back to
`row{row_idx}_{label}.tiff` if no mask path is available.

- TIFF axes: `CYX`, with `Labels`/`LUTs`/`mode='composite'` metadata for
  ImageJ/Fiji compatibility. The per-channel `LUTs` (each a `(3, 256)` uint8
  ramp) encode the channel colors chosen in the channel-control panel.
- Cropping uses `microBase.ImageDataset.get_cropped_cells(row_idx, mask_col,
  padding=4)` — the same seam microModel uses for whole-image inference, so
  crops are pixel-identical. Rows whose mask is missing are skipped with a
  warning — the export continues and reports the skipped count. A run with
  zero enabled channels errors out, and a CSV manifest write failure is
  reported as an export error (objects are never reported as exported
  without their manifest). Enabled channels not present in the dataset are a
  hard error (never a silent first-N fallback).
- Export mode combo (Obj range): "Selected images" (index 0) — objects from
  currently selected wells + the Image Filters (fields/stacks/timepoints/
  extra cols) / "Selected wells" (index 1) — all objects from the currently
  selected wells, ignoring the Image Filters / "Annotated" (index 2) /
  "All" (index 3) — every object in the dataset. When the last class is
  deleted, the mode falls back to "Selected images" — never "All" or
  "Selected wells".

`{mask_name}.csv` columns — one CSV file per mask type (e.g. `cell.csv`),
written in append mode on repeated exports. The schema is dynamic: the 9
base columns below are always present, followed by any extra metadata
columns captured by `image_pattern` (sourced from `dataset.schema.extra_cols`
— e.g. `stain`, `batch`). Values are read per-row from `dataset.metadata`.
If an existing CSV at the export path has a different header, the worker
hard-errors (no migrations — user deletes the stale file):

```
filename, well, field, stack, timepoint, label, mask_name,
class_name, channels [, <extra_col_1>, <extra_col_2>, ...]
```

### 6.5 Well grid visibility & Object Label uniqueness

**Well grid hiding.** When a dataset has no `well` metadata (neither
`row`+`col` groups nor an explicit `well` group), the `WellGridCanvas` and
its `WellGridControls` are hidden in `_on_dataset_loaded`. Image selection
then depends solely on the Image Filter panel (field/stack/timepoint/extra
cols, top-100 natsorted per filter). When wells exist, the grid is shown and
is the primary well-selection mechanism (`_selected_wells`).
`_on_full_reset` restores visibility to `True` so the next dataset load
re-evaluates.

**Object Label cross-box uniqueness.** An object can only exist in one
class box at a time. `_ClassBox.dropEvent` enforces this: when a drop
originates from the image display (not from another `_ObjectThumb`), the
target box first removes the object from all other class boxes via
`self._panel._class_boxes`. This invariant makes `_on_crop_finished`'s
"find first box containing the key" logic correct — there is only ever one
box to find.

**Object identity uses `row_idx`, not metadata tuples.** `ObjectKey` is
`(row_idx, label)` — not `(well, field, stack, timepoint, label)`. This
makes label annotation, crop dispatch, full-res zoom, and object export
work for arbitrary metadata combinations, including non-standard datasets
(HPA, single-channel, extra cols only) where `well`/`field`/`stack`/
`timepoint` are absent. Signals (`pixel_clicked`, `full_res_requested`)
carry `row_idx` directly; `FullResWorker`, `CropWorker`, and
`ObjectExportWorker` all key off `row_idx`. The label-annotation DB table
(`{mask}_label`) and the export CSV (`{mask_name}.csv`) still write
`well`/`field`/`stack`/`timepoint` columns, but those values are read
per-row from `dataset.metadata` via `_meta_cell_str` (missing columns →
empty string), not from `ObjectKey`.

**Thumbnail identity also uses `row_idx`.** All thumbnail-keyed lookups in
`ImageDisplay` use `row_idx` as the dict key — never
`(well, field, stack, timepoint)` tuples. Non-standard datasets collapse
those tuples to `("","","","")`, making every thumbnail share one dict
entry and receiving the last result's pixmap. Affected methods:
`update_pixmaps_in_place` (channel-toggle fast path), `_save_current_view_state`,
`saved_state` restore in `_add_thumbnail_sorted`/`_add_thumbnail_column`,
and `_rebuild_display` (iterates the full `matching` list per combo
instead of taking only `matching[0]`). `_apply_full_res_pixmap` matches
thumbnails by `thumb._row_idx == row_idx`.

**Thumbnail labels are built dynamically.** `ImageWorkerConfig.meta_label`
is computed per-row by `_build_meta_label()` in `main_window.py` from
whichever structural columns (`well`/`field`/`stack`/`timepoint`) exist in
the metadata, using the conventional format `A1 f1 z1 t1`. If no structural
columns exist, it falls back to extra cols as `col=val` pairs, then to
`row {row_idx}`. Worker results carry `meta_label` through to the `QLabel`
above each thumbnail. Sort comparison in `_add_thumbnail_sorted` reads
`thumb._well/_field/_stack/_timepoint` attributes directly (not by parsing
label text).

### 6.6 Select object overlay gating

The Object Overlay panel has a `Select object` dropdown (above `Color by`)
that **gates the entire overlay**. Options are `[None, <mask1>, <mask2>, ...]`
populated from `self._dm.mask_names` (prefix-stripped) in
`_populate_label_controls`. Default is `None`. The choice is ephemeral — it
is NOT persisted to session.yml (consistent with `overlay_col`/`overlay_cmap`/
`overlay_alpha`) and resets to `None` on every app start and on
`_on_full_reset`.

- **`None`** → no overlay whatsoever: no polygons, no hover tooltip, no
  object-drag. `need_mask = need_polygons = False`, so `ImageWorker`/
  `FullResWorker` pass `mask=None` to `_ThumbnailView`. The hover and drag
  branches in `_ThumbnailView.mouseMoveEvent` are guarded by
  `if self._mask is not None`, so they skip automatically and left-drag pans
  the image view.
- **A mask selected** → that mask is loaded for every thumbnail: gold outlines
  are drawn, hover shows `label: N`, and objects are draggable onto Object
  Label class boxes. `need_mask = need_polygons = bool(sel)`.

**Outline vs fill.** `outline_only = bool(sel) and (self._overlay_col is None)`.
When `Color by` is None → outline only (1px gold, no fill). When `Color by`
is also set → fill polygons by Color-by value + gold outline
(`outline_only=False`). `Color by` alone (with `Select object` = None)
renders nothing — Color by only takes effect once a mask is selected.

**Selected mask lookup, not first-mask fallback.** `_downscale_mask`,
`FullResWorker.run`, and the crop path (`_on_crop_requested`) look up
`mask_dict[f"mask_{selected_name}"]`. If the selected mask is absent for a
particular image (e.g. NaN mask path skipped), nothing is drawn / the crop
is aborted with a warning — the code does NOT silently substitute the first
mask.

**Drag-to-label crop target.** The dragged object IS the selected object
mask. `_on_crop_requested` and the write-to-DB flow read
`ic.get_selected_object_mask()`; if `""` (None selected) the crop aborts
(drag is already disabled when None, so this is just a guard). The Object
Export panel's separate `Object` dropdown (`_export_mask_combo`, populated
via `set_label_masks` / read via `get_export_mask`) is independent and
controls only the export flow — it is unaffected by `Select object`.

**Cache invalidation.** `_mask_cache` and `_polygon_cache` are keyed by
`row_idx` and become stale when the selected mask changes, so
`_on_object_mask_changed` clears both before scheduling a refresh.
`_detect_change` includes `object_mask` in its state dict and categorizes a
change as `"overlay"` (re-render reusing `_raw_cache`, not a full filter
reload).

### 6.7 Session reset

The Data tab has a "Reset" button (in `DataView.btn_row`) that calls
`MainWindow._on_full_reset()`, restoring the GUI to initial startup state
without restarting the process:

1. Shuts down PyGwalker (HTTP server + loader thread).
2. Bumps `_gen` and `_export_gen` to invalidate all pending worker results.
3. `_thread_pool.waitForDone(2000)` — running export workers finish in the
   background (they hold their own `_dm` reference); results are discarded
   via the gen check.
4. Closes `DataModule`, clears all caches and state fields.
5. Resets all widgets to defaults (ImageDisplay, WellGridCanvas, LabelPanel,
   ImageControls, WellGridControls, DataView, PixelInfo).
6. Switches to the Data tab.

### 6.8 Channel display ranges follow the dataset dtype

Channel vmin/vmax spin boxes are `QDoubleSpinBox` with range/decimals derived
from `DTYPE_MAX` (uint8→255, uint16→65535, uint32→4294967295, float32/64→1.0
with 4 decimals). A hardcoded 16-bit range would clamp `vmax` and silently
corrupt the display of uint32/float datasets.

### 6.9 Adding a new widget

1. Add the widget class under `widgets/`.
2. Instantiate it in `MainWindow.__init__` and wire its signals to private
   `_on_*` handlers.
3. If it needs dataset/db access, go through `self._dm` (the `DataModule`).
   Do not instantiate `ImageDataset` directly in a widget.
4. If it does heavy work, wrap it in a `QRunnable` under `worker.py` rather
   than blocking the UI thread (keep the reference in `_active_workers`).

---

## 7. microModel — SSL pretrain + train + infer

**Path:** `microMax/microModel/`
**Version:** `0.4.0`  •  **Entry:** `micromodel` (CLI: `pretrain`, `train`,
`infer`, `vis-augment`, `vis-reduction`, `vis-reduction-interactive`)
**Deps on microBase:** `CellDataset`, `ImageDataset`, `build_pipeline`,
`apply`, `normalize`, `read_mask`, `cells.get_labels`, `load_yaml`.
**Additional deps:** `lightly>=1.5` (SSL projection/prediction heads,
DINOLoss/IBOTPatchLoss/NegativeCosineSimilarity, `MaskedVisionTransformerTIMM`,
momentum/scheduler utilities), `timm` (backbones + ViT rebuild for DINOv2).

### 7.1 Package layout

```
src/microModel/
├── __init__.py            __version__
├── __main__.py            re-exports cli.main:main for `python -m microModel`
├── cli/
│   └── main.py            argparse dispatch to 6 subcommands
├── pretrain.py            pretrain_ssl — generic SSL loop (BYOL/DINOv2)
├── train.py             train — classification from SSL backbone or scratch
├── infer.py               run_inference — inference + feature extraction -> infer.db
├── backbone.py            FocalLoss, build_backbone, ClassificationHead, Model,
│                          load_model_from_bundle, load_ssl_backbone_from_bundle
├── dataset.py             SSLMultiViewDataset, SingleCellDataset, WholeImageCellDataset
├── vis.py                 show_augmentation, show_reduction, plot_training_results
├── vis_interactive.py     Flask server for interactive PCA/UMAP inspection
├── utils.py               logger, setup_logging, set_seed, select_device,
│                          load_label_csv, resolve_output_paths, reducer pickle
└── models/
    ├── __init__.py        _SSL_REGISTRY, build_ssl_model, get_train_step, get_criterion
    ├── byol.py            BYOL model + build_byol + train_step (2 views, momentum)
    └── dinov2.py          DINOv2 model + build_dinov2 + train_step (global+local views, teacher-student)
configs/
├── pretrain_byol.yml
├── pretrain_dinov2.yml
├── train.yml
├── infer_single_cell.yml
└── infer_whole_image.yml
```

### 7.2 The preprocessing chain

`microModel` reuses `microBase.augment.build_pipeline` / `apply` / `normalize`.
There are two dataset types with different chains:

- **`SSLMultiViewDataset`** (pretrain): for each cell, builds N augmentation
  pipelines from `augmentation_views` (a list of view specs) and returns a
  list of N view tensors. Each view applies:
  `load (raw HWC) → convert to float [0, 1] by data.max_value → extract mask (pixel != 0) → augment (one view pipeline; mask co-transformed) → normalize → torch.from_numpy(CHW)`.
  The float conversion happens once at load: integer images (8/12/16-bit)
  are divided by the REQUIRED `data.max_value` (65535 for 16-bit, 255 for
  8-bit), so every view pipeline and the fixed-reference stats share one
  [0, 1] domain. There are NO `ToFloat` steps in augmentation specs.
  Normalization has two modes: per-view self-normalization (default) and
  fixed-reference (`normalize.fixed_reference: true`), where the full
  transform — per-channel percentile clip, min-max rescale to [0, 1], then
  z-score — is computed ONCE on the raw cell (pre-augmentation, in the
  [0, 1] domain) and applied as a fixed transform to every view. The
  fixed-reference transform is **bit-identical to per-view normalization on
  un-augmented input** for all three methods (`per_channel` / `global` /
  `null`); with augmentation, per-view re-normalization cancels linear
  photometric augmentation (affine-equivariant), leaving only
  gamma/noise/blur as photometric diversity.
- **`SingleCellDataset`** (train) and **`WholeImageCellDataset`** (infer
  whole-image): identical chain — single view, deterministic
  `augmentation_infer` for validation/inference, random `augmentation_train`
  for training. Both honor `normalize.fixed_reference` (train reads its own
  config; infer reads the required bundle meta key). Training labels come
  from a `labels` list built by `train` (label_csv or label_from_dir
  resolution) — the dataset never re-derives a label from the file path.
  Cropping failures inside `WholeImageCellDataset.__getitem__` (mask
  corrupted or changed between indexing and cropping) raise `ValueError`
  instead of hard-exiting, so DataLoader workers fail with a clear message.

Input image size is defined entirely by the augmentation steps
(`LongestMaxSize + PadIfNeeded + Resize`). There is no separate
`model.input_size` parameter.

### 7.3 SSL pretraining flow

```
pretrain_ssl(config)
   │
   ├─► for each root in data.root:
   │       CellDataset(root, channel_layout, image_pattern).build_metadata()
   │       _build_records(...)   # one (cell_ds, idx) pair per cell
   │
   ├─► validate: all roots must resolve to the SAME channel set (hard-exit
   │       on mismatch — one bundle carries one `channels` meta)
   ├─► subsample(pairs, sample_max, sample_by, seed)   # per-root grouping
   ├─► one SSLMultiViewDataset per CellDataset, then ConcatDataset
   │       (each cell -> N augmented views via augmentation_views)
   │
   ├─► build_ssl_model(method, backbone_cfg, method_cfg, device)
   │       dispatches to models/byol.py or models/dinov2.py
   ├─► get_criterion(method, ...) + get_train_step(method)
   ├─► DINOv2 requires an EVEN training.batch_size (hard-exit on odd)
   ├─► optimizer: sgd (BYOL default) | adamw (DINOv2 default)
   │
   └─► save <output_dir>/{model_{epoch}.pt (every save_interval),
                          model.pt (final), <config>.yml}
```

**SSL bundles (`model.pt` / `model_{epoch}.pt`)** — saved by
`pretrain_ssl`. Interval bundles (every `training.save_interval` epochs,
named 1-based without zero padding, e.g. `model_9.pt` for the 9th epoch) and
the final `model.pt` share one schema, and every saved bundle is complete —
usable for exact resume, train transfer, and feature extraction:

```python
{
    "state_dict": <full SSL model state dict — student/online backbone +
                   projection/prediction heads + momentum/teacher nets>,
    "meta": {
        "method": "byol" | "dinov2",
        "backbone": <timm name>,
        "in_chans": int,
        "channels": [1-based indices],
        "channel_layout": "CHW" | "HWC" | None,
        "max_value": float,
        "feat_dim": int,
        "augmentation_infer": <deterministic view spec>,
        "normalize_method", "normalize_with_masking", "normalize_fixed_reference",
        "clip_low", "clip_high",
        "image_pattern": <regex or None>,
    },
    "config": <full config dict>,
    "optimizer_state_dict": ...,
    "epoch": int,            # 1-based (last completed epoch, matches filename)
    "loss_history": [...],
    "dino_loss_history"/"ibot_loss_history"/"koleo_loss_history": [...]  # dinov2 only
    "method": "byol" | "dinov2",
}
```

Bundles are written atomically (`*.pt.tmp` + `os.replace`) so an interrupted
run never leaves a corrupt file. `train`/`infer` pull the student/online
backbone out of `state_dict` via `backbone.extract_backbone_state_dict`
(prefix `student_backbone.vit.*` for DINOv2 — the `mask_token` param is
excluded by the prefix — `backbone.*` for BYOL/conv).

**BYOL** (`models/byol.py`): online backbone + `BYOLProjectionHead` +
`BYOLPredictionHead`, with a momentum target network (deepcopy + freeze).
Backbone features are pooled via `build_backbone`'s `pool_fn` (identity for
already-pooled `(B, C)`, spatial mean for `(B, C, H, W)`, token mean for
`(B, N, C)`), so any timm conv or ViT backbone works. `train_step` uses
exactly 2 views (`views[0:2]`); extra views are ignored. Momentum schedule
is cosine `momentum_start` → `momentum_end` (default 0.996 → 1.0) over
`total_steps`, and the momentum EMA update runs once per optimizer step,
right after `optimizer.step()`. LR schedule: linear warmup (0 →
`training.lr` over `warmup_epochs`) then cosine decay (`training.lr` →
`lr_final`). `transfer_warmup_epochs` (default 1) replaces `warmup_epochs`
in transfer resume mode. Loss: `NegativeCosineSimilarity`. Note: lightly
1.5.x's `NegativeCosineSimilarity` returns `-cosine_similarity(...)`, so the
loss range is `[-1, 1]` and a DECREASING (more negative) loss means the
views are becoming more aligned — negative losses are expected, not a bug.

**AMP hardening (pretrain)**: with `amp: true`, the BYOL MLP heads
(`forward` / `forward_momentum`) run in fp32 — the pooled backbone output is
cast to `float` and the heads execute under `torch.amp.autocast(enabled=False)`.
Additionally `pretrain_ssl` sets
`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`
on CUDA, and `train_step` (both methods) takes an optional `scaler`
(`torch.amp.GradScaler`) and `grad_clip` — the pretrain loop passes both, so
the loss is scaled before backward and gradients are globally clipped
(`training.grad_clip`, default 10.0). If the loss is ever non-finite,
`pretrain_ssl` hard-exits instead of silently writing a corrupted `model.pt`.

**Gradient accumulation (pretrain)**: `training.grad_accum_steps` (default
`1`) accumulates gradients over K micro-batches — effective batch =
`batch_size * K` at ~1/K of the optimizer-step memory. `train_step` (both
methods) takes a `step=True` kwarg: backward runs every micro-batch;
`optimizer.step()` / `zero_grad()` / `scaler.update()` / grad clipping (and
the weight-decay schedule + teacher momentum update, both methods) run only
when `step=True`. `pretrain_ssl` computes `do_step` from the micro-batch
index (step every K-th batch and at the epoch tail, since `drop_last=True`
leaves a partial window that must flush). BN batch statistics still use the
micro-batch size.

**DINOv2** (`models/dinov2.py`): teacher + student ViT, both rebuilt via
`backbone.build_dinov2_vit` (the single shared ViT factory) —
`timm.create_model` called with special init args (`pos_embed="learn"`,
`dynamic_img_size=True`, `init_values=1e-5`). Any timm ViT with a class token
is accepted (e.g. `vit_small_patch16_224`, `vit_tiny_patch16_224`,
`vit_large_patch14_224`); non-ViTs and `gap`/no-cls variants hard-exit. The
teacher is frozen. `backbone.pretrained` is honored: `pretrained=True` loads
ImageNet weights into teacher + student (timm adapts the first conv to the
`data.channels` count via `adapt_input_conv`; `strict=False` keeps the
missing `layer_scale` params at their `1e-5` init). The lightly wrapper is
constructed with `weight_initialization="skip"` when `pretrained=True`.
`backbone.in_chans` is NOT a config key — derived from the resolved
`data.channels` (and from the bundle meta on resume; a mismatch hard-exits).
`dinov2.input_dim` is optional (`null` = derived from the ViT's `embed_dim`);
a non-null value must match or pretrain hard-exits. `augmentation_views[0:2]`
are global views (2 required), `views[2:]` are local views (any count: 1, 2,
4, 6, 8, ...). The student receives masked global views + all local views;
the teacher receives unmasked global views. Uses
`MaskedVisionTransformerTIMM` from lightly. Losses: `DINOLoss` (cls token) +
`IBOTPatchLoss` (masked patches) + `KoLeoLoss` (student cls token of global
views). Teacher temperature, weight decay, and momentum are cosine-scheduled
over `total_steps = epochs * len(loader)`. In `continue` mode
`global_step = start_epoch * len(loader)` keeps the absolute step position;
in `transfer` mode `global_step` starts at 0 and the schedules restart.
`warmup_epochs` controls BOTH the teacher temperature warmup AND the LR
warmup (linear 0 → `training.lr`, then cosine decay → `lr_final`).
`transfer_warmup_epochs` (default 1) replaces `warmup_epochs` in transfer
resume mode. `train_step` returns `(loss, dino, ibot, koleo)` and the
per-epoch log line includes the three components (DINOv2 only; BYOL logs the
plain total). Views move to the device explicitly. Global views are 224×224
(whole cell, fit+pad) and local views 112×112 (RandomResizedCrop
sub-regions; the shipped config uses `[98, 98]` for patch-14 models) — mixed
sizes are supported via `dynamic_img_size`; only views in the same group
(`views[0:2]` vs `views[2:]`) must share one size, since each group is
`torch.cat`-ed. Every view's size must be divisible by the backbone's patch
size (timm hard-asserts this); the patch mask is built from the ACTUAL input
grid, with masking starting after all prefix tokens (cls + register tokens).

**Head collapse guard**: `train_step` tracks an EMA of the student head's
logit spread (std across the code dims) and, when it drops below `5e-3`,
re-initializes the projection heads (`DINOv2.reinit_heads` — student fresh,
teacher deepcopy, 200-step cooldown). `train_step` re-points the optimizer's
param groups at the model's current parameters (the optimizer was created
with the pre-reinit heads), and the projection-head forwards run in fp32
(`autocast off`). `dinov2.head_batch_norm` (BatchNorm inside the heads) is on
by default — it normalizes the head output scale and is a critical
anti-collapse mechanism. The teacher's BN buffers are kept in sync via the
buffer-aware momentum update, since lightly's `update_momentum` only EMAs
parameters.

**Resume** (`resume.ssl_model`, `resume.type`): may point at any saved
bundle (`model_{epoch}.pt` or `model.pt`). `resume.type` (default
`continue`) selects the resume semantics:
- **`continue`** — exact same-data extension: the full model state (incl.
  heads + momentum/teacher nets), optimizer state, epoch counter (1-based),
  loss history, and DINOv2 component histories are restored; the run
  continues where it left off.
- **`transfer`** — domain transfer / pretrained-weight init (e.g. new data):
  only the model weights are carried over; the run restarts at epoch 0 with
  fresh schedules (LR warmup + cosine decay, teacher temperature warmup,
  momentum restarting at its start value) and a fresh optimizer. Loss
  history is reset.
Locked keys (bundle wins on mismatch): `method`, `backbone.name`,
`augmentation_views`, the `byol:`/`dinov2:` method block, `normalize.*`.
`backbone.in_chans` is not locked: on resume the bundle's `meta.in_chans` is
used, and a mismatch with the resolved `data.channels` hard-exits. Unknown
`resume.type` values hard-exit. `resume.ssl_model` must point at an SSL
bundle (meta has `method`) — a train bundle hard-exits with a clear message.

**UMAP check (`training.n_image_umap`, pretrain only)**: optional
per-interval visual monitoring. A fixed random image subset (picked ONCE at
run start with the training seed; max `training.n_image_umap` images;
disabled when 0/null) is re-embedded with a FRESH `UMAP(random_state=seed)`
at every `save_interval` epoch (right after `model_{epoch}.pt`) and at the
final epoch (unless it was already the last interval save). A pre-training
baseline is also written before the first epoch as `umap_check_epoch_0.pdf`
(0-based; the model BEFORE any training); it is skipped when actually
resuming with `resume.type=continue`. Features come from the online/student
backbone under the deterministic `augmentation_infer` pipeline, so per-epoch
PDFs are directly comparable. Output:
`<output_dir>/umap_check_epoch_{epoch}.pdf`. A failed check is logged and
skipped — it never aborts training.

### 7.4 Training flow

```
train(config)
   │
   ├─► for each root in data.root:
   │       CellDataset(root, channel_layout, image_pattern).build_metadata()
   │       _build_records_from_cell_dataset(...)   # one record per cell + label
   │
   ├─► resolve labels (label_from_dir | label_csv)   # labels honored verbatim
   ├─► validate: all roots must resolve to the SAME channel set (hard-exit)
   ├─► subsample(records, sample_max, sample_by, seed)
   ├─► stratified_split(records, val_ratio, seed)   # hard-exit on empty val split
   │
   ├─► if resume.ssl_model: _build_model_from_ssl(ssl_bundle, ...)
   │       DINOv2 bundle -> rebuild ViT via timm + cls-token pooling
   │       BYOL/conv bundle -> build_backbone + mean pooling
   │   else: build_backbone (timm pretrained, from scratch)
   ├─► ClassificationHead + FocalLoss + AdamW + AMP + early stopping (patience)
   │
   └─► save <output_dir>/{model.pt, model_{epoch}.pt (every save_interval),
                          training_plot.pdf, training_report.txt, <config>.yml}
```

**Train bundles** — `model.pt` (final) and `model_{epoch}.pt` (every
`training.save_interval` epoch; filename epoch is 1-based, no zero padding —
`model_9.pt` = 9th epoch completed) share one schema and every saved bundle
is complete — usable for exact resume (`resume.sl_model`) and inference.
`model.pt` stores the eval (best-validation) weights; interval bundles store
the raw training weights at that epoch (with matching optimizer state) so an
exact resume continues the trajectory. Compatible with `infer.py` classify
mode:

```python
{
    "state_dict": <eval Model state dict (model.pt) / training Model state dict (model_{epoch}.pt)>,
    "meta": {
        "class_names": [...], "num_classes": int,
        "channels": [...], "channel_layout": ..., "in_chans": int,
        "backbone": <timm name>,
        "ssl_method": "byol" | "dinov2" | None,   # None if from scratch
        "augmentation_train", "augmentation_infer",
        "normalize_method", "normalize_with_masking", "normalize_fixed_reference",
        "clip_low", "clip_high",
        "image_pattern": ...,
    },
    "config", "optimizer_state_dict", "epoch",
    "train_loss_history", "val_acc_history", "val_f1_history",
    "best_val", "best_state",
}
```

Note the train bundle's `epoch` key is **0-based** (the last completed loop
index; the filename `model_{epoch+1}.pt` is 1-based) — unlike SSL bundles,
whose `epoch` key is 1-based and matches the filename. Resuming across
bundle types, account for the off-by-one.

**DINOv2 backbone handling**: when `resume.ssl_model` points to a DINOv2
bundle (or when loading a train bundle with `meta.ssl_method == "dinov2"`),
the ViT is rebuilt via `backbone.build_dinov2_vit` with
`pos_embed="learn", dynamic_img_size=True, init_values=1e-5` and
**cls-token pooling** (`x[:, 0]`). For BYOL or conv backbones, `build_backbone`
is used with global mean pooling. The final eval-model rebuild (and the
saved `model.pt` meta `backbone`/`in_chans`) uses the backbone that was
ACTUALLY trained — the SSL bundle's `meta.backbone`/`meta.in_chans` when
transferring, NOT the train config's `model.backbone`.

**Resume** (`resume.sl_model`): may point at `model.pt` or any
`model_{epoch}.pt`. It must point at a TRAIN bundle (meta has
`num_classes`/`class_names`) — an SSL bundle hard-exits. Locked keys —
`model.backbone`, `model.pretrained`, `model.focal_gamma`,
`augmentation_train`, `augmentation_infer`, `normalize.*`. Fallback keys
(use new config if present, else bundle): `training.epochs`, `training.lr`,
`training.weight_decay`, `training.betas`, `training.patience`.
`resume.sl_model` and `resume.ssl_model` are mutually exclusive — if both
are set, `train` prints an error and hard-exits before any data loading.
When `resume.sl_model` is set, the model architecture and weights come
entirely from the checkpoint: the backbone is built with `pretrained=False`,
and `data.channels` is validated against the checkpoint's input channels
(inferred from its first conv/patch-embed weight, `shape[1]`) with a clean
hard-exit on mismatch.

### 7.5 Inference flow + DB schema

```
run_inference(config)
   │
   ├─► load bundle (model)
   ├─► dispatch on bundle keys:
   │     • "state_dict" + "num_classes" in meta -> classify model (train bundle)
   │     • "state_dict" (no num_classes) -> features-only (SSL bundle)
   │
   ├─► output flags from inference.pred_class / inference.feature:
   │     • pred_class (default = bundle is classify-capable): write pred/pred_prob.
   │       Explicit true with an SSL bundle -> warning, features only.
   │     • feature (default true): write the features BLOB.
   │
   ├─► dispatch on config.mode:
   │     • "single_cell"  -> CellDataset + _SingleCellInferDataset
   │     • "whole_image"  -> ImageDataset + WholeImageCellDataset
   │
   ├─► _forward_pass(loader, model, device, write_features, classify_mode, write_pred_class, pool_fn)
   │     classify_mode=True:  model(x) -> (logits, pooled)
   │     classify_mode=False: backbone(x) -> pool_fn -> features ; pred_class/pred_prob NULL
   │
   └─► _write_db(db_path, meta_rows, all_logits, all_features, ...)
```

DB path resolution:

- `output_dir` set ⇒ single shared DB at `{output_dir}/{db_name}` for all
  datasets. Config file is copied once to `{output_dir}/{config_name}.yml`.
- `output_dir` null (default) ⇒ per-dataset DB at `{data_dir}/{db_name}`.
  Config file is copied to each `{data_dir}/{config_name}.yml`.

In whole-image mode, `data.channel_layout: null` in the config takes
precedence over the bundle's `channel_layout`, allowing one-channel-per-file
inference even when the model was trained on multi-channel cell TIFFs.
`augmentation_infer` and `normalize` are always loaded from the bundle meta
— config files must not modify these. `normalize.fixed_reference` (bundle
meta `normalize_fixed_reference`) is a required meta key, honored by
inference datasets. Images are converted to float [0, 1] by the REQUIRED
`data.max_value` from the inference config (a data property of the new
dataset, like `channels`/`channel_layout` — the bundle's `max_value` is
provenance only).

All `data.*` settings in the inference config must be explicitly set (null is
allowed; missing is not).

**DB schema** — single `inference` table + 3 lazily-created reduction
tables. SQLite, **no `_meta`, no schema versioning**. Old DBs must be
deleted manually:
`inference` (uid, directory, filename, [mask_filename, label] for
whole-image, ground_truth, dynamic metadata TEXT cols, pred_class, pred_prob,
features BLOB), `reduction_pca`, `reduction_umap`, `reduction_pca_variance`.
`reduction_*` tables are written by `vis.show_reduction` (not `infer.py`),
fit-once-on-merged-DBs then applied per-DB. `show_reduction(config,
save_plots=True, raise_on_error=False)` accepts two opt-outs used by
microProfiler (data-only suite): `save_plots=False` skips the PDF scatters
(tables + fitted reducers are still written), and `raise_on_error=True`
converts logged-error abort paths into raised exceptions. Sampling
(`reduction.sample_per_class`) gates whichever reducer still needs FITTING —
a pre-fitted reducer is validated/transformed on all data. Save location:
`output_dir` when set, otherwise **every processed dataset's own directory**
(reducers + plots are written once per dataset dir so each dataset folder is
self-contained).

`inference` is written by `_write_db` with `INSERT` (one row per cell); `uid`
auto-increments. Re-running inference on an existing DB
**deletes the old `inference` rows first** (logged warning) — the data is
replaced, so stale `reduction_pca`/`reduction_umap` rows cannot reference
dead uids. The delete is scoped to the `directory` values of the rows being
written, so multiple datasets sharing one `output_dir` accumulate — each
dataset replaces only its own directories' rows. Deleting `infer.db` manually
is still required to change its schema (no migrations).

### 7.6 Config schemas

Five YAML configs under `microModel/configs/`:

- **`pretrain_byol.yml`** / **`pretrain_dinov2.yml`**: `mode` (single_cell
  only; whole_image planned), `method`, `data`
  (root, channels, channel_layout, image_pattern, `max_value` — REQUIRED,
  max possible intensity of the input dtype (65535 for 16-bit, 255 for
  8-bit); images are divided by it on load; label_from_dir/label_csv
  ignored, sample_max, sample_by), `backbone` (name — any timm model for
  BYOL, any timm ViT with a class token for DINOv2; `pretrained`, default
  `false` for both methods; `in_chans` is NOT a config key — derived from
  `data.channels`, stored in the bundle meta),
  `augmentation_views` (list of view specs; BYOL uses [0:2], DINOv2 uses
  [0:2] global + [2:] local — no `ToFloat` steps: the float [0, 1]
  conversion happens at load via `data.max_value`), `augmentation_infer`,
  `normalize` (method, with_masking, clip_low, clip_high, `fixed_reference`
  — true = the full clip+rescale+z-score transform computed once on the raw
  cell and applied as a fixed transform to every view, bit-identical to
  per-view normalization on un-augmented input),
  `dataloader`, `output_dir`, `resume.ssl_model`, `resume.type` (continue |
  transfer, default continue — see §7.3 Resume), `vis_augment`
  (num_samples; an optional `num_views` caps the grid, `null`/absent = all
  views; the preview grid shares one global x/y frame across all
  samples/views, so view size differences — e.g. 224 global vs 98 local —
  are visually apparent), `training` (optimizer, lr, weight_decay, betas,
  grad_clip, grad_accum_steps, batch_size — EVEN required for DINOv2,
  epochs, save_interval, n_image_umap — periodic UMAP check incl. an
  `umap_check_epoch_0.pdf` pre-training baseline (see §7.3), amp).
  `training.save_interval` writes `model_{epoch}.pt` bundles; the final
  epoch always writes `model.pt`.
  DINOv2 also has a `dinov2:` block (input_dim — null = derived from the ViT
  embed_dim, a non-null value must match or pretrain hard-exits —
  drop_path_rate, ibot_separate_head, head_batch_norm, warmup_epochs,
  transfer_warmup_epochs, teacher_temp_start/end, weight_decay_start/end,
  momentum_start/end, koleo_weight, lr_final). BYOL has a
  `byol:` block (proj_hidden_dim, proj_out_dim, pred_hidden_dim, pred_out_dim,
  warmup_epochs, transfer_warmup_epochs, momentum_start/end, lr_final).
- **`train.yml`**: `mode` (single_cell only; whole_image planned), `method`
  (classification only; segmentation planned), `data` (root, channels,
  channel_layout, image_pattern, `max_value` — REQUIRED (see pretrain),
  label_from_dir, label_csv, sample_max, sample_by), `output_dir`,
  `resume.sl_model` (train checkpoint; mutually exclusive with
  `resume.ssl_model` — hard-exit if both set), `resume.ssl_model` (SSL
  bundle for backbone transfer; null = from scratch), `augmentation_train`,
  `augmentation_infer`, `normalize` (same keys as pretrain, incl.
  `fixed_reference` — set true to match an SSL-pretrained backbone's
  fixed-reference feature statistics), `dataloader`, `model` (backbone,
  pretrained, focal_gamma — `backbone`/`pretrained` used only in true
  from-scratch mode: ignored when either resume key is set), `vis_augment`,
  `training` (batch_size, epochs, save_interval, lr, weight_decay, betas,
  val_ratio, amp, patience).
- **`infer_single_cell.yml`** / **`infer_whole_image.yml`**: `mode`,
  `model` (string path — SSL bundle OR train bundle), `data` (root,
  channels, channel_layout, image_pattern, `max_value` — REQUIRED, a data
  property of the new dataset (the bundle's `max_value` is provenance only)
  [, mask_pattern, image_subdir_pattern, mask_name for whole_image],
  label_from_dir, label_csv, sample_max, sample_by), `output_dir`,
  `dataloader`, `inference` (pred_class — default = bundle is
  classify-capable; explicit true with an SSL bundle warns and writes
  features only, feature — default true, db_name, batch_size), `reduction`
  (color_by, var_threshold — 0.95 default, auto-selected PCA dims covering
  this fraction of variance, floored at 2; also the UMAP preprocessing dim,
  sample_per_class, reducer_pca, reducer_umap). `augmentation_infer` +
  `normalize` are NOT in the config — they come from the bundle meta.

### 7.7 CLI usage

```powershell
micromodel pretrain --config configs/pretrain_byol.yml
micromodel pretrain --config configs/pretrain_dinov2.yml
micromodel vis-augment --config configs/pretrain_byol.yml
micromodel train --config configs/train.yml
micromodel vis-augment --config configs/train.yml
micromodel infer --config configs/infer_single_cell.yml
micromodel infer --config configs/infer_whole_image.yml
micromodel vis-reduction --config configs/infer_single_cell.yml
micromodel vis-reduction-interactive --config configs/infer_single_cell.yml --port 5000
```

Every module in `microModel` is consumed exclusively by the CLI. The only
"interactive" surface is the Flask server in `vis_interactive.py`
(`http://127.0.0.1:5000`), not a Qt GUI. The interactive viewer applies
`reduction.sample_per_class` sub-sampling (stratified by the first
`color_by`) when loading points from the DB. Viewer robustness: microBase
hard-exits (`sys.exit`) inside `_api_image` (missing/out-of-range channel
files, cell label absent from a re-read mask) are converted to JSON error
responses — a stale DB row or regenerated mask never kills the server.
Display conversion divides by the CONFIGURED `data.max_value` (both
"original" and "model" sizes), so 12-bit data renders identically in both
modes.

### 7.8 Adding a new SSL method

1. Create `src/microModel/models/<method>.py` with:
   - A model class (nn.Module) — use lightly heads/losses where possible.
   - `build_<method>(backbone_cfg, method_cfg, device) -> nn.Module`.
   - `train_step(model, batch, optimizer, epoch, total_epochs, device,
     criterion[, step_info]) -> float` (does its own backward + step).
2. Register in `models/__init__.py`:
   - Add `"<method>": <method>.build_<method>` to `_SSL_REGISTRY`.
   - Add a branch in `get_train_step` returning `<method>.train_step`.
   - Add a branch in `get_criterion` building the method's loss(es).
3. Add any method-specific config validation in `pretrain.pretrain_ssl`
   (e.g., view count checks like the BYOL ≥2 / DINOv2 ≥3 guards).
4. Add a `configs/pretrain_<method>.yml`.
5. Add a `<method>:` block in the config for method-specific hyperparameters;
   `pretrain_ssl` reads it via `config.get(method, {})`.

### 7.9 Adding a new augmentation

Register in `microBase.augment` via `@register`,
reference by name in the YAML `augmentation_views` / `augmentation_train` /
`augmentation_infer` lists. No microModel change needed.

---

## 8. Common workflows

### 8.1 Profile a fresh dataset

```powershell
microprofiler run --config pipeline_config.yml --dataset-dir D:\data\plate1
```

Produces `D:\data\plate1\result.db` (tables `image`, `cell`, ...),
`<stem>_cp_masks_cell.png` next to each image, and updates
`D:\data\plate1\session.yml`. Steps run only where the config sets
`run: true`.

### 8.2 Visually inspect + export cells for training

```powershell
microvis D:\data\plate1
```

- Patterns are pre-filled from `session.yml` (if present); click "Load
  Dataset" to scan the data.
- Channel colors are restored from `session.yml` on first load.
- Drag objects onto class boxes to annotate.
- Click "Export" → writes multi-channel TIFFs + `{mask_name}.csv` (e.g.
  `cell.csv`) under `D:\data\plate1\objects_exported\<class>\`. The CSV is
  appended on repeated exports of the same mask.
- Click "Reset" to clear the session and return to initial startup state.

### 8.3 Train a classifier on exported cells

Edit `configs/train.yml`:

```yaml
data:
  root:
    - D:\data\plate1\objects_exported\class_A
    - D:\data\plate1\objects_exported\class_B
  label_from_dir: true
```

```powershell
micromodel train --config configs/train.yml
micromodel vis-augment --config configs/train.yml
```

Produces `runs/model.pt`, `runs/training_plot.pdf`, `runs/augment_preview.pdf`.

### 8.4 Run inference on whole images

```powershell
micromodel infer --config configs/infer_whole_image.yml
micromodel vis-reduction --config configs/infer_whole_image.yml
micromodel vis-reduction-interactive --config configs/infer_whole_image.yml --port 5000
```

Produces `{output_dir}/infer.db` (1 `inference` table + 3 reduction tables),
`runs/feature_pca_*.pdf`, `runs/feature_umap_*.pdf`,
`runs/reducer_{pca,umap}.pkl`, and a browser UI at
`http://127.0.0.1:5000` for clicking individual cells.

### 8.5 SSL pretrain → train → infer

```powershell
micromodel pretrain --config configs/pretrain_dinov2.yml
micromodel vis-augment --config configs/pretrain_dinov2.yml
micromodel train --config configs/train.yml
micromodel infer --config configs/infer_whole_image.yml
micromodel vis-reduction --config configs/infer_whole_image.yml
micromodel vis-reduction-interactive --config configs/infer_whole_image.yml --port 5000
```

Set `resume.ssl_model: runs/model.pt` in `train.yml` to transfer the
SSL backbone into the classifier. Produces `runs/model.pt` (pretrain),
`runs/model.pt` (train), `{output_dir}/infer.db` (infer + reduction),
and a browser UI at `http://127.0.0.1:5000`.

---

## 9. Testing

Only `microBase` has a test suite (under `microMax/microBase/tests/`):

```powershell
cd c:\Users\haohe\GitHub\microMax\microBase
C:\Users\haohe\miniforge3\envs\micro\python.exe -m pytest tests/ -v
```

The three consumers rely on smoke tests via their CLIs:

```powershell
microprofiler --help
microvis --version
micromodel --help
```

When adding a feature to `microBase`, add a test under `tests/test_<module>.py`.
When adding a feature to a consumer, run its CLI with a small synthetic
dataset to smoke-test.

---

## 10. Quick reference — file locations

| What | Where |
|---|---|
| `microBase` source | `microMax/microBase/src/microBase/` |
| `microBase` tests | `microMax/microBase/tests/` |
| `microProfiler` source | `microMax/microProfiler/src/microProfiler/` |
| `microProfiler` example config | `microMax/microProfiler/examples/pipeline_config.yml` |
| `microVis` source | `microMax/microVis/src/microVis/` |
| `microModel` source | `microMax/microModel/src/microModel/` |
| `microModel` configs | `microMax/microModel/configs/` |
| Per-dataset state | `<dataset>/session.yml` |
| Per-dataset BaSiC models | `<dataset>/.microprofiler/BaSiC_model/` |
| Profiling DB | `<dataset>/result.db` |
| Inference DB (per-dataset) | `<dataset>/infer.db` (or any per-block `output_db` name) |
| Inference DB (shared) | `{output_dir}/infer.db` |
| Fitted reduction reducers | `<dataset>/reducer_{pca,umap}.pkl` (microProfiler inference step; microModel CLI saves them under `output_dir` or each dataset dir) |
| microVis log | `%TEMP%/microVis.log` |
| microModel training output | `<output_dir>/` (default `runs/`) |
| microModel pretrain bundles | `<output_dir>/model.pt` (final) + `model_{epoch}.pt` (every `save_interval`; 1-based, no zero padding), default `runs/` |
| microModel pretrain UMAP check | `<output_dir>/umap_check_epoch_{epoch}.pdf` (epoch 0 baseline + one per `save_interval` + final, 1-based, when `training.n_image_umap` > 0; no epoch 0 for `resume.type=continue`) |
| microModel pretrain loss curve | `<output_dir>/loss_curve.pdf` (always written when loss history is non-empty) |
| microModel train bundles | `<output_dir>/model.pt` (final) + `model_{epoch}.pt` (every `save_interval`; 1-based, no zero padding), default `runs/` |
| microModel run log | `<output_dir>/micromodel.log` (append; also for infer/vis when `output_dir` is set) |

---

## 11. When you change something — checklist

- **Overarching rule:** Every time you change, optimize, add, or remove a
  feature or an input/output format, update the relevant sections of this
  file (`AGENTS.md`) to keep it in sync with the code. Stale docs are worse
  than no docs.
- **Changed a `microBase` public symbol?** Bump `microBase.__version__`,
  update tests, re-install all four packages.
- **Changed a config schema?** Update the relevant section in this file
  (§5.3, §7.6) and the example YAML.
- **Changed a DB schema?** Document it in §5.4 or §7.5. Remember: no
  migrations — users delete the old DB.
- **Added a new module?** Add a row in the relevant package's module map
  (§4.1, §5.1, §6.1, §7.1) and update the dependency table.
- **Renamed a public symbol?** Update the cross-cutting naming list in §3.7
  and grep the whole `microMax/` tree for stale references.
- **Added a new console subcommand?** Update §5/§7 CLI tables and the
  `pyproject.toml` `[project.scripts]` entry.
