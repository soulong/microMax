"""microModel bridge: concentrates every piece of microModel's contract.

Everything that knows microModel's bundle schema or inference-config dict
lives here, so a microModel schema change touches exactly one file. microModel
is imported lazily (only when inference actually runs), keeping microProfiler
importable without microModel installed.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from pathlib import Path

from microBase import MicroMaxError
from microBase.db_contracts import FIND_CLUSTER_TABLE, reduction_table_name
from microProfiler.config import PipelineConfig, resolve_inference_db

logger = logging.getLogger(__name__)


def _detect_reducer_kind(path) -> str:
    """Probe a pickled reducer and return its DR method (pca/umap/pacmap/localmap).

    microModel's pre-fit reducer keys are per-method (reduction_<method>),
    so the single user-chosen reducer file must be mapped to the right key.
    Detection is by runtime type name — no umap/pacmap import needed:
      PCA instance                 -> pca
      {"pca_pre": PCA, "umap": …}  -> umap (microModel's UMAP pipeline dict)
      PaCMAP / LocalMAP instance   -> pacmap / localmap
    Raises ValueError for anything else (callers surface it as a per-dataset
    failure / GUI popup).
    """
    import pickle

    with open(path, "rb") as f:
        obj = pickle.load(f)
    name = type(obj).__name__
    if name == "PCA":
        return "pca"
    if isinstance(obj, dict) and obj.get("umap") is not None:
        return "umap"
    if name == "PaCMAP":
        return "pacmap"
    if name == "LocalMAP":
        return "localmap"
    raise ValueError(
        f"Not a recognized reducer pickle (pca/umap/pacmap/localmap): {path} "
        f"(got {type(obj).__name__})"
    )


def _reduction_run_state(entry):
    """(enabled, cluster_active) — whether reduction and/or cluster prediction runs.

    enabled: the Dimension-reduction group. cluster_active: the Cluster
    group checked AND a cluster.pkl chosen (prediction needs both).
    """
    red = entry.reduction
    cluster_active = bool(red.cluster_enabled and red.cluster)
    return bool(red.enabled or cluster_active), cluster_active


def expected_reduction_tables(entry) -> frozenset:
    """Tables run_reduction will write for one entry's reduction config.

    One reduction_<method> table per configured method plus find_cluster
    when cluster prediction runs. Mirrors _build_mm_inference_config's method
    selection exactly, so the completeness check can never demand a table the
    bridge will not write (or miss one it will).
    """
    if not entry.reduction:
        return frozenset()
    enabled, cluster_active = _reduction_run_state(entry)
    if not enabled:
        return frozenset()
    red = entry.reduction
    if red.reducer:
        methods = set()
        for path in red.reducer:
            try:
                methods.add(_detect_reducer_kind(path))
            except OSError:
                # Unreadable path: the run itself will fail loudly; demanding
                # no tables keeps the completeness check from false-skipping.
                return frozenset()
            except Exception as e:
                # Corrupt/unknown pickle: returning "nothing expected" would
                # silently mark the dataset complete and skip it. Raise so the
                # caller reprocesses and surfaces the real error.
                raise RuntimeError(
                    f"Cannot read reducer pickle {path}: {e}") from e
    else:
        # Same defaulting as _build_mm_inference_config: explicit method wins
        # (an EMPTY list means nothing fitted), otherwise [pca] for a
        # cluster-only run and [pca, umap] when the dimension-reduction
        # group itself is enabled. A checked Cluster group keeps its PCA
        # reference even when no DR method is selected.
        methods = set(
            red.method if red.method is not None
            else (["pca", "umap"] if red.enabled else ["pca"])
        )
        if not methods and cluster_active:
            methods = {"pca"}
    tables = {reduction_table_name(m) for m in methods}
    if cluster_active or red.cluster_res:
        tables.add(FIND_CLUSTER_TABLE)
    return frozenset(tables)


def run_mm_inference(mm_cfg, **kwargs):
    """Lazily import and call microModel.infer.run_inference (see _call_micromodel)."""
    from microModel.infer import run_inference
    return run_inference(mm_cfg, **kwargs)


def run_mm_reduction(mm_cfg, **kwargs):
    """Lazily import and call microModel.reduction.run_reduction (see _call_micromodel)."""
    from microModel.reduction import run_reduction
    return run_reduction(mm_cfg, **kwargs)


class _ProgressTee(io.TextIOBase):
    """Tee microModel's stderr to three places at once:

    - the real terminal (tqdm renders exactly like `micromodel infer`),
    - the error buffer (kept for the failure message on SystemExit),
    - the progress collector: tqdm lines (``Infer:  33%|██ 1/3 [..]``) become
      ``report(step_key, cur, tot)`` (a real status-bar bar), any other text
      line becomes ``report(step_key, 0, 0, text)`` (a status message).
    """

    _TQDM_RE = re.compile(r"^[^:\s]+:\s*\S")

    def __init__(self, err_buf, real_stderr, progress, step_key):
        super().__init__()
        self._buf = err_buf
        self._real = real_stderr
        self._progress = progress
        self._step_key = step_key
        self._pending = ""

    def _emit(self, seg: str) -> None:
        seg = re.sub(r"\x1b\[[0-9;]*m", "", seg).strip()
        if not seg:
            return
        if self._TQDM_RE.match(seg):
            m = re.search(r"(\d+)\s*/\s*(\d+)", seg)
            if m:
                self._progress.report(
                    self._step_key, int(m.group(1)), int(m.group(2)), "")
                return
        self._progress.report(self._step_key, 0, 0, seg[:200])

    def write(self, s: str) -> int:
        if self._real is not None:
            try:
                self._real.write(s)
                self._real.flush()
            except Exception:
                pass
        self._buf.write(s)
        self._pending += s
        parts = self._pending.replace("\r", "\n").split("\n")
        self._pending = parts.pop()
        for part in parts:
            self._emit(part)
        return len(s)

    def flush(self) -> None:
        # Emit a trailing partial line (no newline seen yet) so its status
        # text is not silently lost.
        if self._pending:
            pending, self._pending = self._pending, ""
            self._emit(pending)
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(self._real is not None and self._real.isatty())

    def fileno(self):
        if self._real is not None and hasattr(self._real, "fileno"):
            return self._real.fileno()
        raise io.UnsupportedOperation("fileno")


class _MicroModelLogForwarder(logging.Handler):
    """Forward microModel logger records (e.g. "Fitting PCA + UMAP...",
    "Writing to <db>") to the real terminal with the CLI's format and to the
    progress collector as status messages — so the reduction stage is visible
    both in the terminal and in the GUI status bar.
    """

    def __init__(self, real_stderr, progress, step_key):
        super().__init__(level=logging.INFO)
        self._real = real_stderr
        self._progress = progress
        self._step_key = step_key
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            if self._real is not None:
                self._real.write(line)
                self._real.flush()
            msg = record.getMessage().strip()
            if msg:
                self._progress.report(self._step_key, 0, 0, msg[:200])
        except Exception:
            self.handleError(record)


def _call_micromodel(fn, mm_cfg, err_prefix: str, progress=None,
                     step_key: str = "Inference", **kwargs):
    """Call a microModel function, converting its MicroMaxError (library
    exceptions) into RuntimeError with the captured stderr message, so the GUI
    worker surfaces a popup and the CLI treats the dataset as failed.

    SystemExit is still caught defensively (a dependency may call sys.exit).
    When a progress collector is given, microModel's stderr is teed to the
    real terminal AND forwarded to the collector (live tqdm progress + status
    lines); microModel's INFO logs are forwarded the same way. When the run
    was cancelled (progress.cancel_check), the failure is re-raised as
    InterruptedError — a cancel is not a failure.
    """
    import contextlib

    real_stderr = sys.stderr
    err_buf = io.StringIO()
    log_handler = None
    prev_level = None
    mm_logger = logging.getLogger("microModel")
    if progress is not None:
        tee = _ProgressTee(err_buf, real_stderr, progress, step_key)
        log_handler = _MicroModelLogForwarder(real_stderr, progress, step_key)
        prev_level = mm_logger.level
        mm_logger.setLevel(logging.INFO)
        mm_logger.addHandler(log_handler)
    else:
        tee = err_buf
    try:
        with contextlib.redirect_stderr(tee):
            fn(mm_cfg, **kwargs)
    except (MicroMaxError, SystemExit) as e:
        if (log_handler is not None
                and getattr(progress, "cancel_check", None)
                and progress.cancel_check()):
            raise InterruptedError from e
        msg = err_buf.getvalue().strip() or str(e) or f"microModel failed ({e})"
        raise RuntimeError(f"{err_prefix}: {msg}") from e
    finally:
        if log_handler is not None:
            mm_logger.removeHandler(log_handler)
            if prev_level is not None:
                mm_logger.setLevel(prev_level)


def _build_mm_inference_config(entry, cfg: PipelineConfig, ds, root_dir: Path) -> dict:
    """Build the microModel whole-image inference config dict for one block.

    output_dir is always null so the inference DB (and, with reduction, the
    fitted reducer pickles) land under the dataset dir itself.

    ``entry.channels`` must be a non-empty ordered list — the pipeline skips
    blocks with no channels before calling this bridge, and an empty/null
    list must never reach microModel as "use all channels".
    """
    intensity_cols = ds.intensity_colnames
    if not entry.channels:
        raise RuntimeError(
            "Inference channels must be configured for every block "
            "(a block with no channels is skipped before inference)."
        )
    channels = []
    for ch in entry.channels:
        try:
            channels.append(intensity_cols.index(ch) + 1)
        except ValueError:
            raise RuntimeError(
                f"Inference channel {ch!r} not found in dataset channels "
                f"{intensity_cols}."
            )
    mm_cfg = {
        "mode": "whole_image",
        "model": os.path.abspath(entry.model),
        "output_dir": None,
        # GUI inference uses 4 DataLoader worker processes (spawned thanks to
        # the picklable ImageDataset cache — §4.3) with the same prefetch as
        # microModel's CLI. persistent_workers=False tears the workers down
        # after each inference pass (clean teardown on completion/cancel);
        # tqdm/log output still comes from the pipeline worker thread and is
        # teed to the terminal + GUI status bar.
        "dataloader": {"num_workers": 4, "prefetch_factor": 2,
                       "persistent_workers": False},
        "data": {
            "file_dir": [str(root_dir)],
            "channels": channels,
            "channel_layout": None,
            "image_pattern": cfg.image_pattern or ds.image_pattern,
            "mask_pattern": cfg.mask_pattern or ds.mask_pattern,
            "image_subdir_pattern": cfg.image_subdir_pattern or ds.image_subdir_pattern,
            "mask_name": entry.mask_name,
            "max_value": float(entry.max_value),
            "label_from_dir": False,
            "label_csv": None,
            "sample_max": None,
            "sample_by": "per_dataset",
        },
        "inference": {
            "pred_class": bool(entry.pred_class),
            "feature": bool(entry.feature),
            "db_name": resolve_inference_db(entry),
            "batch_size": 128,
        },
    }
    if entry.reduction:
        enabled, cluster_active = _reduction_run_state(entry)
        if not enabled:
            return mm_cfg
        red_cfg = {
            "color_by": entry.reduction.color_by,
            "cluster": entry.reduction.cluster if cluster_active else None,
            "cluster_res": entry.reduction.cluster_res,
            "sample_per_class": entry.reduction.sample_per_class if entry.reduction.sample_per_class is not None else 10000,
        }
        if entry.reduction.reducer:
            # One or more user-chosen reducer pickles — detect each kind and
            # map to microModel's per-method pre-fit keys (raises ValueError
            # for an unrecognized pickle). The DR-method selection is ignored
            # in this mode: every listed reducer transforms directly, in
            # microModel's canonical pca -> umap -> pacmap -> localmap order.
            kinds = []
            for path in entry.reduction.reducer:
                kind = _detect_reducer_kind(os.path.abspath(str(path)))
                if kind in kinds:
                    raise ValueError(
                        f"Two reducer pickles share the DR method '{kind}' "
                        f"(one reduction_<method> table per method) — "
                        f"remove the duplicate: {path}")
                kinds.append(kind)
                red_cfg[f"reduction_{kind}"] = os.path.abspath(str(path))
            red_cfg["method"] = kinds
        else:
            # No reducer: fit fresh. An explicit EMPTY method list means
            # "nothing fitted" (the GUI default) — microModel would treat an
            # empty list as its pca+umap fallback, so the reduction stage is
            # simply not scheduled. A checked Cluster group still needs the
            # PCA reference embedding and falls back to [pca]; the None
            # (absent key) fallback keeps the legacy default for CLI configs.
            if entry.reduction.method is not None:
                methods = list(entry.reduction.method)
            else:
                methods = (["pca", "umap"] if entry.reduction.enabled else ["pca"])
            if not methods:
                if cluster_active:
                    methods = ["pca"]
                else:
                    return mm_cfg
            red_cfg["method"] = methods
        mm_cfg["reduction"] = red_cfg
    return mm_cfg


def read_bundle_meta(model_path: str) -> dict:
    """Load a microModel bundle's meta dict (lazy torch import).

    Raises ImportError when microModel/torch are missing and RuntimeError for
    anything that is not a microModel bundle — callers surface these as popups.
    """
    import importlib.util
    if importlib.util.find_spec("microModel") is None:
        raise ImportError(
            "Inference requires the 'microModel' package, which is not "
            "installed. Install microModel and restart microProfiler."
        )
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Inference requires 'torch' (installed with microModel), which is "
            "not installed."
        ) from e
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise RuntimeError("Not a microModel bundle (missing 'state_dict').")
    meta = bundle.get("meta") or {}
    if not isinstance(meta, dict):
        raise RuntimeError("Bundle 'meta' is not a dict.")
    return meta
