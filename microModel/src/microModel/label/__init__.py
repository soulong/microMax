"""microModel label — interactive multi-label annotation web app.

`micromodel label --config <yml>` opens a local grid-batch labeling page
(Flask) over pre-cropped single-cell folders or file lists, with
model-assisted recommendation scores (exemplar kNN + per-label logistic
refits + classify head), cluster-assisted bulk labeling (Leiden over the
shared embedding space, whole-cluster writes), an exact undo and a
training-ready CSV export.

Subpackage layout:
  db        AnnotationDB — the append-friendly SQLite project DB
  engines   SuggestEngine (kNN exemplar scoring),
            fit_label_model (per-label logistic refit for the manual
            Refresh-model action), review_items (mislabel consistency
            check), diverse ordering
  features  cached per-root feature/probability extraction (train bundle;
            the SSL variant is deduplication's, re-used here)
  cluster   Leiden clusters over the shared space: resolution search for a
            target cluster count, medoids, npz cache — the data behind the
            Clusters tab and the whole-cluster writes
  imaging   TIFF crop -> percentile-normalized PNG + a small render LRU
  server    LabelServer — config parsing, the Flask API, workflow logic
  ui        the embedded single-page UI (HTML/CSS/JS)
"""

from .db import (DB_NAME, PALETTE, STATE_NEG, STATE_POS, AnnotationDB,
                 migrate_project_db)
from .cluster import build_or_load, medoid_rows, resolve_resolution
from .engines import (MIN_FIT_NEG, MIN_FIT_POS, SuggestEngine,
                      _diverse_order, fit_label_model, review_items)
from .imaging import _render_png
from .server import LabelServer, run_label

__all__ = [
    "DB_NAME", "PALETTE", "STATE_POS", "STATE_NEG", "AnnotationDB",
    "migrate_project_db", "SuggestEngine", "_diverse_order",
    "fit_label_model", "review_items", "MIN_FIT_POS", "MIN_FIT_NEG",
    "_render_png", "LabelServer", "run_label",
    "build_or_load", "medoid_rows", "resolve_resolution",
]

