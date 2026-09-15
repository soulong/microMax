"""microModel label — interactive multi-label annotation web app.

`micromodel label --config <yml>` opens a local grid-batch labeling page
(Flask) over pre-cropped single-cell folders or file lists, with
model-assisted suggestions (kNN + classify head + per-label ML), an
auto-annotate pass, an exact undo and a training-ready CSV export.

Subpackage layout:
  db        AnnotationDB — the single append-friendly SQLite project DB
  engines   SuggestEngine (kNN), MLModelEngine (active learning),
            review_items (mislabel consistency check), diverse ordering
  features  cached per-root feature/probability extraction (train bundle;
            the SSL variant is deduplication's, re-used here)
  imaging   TIFF crop -> percentile-normalized PNG + a small render LRU
  server    LabelServer — config parsing, the Flask API, workflow logic
  ui        the embedded single-page UI (HTML/CSS/JS)
"""

from .db import (DB_NAME, PALETTE, STATE_NEG, STATE_POS, AnnotationDB,
                 migrate_project_db)
from .engines import MLModelEngine, SuggestEngine, _diverse_order, review_items
from .imaging import _render_png
from .server import LabelServer, run_label

__all__ = [
    "DB_NAME", "PALETTE", "STATE_POS", "STATE_NEG", "AnnotationDB",
    "migrate_project_db", "SuggestEngine", "MLModelEngine", "_diverse_order",
    "review_items", "_render_png", "LabelServer", "run_label",
]
