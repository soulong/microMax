"""Image rendering for the label app — TIFF crop -> percentile-normalized PNG.

The server renders cropped single cells as PNGs on demand. Rendering a
page of K thumbnails re-reads K TIFFs and re-runs the geometric model
pipeline every time — selection clicks, page flips back and forth and
zoom toggles re-request the same cells constantly, so a small LRU over
the rendered bytes sits in front of the pipeline
(:class:`RenderCache`, keyed by everything that changes the pixels).
"""

import io
import threading
from collections import OrderedDict

import numpy as np

# Bump when the rendering pipeline changes so browsers and the in-process
# cache drop renders of the old pipeline.
RENDER_VERSION = 3

# Composite colors for multi-channel display (channel 1..8, additive RGB).
CHANNEL_COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 1.0),
                  (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0),
                  (1.0, 0.5, 0.0), (1.0, 1.0, 1.0)]


def _render_png(img_hwc, max_px, lo_pct=0.1, hi_pct=99.9, gamma=1.0):
    """Percentile-normalize a float [0,1] HWC crop and encode a PNG.

    Per-channel robust range (lo_pct / hi_pct percentiles, default
    0.1 / 99.9) computed over the NONZERO pixels — cropped cells have
    exact-zero background, and including it would wash the cell out.
    Narrowing the window (e.g. 2 / 95) raises display contrast; gamma > 1
    brightens midtones, gamma < 1 darkens (out = in ** 1/gamma — display
    only, features always use the raw intensities). One channel renders
    grayscale; several channels are additively composited with
    CHANNEL_COLORS.
    """
    from PIL import Image

    # Server-side clamps: the query params are user input.
    lo_pct = min(49.0, max(0.0, float(lo_pct)))
    hi_pct = min(100.0, max(51.0, float(hi_pct)))
    gamma = min(5.0, max(0.2, float(gamma)))
    inv_gamma = (1.0 / gamma) if gamma else 1.0

    h, w, c = img_hwc.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(c):
        ch = img_hwc[:, :, i]
        vals = ch[ch > 0]
        if vals.size:
            lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            norm = norm ** inv_gamma
        if c == 1:
            canvas[..., 0] = canvas[..., 1] = canvas[..., 2] = norm
        else:
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            canvas += norm[..., None] * np.asarray(color, dtype=np.float32)
    u8 = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

    img = Image.fromarray(u8)
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class RenderCache:
    """Small thread-safe LRU over rendered PNG bytes.

    Keyed by (path, mtime, channels, layout, max_value, max_px, lo, hi,
    gamma, has_aug, RENDER_VERSION) — any change to the pixels produces a
    new key, so a stale entry can never be served; the file mtime covers
    edits to the TIFF itself. In-process only (cleared on restart, which
    is exactly when the model pipeline can change).
    """

    def __init__(self, capacity=256):
        self.capacity = int(capacity)
        self._store = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)
