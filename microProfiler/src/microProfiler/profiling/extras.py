"""Extra property factories for regionprops_table.

Features: radial distribution, granularity spectrum, GLCM texture, Pearson correlation.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, mean as nd_mean
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, dilation, erosion, reconstruction
from skimage.transform import resize

logger = logging.getLogger(__name__)


def _named(fn, name: str):
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. RADIAL DISTRIBUTION  (CellProfiler MeasureObjectIntensityDistribution port)
# ═══════════════════════════════════════════════════════════════════════════


def _radial_features_one_object(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    nbins: int,
) -> np.ndarray:
    """Compute FracAtD, MeanFrac, RadialCV for a single object.

    Returns an ``(3, nbins)`` array:
      row 0 = FracAtD  (fraction of total intensity in each ring)
      row 1 = MeanFrac (intensity per pixel, normalised by ring area)
      row 2 = RadialCV (coefficient of variation across 8 angular wedges)

    Ring ordering: bin 0 = innermost (centre), bin nbins-1 = outermost (edge).
    Boundary pixels (distance to edge ≈ 0) are clipped into the outermost ring.
    """
    mask = regionmask.astype(bool)
    img = intensity.astype(float)
    nbins = int(nbins)
    if nbins < 1:
        raise ValueError("nbins must be >= 1")

    if not mask.any():
        return np.zeros((3, nbins), dtype=float)

    # Distance from each pixel to the boundary of the object.
    d_to_edge = distance_transform_edt(mask)
    # Medial axis: the point farthest from the edge.
    cy, cx = np.unravel_index(int(np.argmax(d_to_edge)), d_to_edge.shape)
    # Distance from each pixel to the medial axis point (Euclidean).
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(float)
    d_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # CP scaled normalisation: 0 = centre, 1 = edge. The +1e-3 keeps the
    # denominator positive at boundary pixels (d_to_edge == 0).
    norm = d_from_center / (d_from_center + d_to_edge + 1e-3)

    bin_idx = np.clip(np.floor(norm * nbins).astype(int), 0, nbins - 1)
    # The mask is True for object pixels, False for background. We work only
    # on the object pixels.
    obj_pixels = mask
    obj_intensity = img[obj_pixels]
    obj_bin = bin_idx[obj_pixels]
    obj_dx = xx[obj_pixels] - cx
    obj_dy = yy[obj_pixels] - cy
    obj_d_to_edge = d_to_edge[obj_pixels]

    total_intensity = obj_intensity.sum()
    total_pixels = obj_intensity.size
    if total_intensity == 0 or total_pixels == 0:
        return np.zeros((3, nbins), dtype=float)

    fracatd = np.zeros(nbins, dtype=float)
    meanfrac = np.zeros(nbins, dtype=float)
    radialcv = np.zeros(nbins, dtype=float)

    for b in range(nbins):
        sel = obj_bin == b
        if not sel.any():
            continue
        bin_intensity_sum = obj_intensity[sel].sum()
        bin_pixel_count = int(sel.sum())
        fracatd[b] = bin_intensity_sum / total_intensity
        bin_pixel_frac = bin_pixel_count / total_pixels
        meanfrac[b] = fracatd[b] / (bin_pixel_frac + np.finfo(float).eps)

        # RadialCV: 8 angular wedges, mean intensity per wedge, then CV.
        bx = obj_dx[sel]
        by = obj_dy[sel]
        bi = obj_intensity[sel]
        # 8 wedges by (sign(dx), sign(dy), |dx|>|dy|) → index in [0, 7].
        wedge_idx = (
            (bx > 0).astype(int)
            + (by > 0).astype(int) * 2
            + (np.abs(bx) > np.abs(by)).astype(int) * 4
        )
        wedge_sums = np.bincount(wedge_idx, weights=bi, minlength=8)
        wedge_counts = np.bincount(wedge_idx, minlength=8).astype(float)
        present = wedge_counts > 0
        if present.sum() < 2:
            radialcv[b] = 0.0
        else:
            wedge_means = np.zeros(8)
            wedge_means[present] = wedge_sums[present] / wedge_counts[present]
            mu = wedge_means[present].mean()
            if mu > 0:
                radialcv[b] = float(wedge_means[present].std() / mu)
            else:
                radialcv[b] = 0.0

    # Internal bin 0 = innermost (norm=0 at centre), bin nbins-1 = outermost.
    return np.stack([fracatd, meanfrac, radialcv], axis=0)


def _radial_all(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    nbins: int,
) -> np.ndarray:
    """Return a flat ``(3*nbins,)`` array for regionprops:
    [FracAtD_1..N, MeanFrac_1..N, RadialCV_1..N].
    Bin 1 = innermost, bin N = outermost.
    """
    stacked = _radial_features_one_object(regionmask, intensity, nbins=nbins)
    return stacked.reshape(-1)


def make_radial_distribution(
    nbins: int = 4,
    ch_name: str = "ch0",
) -> list:
    """Create radial distribution callables for regionprops_table.

    Generates three features per ring:
      * ``radial_fracatd_bin{b}of{n}_{ch}``  — FracAtD
      * ``radial_meanfrac_bin{b}of{n}_{ch}`` — MeanFrac
      * ``radial_radialcv_bin{b}of{n}_{ch}`` — RadialCV

    Ring ordering is inner→outer (bin 1 is the innermost ring, closest to the
    object centre), matching CellProfiler's ``FracAtD_<b>of<n>_<image>``.
    """
    def _compute(mask, intensity):
        return _radial_all(mask, intensity, nbins=nbins)

    fns = []
    n_features = 3
    for feat_idx, feat_name in enumerate(("fracatd", "meanfrac", "radialcv")):
        for b in range(nbins):
            idx = feat_idx * nbins + b

            def _fn(mask, intensity, _idx=idx):
                return float(_compute(mask, intensity)[_idx])

            fns.append(_named(_fn, f"radial_{feat_name}_bin{b + 1}of{nbins}_{ch_name}"))
    return fns


# ═══════════════════════════════════════════════════════════════════════════
# 2. GRANULARITY  (CellProfiler MeasureGranularity port)
# ═══════════════════════════════════════════════════════════════════════════


def _subsample(image, mask, factor):
    """Subsample image & mask by ``factor`` using map_coordinates (CP style).

    Returns (pixels, new_mask, new_shape).
    """
    h, w = image.shape
    new_shape = (max(1, int(round(h * factor))), max(1, int(round(w * factor))))
    if new_shape == (h, w):
        return image.astype(float).copy(), mask.astype(bool).copy(), new_shape
    i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / factor
    pixels = map_coordinates(image.astype(float), (i, j), order=1)
    mask_resampled = map_coordinates(mask.astype(float), (i, j), order=1) > 0.9
    return pixels, mask_resampled, new_shape


def _resample_to(image, target_shape, src_shape):
    """Map a (src_shape) image back to (target_shape) using map_coordinates."""
    if src_shape == target_shape:
        return image
    if target_shape[0] == 1 or target_shape[1] == 1:
        return np.full_like(image, np.nan)
    i, j = np.mgrid[0:target_shape[0], 0:target_shape[1]].astype(float)
    i *= float(src_shape[0] - 1) / float(target_shape[0] - 1)
    j *= float(src_shape[1] - 1) / float(target_shape[1] - 1)
    return map_coordinates(image, (i, j), order=1)


def _granularity_per_object_full(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    spectrum_length: int,
    subsample_size: float,
    image_sample_size: float,
    background_radius: int,
) -> dict:
    """Per-object granularity spectrum from the FULL image (CP-style port).

    Mirrors ``MeasureGranularity.run`` in cellprofiler/modules/measuregranularity.py:

    1. Subsample image & mask by ``subsample_size`` (map_coordinates, order=1).
    2. Tophat background reduction using ``image_sample_size`` and a disk of
       ``background_radius``: subsample, erode, dilate, map back, subtract
       from the granularity image (clip negatives to zero).
    3. For each step i in 1..N:
       a. Erode once with ``disk(1)`` on the subsampled grid.
       b. Geodesic reconstruction of the eroded image under the
          background-subtracted image.
       c. Resample the reconstructed image back to the original grid
          (``map_coordinates`` with order=1, using the inverse of
          ``subsample_size``).
       d. For each labelled object, compute the per-object mean of the
          resampled reconstructed image, restricted to the image mask.
       e. ``GS_n = (prev_mean - current_mean) * 100 / start_mean``, where
          ``start_mean`` is the per-object mean of the original
          (pre-erosion) image within the object.

    Returns a dict ``{label: spectrum_array}`` where each array has shape
    ``(spectrum_length,)``.
    """
    spectrum_length = int(spectrum_length)
    subsample_size = float(subsample_size)
    image_sample_size = float(image_sample_size)
    background_radius = int(background_radius)

    if spectrum_length < 1:
        raise ValueError("spectrum_length must be >= 1")
    if not (0 < subsample_size <= 1):
        raise ValueError("subsample_size must be in (0, 1]")
    if not (0 < image_sample_size <= 1):
        raise ValueError("image_sample_size must be in (0, 1]")
    if background_radius < 1:
        raise ValueError("background_radius must be >= 1")

    labels_full = regionmask.astype(np.int32)
    intensity_full = intensity.astype(float)
    orig_shape = intensity_full.shape
    if labels_full.shape != orig_shape:
        raise ValueError("regionmask and intensity must share shape")

    unique_labels = np.unique(labels_full)
    unique_labels = unique_labels[unique_labels != 0]
    if unique_labels.size == 0:
        return {}

    # Image mask: True where the original image is valid (background 0 allowed).
    img_mask = np.ones(orig_shape, dtype=bool)
    # Per-object start_mean uses the original pixel data, restricted to the
    # image mask. CP zeros the labels outside the image mask.
    labels_masked = np.where(img_mask, labels_full, 0)

    start_means_full = nd_mean(
        intensity_full, labels_masked, unique_labels,
    )
    start_means = np.asarray(start_means_full, dtype=float)
    start_means = np.where(
        np.isfinite(start_means) & (start_means > 0),
        start_means,
        np.finfo(float).eps,
    )

    # Subsample to the granularity grid.
    pixels, gran_mask, new_shape = _subsample(
        intensity_full, img_mask, subsample_size,
    )

    # Background reduction (tophat) on the granularity grid.
    if image_sample_size < 1:
        back_pixels, back_mask, back_shape = _subsample(
            pixels, gran_mask, image_sample_size,
        )
    else:
        back_pixels = pixels
        back_mask = gran_mask
        back_shape = new_shape

    se = disk(background_radius, dtype=bool)
    back_pixels_masked = np.where(back_mask, back_pixels, 0.0)
    back_eroded = erosion(back_pixels_masked, footprint=se)
    back_dilated = dilation(np.where(back_mask, back_eroded, 0.0), footprint=se)

    if image_sample_size < 1:
        back_dilated = _resample_to(back_dilated, new_shape, back_shape)

    pixels = pixels - back_dilated
    pixels[pixels < 0] = 0
    pixels[~gran_mask] = 0

    fp = disk(1, dtype=bool)
    ero = pixels.copy()

    # object_records: per-object cumulative current mean (resampled to
    # original resolution). Initialised to start_mean.
    current_means = start_means.copy()
    spectra = {int(lbl): np.zeros(spectrum_length, dtype=float) for lbl in unique_labels}

    for i in range(1, spectrum_length + 1):
        ero_masked = np.where(gran_mask, ero, 0.0)
        ero = erosion(ero_masked, footprint=fp)
        rec = reconstruction(ero, pixels, footprint=fp)
        # Resample the reconstructed image back to the original grid.
        rec_original = _resample_to(rec, orig_shape, new_shape)
        # Per-object mean of the resampled reconstructed image, restricted
        # to the image mask. CP zeros labels outside the mask.
        new_means = np.asarray(
            nd_mean(rec_original, labels_masked, unique_labels), dtype=float,
        )
        new_means = np.where(np.isfinite(new_means), new_means, 0.0)
        # Spectrum bin for this step: (prev - current) * 100 / start.
        step_values = (current_means - new_means) * 100.0 / start_means
        for k, lbl in enumerate(unique_labels):
            spectra[int(lbl)][i - 1] = float(step_values[k])
        current_means = new_means

    return spectra


def make_granularity(
    spectrum_length: int = 16,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    background_radius: int = 10,
    ch_name: str = "ch0",
):
    """Create a granularity callable that operates on the full image.

    Returns a :class:`_GranularityComputer` whose ``__call__`` signature is
    ``(mask, intensity) -> dict[int, np.ndarray]`` and that should be
    invoked once per (image, channel) with the FULL label image and the
    FULL intensity. The returned dict maps each object label to its
    granularity spectrum of shape ``(spectrum_length,)``, matching
    CellProfiler's ``Granularity_<n>_<image>`` semantics.

    Per-object column names (on the returned object) are
    ``granularity_step{n}_{ch_name}`` for ``n = 1..spectrum_length``.
    """
    logger.debug(
        "Granularity: spectrum_length=%d, subsample=%s, image_sample=%s, "
        "background_radius=%d, ch=%s",
        spectrum_length, subsample_size, image_sample_size, background_radius, ch_name,
    )

    column_names = [f"granularity_step{n}_{ch_name}" for n in range(1, spectrum_length + 1)]

    def _compute(mask, intensity):
        return _granularity_per_object_full(
            mask,
            intensity,
            spectrum_length=spectrum_length,
            subsample_size=subsample_size,
            image_sample_size=image_sample_size,
            background_radius=background_radius,
        )

    return _GranularityComputer(
        compute=_compute,
        column_names=column_names,
        ch_name=ch_name,
        spectrum_length=spectrum_length,
    )


class _GranularityComputer:
    """Wraps the granularity callable with metadata for the caller.

    Attributes
    ----------
    column_names : list[str]
        Column names ``granularity_step{n}_{ch_name}`` for ``n = 1..N``.
    ch_name : str
        Channel name.
    spectrum_length : int
        Number of spectrum bins.
    """

    __slots__ = ("compute", "column_names", "ch_name", "spectrum_length")

    def __init__(self, compute, column_names, ch_name, spectrum_length):
        self.compute = compute
        self.column_names = column_names
        self.ch_name = ch_name
        self.spectrum_length = spectrum_length

    def __call__(self, mask, intensity):
        return self.compute(mask, intensity)


# ═══════════════════════════════════════════════════════════════════════════
# 3. GLCM
# ═══════════════════════════════════════════════════════════════════════════

_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation",
               "asm", "entropy")


def _glcm_all(
    regionmask: np.ndarray,
    intensity: np.ndarray,
    *,
    distances: tuple,
    angles: tuple,
    levels: int,
    props: tuple,
) -> np.ndarray:
    """GLCM features — flat array: [d0_p0, d0_p1, ..., dN_pM].

    Background handling: pixels outside the object mask are quantised to 0
    and the corresponding row/column of the GLCM is dropped, equivalent to
    ``ignore_zeros=True`` on a binary mask.
    """
    if levels > 256:
        logger.warning("GLCM levels=%d exceeds maximum of 256 — clamping to 256", levels)
    levels = min(levels, 256)
    img = intensity.astype(float)
    mask = regionmask.astype(bool)
    roi = img[mask]
    n_out = len(distances) * len(props)
    if roi.size == 0:
        return np.zeros(n_out)

    mn, mx = roi.min(), roi.max()

    # Foreground pixels quantised to [1, levels]; background pixels to 0.
    # Dropping row/col 0 from the GLCM is the GLCM equivalent of
    # mahotas' ignore_zeros=True.
    quantised = np.zeros(img.shape, dtype=np.uint16 if levels >= 256 else np.uint8)
    if mx > mn:
        quantised[mask] = (
            ((img[mask] - mn) / (mx - mn) * (levels - 1)).astype(int) + 1
        )
    else:
        # Constant ROI: assign a single non-zero value so the GLCM has mass
        # on one diagonal cell (contrast=0, homogeneity=1, energy=1, …).
        quantised[mask] = 1
    quantised = np.clip(quantised, 0, levels)

    results = []
    for d in distances:
        glcm_full = graycomatrix(
            quantised, distances=[d], angles=list(angles),
            levels=levels + 1, symmetric=True, normed=False,
        )
        # Drop the background row/col (index 0) — only foreground-fg pairs
        # remain in [1, levels] x [1, levels].
        glcm_masked = glcm_full[1:levels + 1, 1:levels + 1, :, :].astype(float)
        for di in range(glcm_masked.shape[2]):
            for a in range(glcm_masked.shape[3]):
                total = glcm_masked[:, :, di, a].sum()
                if total > 0:
                    glcm_masked[:, :, di, a] /= total

        for p in props:
            if p == "asm":
                vals = graycoprops(glcm_masked, "energy")[0] ** 2
            elif p == "entropy":
                entropies = []
                for a in range(glcm_masked.shape[3]):
                    p_mat = glcm_masked[:, :, 0, a]
                    nonzero = p_mat[p_mat > 0]
                    entropies.append(float(-np.sum(nonzero * np.log2(nonzero))))
                vals = np.array(entropies)
            else:
                vals = graycoprops(glcm_masked, p)[0]
            results.append(float(vals.mean()))
    return np.array(results)


def make_glcm(
    distances: Sequence[int] = (2,),
    angles: Sequence[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 256,
    ch_name: str = "ch0",
    props: Sequence[str] = _GLCM_PROPS,
) -> list:
    """Create GLCM texture callables for regionprops_table."""
    distances = tuple(distances)
    angles = tuple(angles)
    props = tuple(props)
    logger.debug("GLCM: distances=%s, angles=%d, levels=%d, props=%s", distances, len(angles), levels, props)

    def _compute(mask, intensity):
        return _glcm_all(
            mask, intensity, distances=distances, angles=angles,
            levels=levels, props=props,
        )

    fns = []
    for di, d in enumerate(distances):
        for pi, p in enumerate(props):
            idx = di * len(props) + pi
            def _fn(mask, intensity, _idx=idx):
                return float(_compute(mask, intensity)[_idx])
            fns.append(_named(_fn, f"glcm_{p}_d{d}_{ch_name}"))
    return fns


# ═══════════════════════════════════════════════════════════════════════════
# 4. PEARSON CORRELATION  (standalone, not an extra_property)
# ═══════════════════════════════════════════════════════════════════════════

def measure_channel_correlation(
    label_image: np.ndarray,
    multichannel_image: np.ndarray,
    channel_pairs: Sequence[Tuple[int, int]] | None = None,
) -> dict:
    """Pearson correlation between channel pairs, per labeled object."""
    if multichannel_image.ndim != 3:
        raise ValueError("multichannel_image must be (H, W, C)")
    n_ch = multichannel_image.shape[2]

    if channel_pairs is None:
        channel_pairs = [(a, b) for a in range(n_ch) for b in range(a + 1, n_ch)]

    labels = np.unique(label_image)
    labels = labels[labels != 0]
    n_obj = len(labels)
    result: dict = {"label": labels}

    for a, b in channel_pairs:
        ch_a = multichannel_image[..., a].astype(float)
        ch_b = multichannel_image[..., b].astype(float)
        pearson = np.full(n_obj, np.nan)
        for i, lbl in enumerate(labels):
            m = label_image == lbl
            va, vb = ch_a[m], ch_b[m]
            if va.std() > 0 and vb.std() > 0:
                pearson[i] = float(np.corrcoef(va, vb)[0, 1])
        result[f"correlation_pearson_ch{a}_ch{b}"] = pearson

    return result
