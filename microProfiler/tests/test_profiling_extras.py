"""Regression tests for profiling extras: radial, granularity, GLCM.

These guard the CellProfiler-parity behaviour reviewed in
``.trae/documents/object_profile_measurement_review.md``:

* radial: geodesic (8-connected, in-mask) centre-to-pixel distance, per-object
  boundary via regionprops slicing, correct FracAtD/MeanFrac/RadialCV;
* granularity: tophat background reduction + cumulative erosion / geodesic
  reconstruction with ``(prev - curr) * 100 / start`` per object;
* GLCM: per-object min-max quantisation kept, the full 15-column Haralick
  feature set, brightness invariance (same texture -> same features).
"""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from skimage.draw import disk
from skimage.graph import MCP_Geometric
from skimage.measure import regionprops_table

from microProfiler.profiling.extras import (
    _radial_features_one_object,
    make_glcm,
    make_granularity,
)
from microProfiler.profiling.object_profiler import measure_objects


# ── helpers ───────────────────────────────────────────────────────────────


def _disk_mask(shape=(64, 64), radius=20, center=None):
    center = center or (shape[0] // 2, shape[1] // 2)
    mask = np.zeros(shape, dtype=bool)
    rr, cc = disk(center, radius)
    mask[rr, cc] = True
    return mask


def _glcm_map(fns, labels, img):
    props = regionprops_table(
        labels, intensity_image=img, properties=["label"], extra_properties=fns,
    )
    return {fn.__name__: props[fn.__name__][0] for fn in fns}


# ── radial ────────────────────────────────────────────────────────────────


def test_radial_uniform_disk():
    mask = _disk_mask()
    img = np.ones(mask.shape, dtype=float)
    frac, meanfrac, radialcv = _radial_features_one_object(mask, img, nbins=4)
    assert np.isclose(frac.sum(), 1.0)
    # Uniform intensity -> every ring's mean intensity equals the object mean.
    assert np.allclose(meanfrac[frac > 0], 1.0, atol=1e-6)
    # Uniform disk -> all 8 wedges of each ring have equal means -> CV = 0.
    assert np.allclose(radialcv, 0.0, atol=1e-9)


def test_radial_center_strong_gradient():
    mask = _disk_mask(radius=20, center=(32, 32))
    img = np.zeros(mask.shape, dtype=float)
    rr, cc = disk((32, 32), 8)
    img[rr, cc] = 5.0
    img[mask & (img == 0)] = 1.0
    frac, meanfrac, _ = _radial_features_one_object(mask, img, nbins=4)
    assert np.isclose(frac.sum(), 1.0)
    # Innermost ring (smallest norm) carries the highest mean intensity.
    assert meanfrac[0] > 1.0
    assert meanfrac[-1] < 1.0


def test_radial_geodesic_differs_from_euclidean_on_concave():
    mask = np.zeros((96, 96), dtype=bool)
    yy, xx = np.mgrid[0:96, 0:96]
    d = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    mask[(d <= 42) & (d >= 30)] = True
    # Cut a right-side notch so the object is C-shaped.
    angle = np.degrees(np.arctan2(yy - 48, xx - 48))
    mask[(angle > -30) & (angle < 30)] = False
    img = np.ones(mask.shape, dtype=float)

    frac, _, _ = _radial_features_one_object(mask, img, nbins=8)
    assert np.isclose(frac.sum(), 1.0)

    # Replay the two distance definitions to prove they bin differently.
    d_to_edge = distance_transform_edt(mask)
    cy, cx = np.unravel_index(int(np.argmax(d_to_edge)), d_to_edge.shape)
    cost = np.where(mask, 1.0, np.inf)
    d_geo = MCP_Geometric(cost, fully_connected=True).find_costs([(cy, cx)])[0]
    d_euc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    def _bins(dist):
        d_o = dist[mask]
        e_o = d_to_edge[mask]
        return np.clip(np.floor(d_o / (d_o + e_o + 1e-3) * 8).astype(int), 0, 7)

    bin_geo, bin_euc = _bins(d_geo), _bins(d_euc)
    # The geodesic route must go around the notch -> some pixels re-bin.
    assert np.any(bin_geo != bin_euc)


def test_radial_touching_objects_use_own_boundary():
    # Two touching disks: regionprops hands the per-object slice to the
    # extra property, so distance-to-edge is measured against the object's
    # OWN boundary, never the union (which has no gap at the seam).
    labels = np.zeros((48, 64), dtype=int)
    rr1, cc1 = disk((24, 24), 14)
    rr2, cc2 = disk((24, 40), 14)
    labels[rr1, cc1] = 1
    labels[rr2, cc2] = 2
    d1 = distance_transform_edt(labels == 1)
    d_union = distance_transform_edt(labels != 0)
    assert np.any((d1 == 1) & (d_union > 1))


# ── granularity ───────────────────────────────────────────────────────────


def test_granularity_small_blob_on_uniform_field():
    img = np.full((64, 64), 100.0)
    rr, cc = disk((32, 32), 3)
    img[rr, cc] = 160.0
    mask = np.ones((64, 64), dtype=int)
    comp = make_granularity(
        spectrum_length=8, subsample_size=0.5, image_sample_size=0.5,
        background_radius=5, ch_name="ch",
    )
    spec = comp(mask, img)[1]
    assert np.all(spec >= -1e-9)
    assert spec[0] > 0.0
    # (start - 0)/start * 100 telescopes into the sum.
    assert spec.sum() <= 100.0
    # The blob is fully eroded by the later steps -> spectrum ends at 0.
    assert abs(spec[-1]) < 1e-9


def test_granularity_brightness_invariance():
    # Same blob at the same location, two brightness levels: tophat, erosion,
    # dilation, reconstruction and resampling are all homogeneous, so the
    # brightness factor cancels exactly in (prev - curr) * 100 / start.
    labels = np.zeros((128, 128), dtype=int)
    rr, cc = disk((64, 64), 9)
    labels[rr, cc] = 1
    comp = make_granularity(
        spectrum_length=6, subsample_size=0.5, image_sample_size=0.5,
        background_radius=2, ch_name="ch",
    )
    s_a = comp(labels, np.where(labels > 0, 200.0, 0.0))[1]
    s_b = comp(labels, np.where(labels > 0, 100.0, 0.0))[1]
    assert np.allclose(s_a, s_b, atol=1e-9)
    assert np.all(s_a >= -1e-9)
    assert s_a.sum() <= 100.0
    assert abs(s_a[-1]) < 1e-9


# ── GLCM ──────────────────────────────────────────────────────────────────


def test_glcm_constant_object():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    img = np.full((32, 32), 50.0)
    g = _glcm_map(make_glcm(distances=(2,), levels=16, ch_name="ch"), mask.astype(int), img)
    assert g["glcm_contrast_d2_ch"] == 0.0
    assert g["glcm_dissimilarity_d2_ch"] == 0.0
    assert g["glcm_homogeneity_d2_ch"] == 1.0
    assert g["glcm_energy_d2_ch"] == 1.0
    assert g["glcm_asm_d2_ch"] == 1.0
    assert g["glcm_entropy_d2_ch"] == 0.0
    assert g["glcm_variance_d2_ch"] == 0.0
    assert g["glcm_sumentropy_d2_ch"] == 0.0
    assert g["glcm_differenceentropy_d2_ch"] == 0.0
    assert g["glcm_infomeasure2_d2_ch"] == 0.0


def test_glcm_random_texture():
    # Rotationally-symmetric random texture: high contrast, low homogeneity.
    rng = np.random.default_rng(3)
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    img = rng.integers(0, 256, (32, 32)).astype(float)
    g = _glcm_map(make_glcm(distances=(1,), levels=32, ch_name="ch"),
                  mask.astype(int), img)
    assert g["glcm_contrast_d1_ch"] > 100.0
    assert g["glcm_homogeneity_d1_ch"] < 0.2
    assert g["glcm_energy_d1_ch"] < 0.2
    assert g["glcm_sumentropy_d1_ch"] > 2.0
    assert g["glcm_differencevariance_d1_ch"] > 0.0


def test_glcm_brightness_invariance():
    # Identical stripe texture at two different brightness levels -> the
    # per-object min-max quantisation maps them to the same GLCM.
    labels = np.zeros((40, 80), dtype=int)
    img = np.zeros((40, 80))
    yy, xx = np.mgrid[0:8, 0:8]
    block = np.where((yy // 2) % 2 == 0, 0.0, 10.0)
    for cy, cx, base, lbl in ((12, 20, 5.0, 1), (12, 60, 250.0, 2)):
        labels[cy:cy + 8, cx:cx + 8] = lbl
        img[cy:cy + 8, cx:cx + 8] = base + block
    fns = make_glcm(distances=(2,), levels=8, ch_name="ch")
    props = regionprops_table(labels, intensity_image=img, properties=["label"],
                              extra_properties=fns)
    by_lbl = {int(lbl): {fn.__name__: props[fn.__name__][i] for fn in fns}
              for i, lbl in enumerate(props["label"])}
    for fn in fns:
        name = fn.__name__
        assert by_lbl[1][name] == pytest.approx(by_lbl[2][name], rel=1e-6,
                                                nan_ok=True)


def test_glcm_feature_columns_complete():
    fns = make_glcm(distances=(2,), levels=16, ch_name="ch")
    prefixes = {fn.__name__.removeprefix("glcm_").split("_d")[0] for fn in fns}
    expected = {
        "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
        "asm", "entropy", "variance", "sumaverage", "sumvariance", "sumentropy",
        "differencevariance", "differenceentropy", "infomeasure1", "infomeasure2",
    }
    assert prefixes == expected


def test_glcm_finite_on_large_random_object():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64)).astype(float)
    mask = np.zeros((64, 64), dtype=int)
    mask[8:56, 8:56] = 1
    fns = make_glcm(distances=(1, 2), levels=16, ch_name="ch")
    props = regionprops_table(mask, intensity_image=img, properties=["label"],
                              extra_properties=fns)
    for fn in fns:
        assert np.all(np.isfinite(props[fn.__name__]))


# ── end-to-end smoke ──────────────────────────────────────────────────────


def test_measure_objects_all_groups_smoke():
    rng = np.random.default_rng(1)
    mask = np.zeros((64, 64), dtype=int)
    rr1, cc1 = disk((20, 24), 12)
    mask[rr1, cc1] = 1
    rr2, cc2 = disk((44, 40), 10)
    mask[rr2, cc2] = 2
    img = np.stack(
        [rng.normal(50.0, 20.0, (64, 64)).astype(np.float32) for _ in range(3)],
        axis=-1,
    )
    df = measure_objects(
        mask=mask,
        img=img,
        channel_names=["DNA", "AGP", "Mito"],
        intensity_channels=["DNA", "AGP"],
        radial_channels=["DNA"],
        radial_kwargs={"nbins": 4},
        granularity_channels=["DNA"],
        granularity_kwargs={"spectrum_length": 8, "subsample_size": 0.5,
                            "image_sample_size": 0.5, "background_radius": 4},
        glcm_channels=["AGP"],
        glcm_kwargs={"distances": [2], "levels": 16},
        correlation_pairs=[("DNA", "AGP")],
    )
    assert len(df) == 2
    assert "radial_fracatd_bin1of4_DNA" in df.columns
    assert "granularity_step1_DNA" in df.columns
    assert "glcm_contrast_d2_AGP" in df.columns
    assert "glcm_infomeasure1_d2_AGP" in df.columns
    assert "glcm_differencevariance_d2_AGP" in df.columns
    assert "correlation_pearson_DNA_AGP" in df.columns
    assert df["radial_fracatd_bin1of4_DNA"].notna().all()


def test_failing_channel_fallback_uses_feature_names():
    """A channel whose feature group fails must fill NaN under the REAL
    column names (skimage names extra properties from func.__name__) —
    bogus fallback names would poison BatchWriter's first-flush schema and
    silently drop the real features for the whole table."""
    from microProfiler.profiling.object_profiler import (
        _run_per_channel_regionprops,
    )

    def boom(label, image):
        raise RuntimeError("boom")

    boom.__name__ = "intensity_mean_ch1"
    mask = np.zeros((32, 32), dtype=int)
    mask[8:24, 8:24] = 1
    img = np.zeros((32, 32, 1), dtype=float)

    df = _run_per_channel_regionprops(mask, img, ["ch1"], {0: [boom]})
    assert "intensity_mean_ch1" in df.columns
    assert df["intensity_mean_ch1"].isna().all()