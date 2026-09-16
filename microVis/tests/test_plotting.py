"""Tests for microVis.processing.plotting: facet/cap helpers + figures + PDF."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from microVis.processing import plotting as P  # noqa: E402


def _df(n=24):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "well": [f"A{i % 4}" for i in range(n)],
        "field": [str(i % 3) for i in range(n)],
        # Cycle 3 against the 4-well cycle so all well × condition
        # combinations are populated.
        "condition": ["ctrl" if i % 3 else "drug" for i in range(n)],
        "value": rng.normal(size=n),
        "area": rng.uniform(10, 100, size=n),
    })


def test_column_classification():
    df = _df()
    assert P.is_continuous(df["value"])
    assert not P.is_continuous(df["well"])
    assert "value" in P.numeric_columns(df)
    assert "well" in P.categorical_columns(df)


def test_facet_groups_full_combination():
    df = _df()
    groups, truncated = P.facet_groups(df, ["well", "condition"])
    assert not truncated
    # 4 wells × 2 conditions = 8 combinations (all non-empty here).
    assert len(groups) == 8
    labels = [label for label, _ in groups]
    assert "well=A0, condition=ctrl" in labels
    total = sum(len(sub) for _label, sub in groups)
    assert total == len(df)


def test_facet_groups_truncation():
    df = _df()
    groups, truncated = P.facet_groups(df, ["well", "condition"], max_combos=3)
    assert truncated is True
    assert len(groups) == 3


def test_point_cap():
    df = _df(100)
    sub, sampled, total = P.apply_point_cap(df, 20)
    assert sampled and total == 100 and len(sub) == 20
    sub2, sampled2, _ = P.apply_point_cap(df, 0)
    assert not sampled2 and len(sub2) == 100


def test_apply_filter():
    df = _df()
    out, err = P.apply_filter(df, "area > 50 & value > 0")
    assert err is None
    assert (out["area"] > 50).all() and (out["value"] > 0).all()
    out2, err2 = P.apply_filter(df, "area >")
    assert out2 is None and err2
    same, err3 = P.apply_filter(df, "  ")
    assert same is df and err3 is None


def test_hover_exclude_omits_columns():
    df = _df()
    fig = P.make_scatter(df, x="value", y="area", color="condition",
                         hover_exclude=("value", "area"))
    cols = fig.axes[0].collections[0]._microvis_cols
    assert "value" not in cols and "area" not in cols
    assert "condition" in cols


def test_figures_build():
    df = _df()
    scatter = P.make_scatter(df, x="value", y="area", color="condition",
                             size="area", facet_cols=["well"], ncols=2)
    line = P.make_line(df, y="value", x="well", color="condition",
                       facet_cols=["field"], ncols=2)
    box = P.make_boxplot(df, y="value", x="condition", color="condition")
    bar = P.make_barplot_mean_sem(df, y="value", x="well", color="condition")
    for fig in (scatter, line, box, bar):
        assert fig is not None
        assert len(fig.axes) >= 1
        # The Data page shows no plot title — the axes labels carry the info.
        assert fig.get_suptitle() == ""


def test_line_groups_and_sem_points():
    df = _df()
    fig = P.make_line(df, y="value", x="well", color="condition",
                      show_points=True)
    ax = fig.axes[0]
    # Raw-points scatter + one errorbar per color group.
    assert len(ax.collections) >= 2
    # X ticks are the 4 well levels (color lives in the legend, NOT on the
    # x axis) — one smooth curve per color group spans those levels.
    ticks = {t.get_text() for t in ax.get_xticklabels()}
    assert ticks == {"A0", "A1", "A2", "A3"}
    # One smooth fit line per color group (2 conditions), drawn as a DENSE
    # spline (many more x samples than the 4 wells); errorbar marker lines
    # carry linestyle 'None'.
    connect_lines = [l for l in ax.lines if l.get_linestyle() == "-"]
    assert len(connect_lines) == 2
    for line in connect_lines:
        assert len(line.get_xdata()) > 4
        assert len(line.get_xdata()) == len(line.get_ydata())


def test_line_smooth_curve_defaults():
    """_smooth_curve: spline for 4+ points, polyline below that; no raises."""
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 0.5, 2.0]
    gx, gy = P._smooth_curve(xs, ys)
    assert len(gx) == len(gy) > 4          # dense spline samples
    assert gx[0] == 0.0 and gx[-1] == 3.0
    # Fewer than 4 points: the raw polyline passes through unchanged.
    assert P._smooth_curve(xs[:3], ys[:3])[0].tolist() == xs[:3]
    # Degenerate input degrades instead of raising.
    assert len(P._smooth_curve([1.0, 1.0, 1.0, 1.0], ys)[0]) >= 1


def test_make_line_requires_x_semantics():
    df = _df()
    # A single x level still renders (one node, no connecting line).
    single = df[df["well"] == "A0"]
    fig = P.make_line(single, y="value", x="well")
    assert fig.axes[0].get_xticks().size == 1


def test_pdf_export_editable_text(tmp_path):
    df = _df()
    fig = P.make_scatter(df, x="value", y="area", color="condition")
    out = tmp_path / "plot.pdf"
    P.save_figure_pdf(fig, out)
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    # Type-42 (TrueType) fonts keep text editable in vector PDFs.
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_hover_shows_point_values():
    from matplotlib.backend_bases import MouseEvent

    df = _df()
    fig = P.make_scatter(df, x="value", y="area", color="condition")
    P.attach_hover(fig)
    fig.canvas.draw()
    ax = fig.axes[0]
    coll = ax.collections[0]
    x_disp, y_disp = ax.transData.transform(np.asarray(coll.get_offsets())[0])
    event = MouseEvent("motion_notify_event", fig.canvas, x_disp, y_disp)
    fig.canvas.callbacks.process("motion_notify_event", event)
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    assert any("value:" in t and "area:" in t for t in texts), texts


def test_scatter_collections_carry_hover_rows():
    df = _df()
    fig = P.make_scatter(df, x="value", y="area", size="area")
    coll = fig.axes[0].collections[0]
    assert len(coll._microvis_rows) == len(df)
    assert "value" in coll._microvis_cols and "area" in coll._microvis_cols


def test_axis_values_categorical_codes():
    df = _df()
    vals, ticks = P.axis_values(df["condition"])
    assert ticks == ["ctrl", "drug"]
    assert set(vals.dropna().unique()) == {0.0, 1.0}
    nums, no_ticks = P.axis_values(df["value"])
    assert no_ticks is None and nums.notna().all()


def test_categorical_x_and_y_are_plottable():
    df = _df()
    # Meta variables on both axes: codes + level-name ticks.
    scatter = P.make_scatter(df, x="condition", y="well")
    ax = scatter.axes[0]
    assert ax.get_xticks().size == 2
    assert {t.get_text() for t in ax.get_xticklabels()} == {"ctrl", "drug"}
    assert ax.get_yticks().size == 4  # A0..A3
    # Categorical Y works for box/bar too.
    box = P.make_boxplot(df, y="condition", x="well")
    assert box.axes[0].get_yticks().size == 2
    bar = P.make_barplot_mean_sem(df, y="condition", x="well")
    assert bar.axes[0].get_yticks().size == 2


def test_legends_sit_outside_the_axes():
    """Every figure legend lives in the right strip, never on the data."""
    df = _df()
    figs = [
        P.make_scatter(df, x="value", y="area", color="condition", size="area"),
        P.make_scatter(df, x="value", y="area", color="condition",
                       facet_cols=["well"], ncols=2),
        P.make_boxplot(df, y="value", x="condition", color="condition"),
        P.make_barplot_mean_sem(df, y="value", x="well", color="condition"),
        P.make_line(df, y="value", x="well", color="condition"),
    ]
    for fig in figs:
        fig.canvas.draw()
        width = fig.bbox.width
        axes_right = max(ax.get_window_extent().x1 for ax in fig.axes) / width
        assert fig.legends, "expected at least one figure legend"
        for leg in fig.legends:
            assert leg.get_window_extent().x0 / width >= axes_right - 0.005, (
                "a legend overlaps the plot area")


def test_facets_share_axes_scatter():
    """Facet panels share limits: wildly different ranges still align."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 300.0, 400.0],
        "y": [10.0, 20.0, 3000.0, 4000.0],
        "g": ["p", "p", "q", "q"],
    })
    fig = P.make_scatter(df, x="x", y="y", facet_cols=["g"])
    fig.canvas.draw()  # materialize the shared autoscale limits
    ax1, ax2 = fig.axes[0], fig.axes[1]
    assert ax1.get_xlim() == ax2.get_xlim()
    assert ax1.get_ylim() == ax2.get_ylim()


def test_facet_categorical_positions_are_global():
    """A category missing from one facet still occupies its global slot."""
    df = pd.DataFrame({
        "cat": ["b", "a", "a"],          # facet p only has "b"
        "y": [1.0, 2.0, 3.0],
        "g": ["p", "q", "q"],
    })
    fig = P.make_scatter(df, x="cat", y="y", facet_cols=["g"])
    fig.canvas.draw()
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["a", "b"]
    # Facet p's single point must sit at "b"'s global index 1 (not 0).
    pts = fig.axes[0].collections[0].get_offsets()
    assert pts[0][0] == 1.0


def test_facet_boxplot_positions_are_global():
    """Boxes land on the same x slots in every facet panel."""
    df = pd.DataFrame({
        "x": ["g1", "g2", "g2", "g2"],
        "y": [1.0, 2.0, 2.5, 3.0],
        "g": ["p", "q", "q", "q"],
    })
    fig = P.make_boxplot(df, y="y", x="x", facet_cols=["g"])
    fig.canvas.draw()
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == 2
    for ax in panels:
        assert [t.get_text() for t in ax.get_xticklabels()] == ["g1", "g2"]
        assert ax.get_ylim() == panels[0].get_ylim()
    # Panel p has only g1 — its box must sit at the global slot 1 (not 0).
    centers = []
    for patch in panels[0].patches:
        xs = patch.get_path().vertices[:, 0]
        centers.append((xs.min() + xs.max()) / 2)
    assert np.allclose(centers, 1.0)


def test_facet_line_positions_are_global():
    """Line nodes land on the same x slots in every facet panel."""
    df = pd.DataFrame({
        "x": ["s1", "s2", "s2", "s2"],
        "y": [1.0, 2.0, 2.5, 3.0],
        "g": ["p", "q", "q", "q"],
    })
    fig = P.make_line(df, y="y", x="x", facet_cols=["g"])
    fig.canvas.draw()
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == 2
    for ax in panels:
        assert [t.get_text() for t in ax.get_xticklabels()] == ["s1", "s2"]
    # Facet p's single node sits at s1's global slot 0, facet q's at 1 —
    # read the errorbar markers (the dots behind carry jitter).
    for panel, slot in ((panels[0], 0.0), (panels[1], 1.0)):
        marker_lines = [l for l in panel.lines if l.get_marker() == "o"]
        assert marker_lines
        assert np.allclose(np.concatenate(
            [l.get_xdata() for l in marker_lines]), slot)


def test_facet_panels_show_axis_numbers():
    """Every facet panel shows its own tick numbers despite sharing axes."""
    df = _df()
    fig = P.make_scatter(df, x="value", y="area", facet_cols=["field"])
    for ax in fig.axes:
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert all(t.get_visible() for t in ax.get_yticklabels())


def test_line_continuous_x_is_numeric_axis():
    """Continuous X in line mode: numeric axis + smooth fit of raw y on x
    (no per-value grouping, no category tick labels)."""
    grid = np.linspace(0.0, 10.0, 200)
    df = pd.DataFrame({"x": grid, "y": np.sin(grid), "g": ["p", "q"] * 100})
    fig = P.make_line(df, y="y", x="x")
    ax = fig.axes[0]
    # A real numeric axis: few auto ticks spanning the data range (the
    # locator pads one step outward), NOT one category per distinct value.
    ticks = ax.get_xticks()
    assert ticks.min() <= 0.0 and ticks.max() >= 10.0 and len(ticks) < 12
    assert ax.get_xlabel() == "x"
    # One dense smooth fit line; no errorbar nodes on the continuous axis.
    lines = [l for l in ax.lines if l.get_linestyle() == "-"]
    assert len(lines) == 1 and len(lines[0].get_xdata()) > 4
    # Categorical color splits into one fit per level over the same axis.
    fig2 = P.make_line(df, y="y", x="x", color="g")
    lines2 = [l for l in fig2.axes[0].lines if l.get_linestyle() == "-"]
    assert len(lines2) == 2
