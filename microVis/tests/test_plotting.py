"""Tests for microVis.processing.plotting: facet/cap helpers + figures + PDF."""

import re

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
    groups = P.facet_groups(df, ["well", "condition"])
    # 4 wells × 2 conditions = 8 combinations (all non-empty here), the
    # UNLIMITED cartesian product — no panel cap.
    assert len(groups) == 8
    labels = [label for label, _ in groups]
    assert "well=A0, condition=ctrl" in labels
    total_rows = sum(len(sub) for _label, sub in groups)
    assert total_rows == len(df)


def test_bubble_area_scales_with_point_scale():
    """With Size by, Point scale must zoom ALL bubbles together: the mapped
    area stays proportional (low limit = base*scale, high limit =
    base*scale*_SIZE_SPAN), and the default scale 1 reproduces the
    historical 20..220 pt² anchors."""
    assert (P._bubble_area(1.0, 0.0), P._bubble_area(1.0, 1.0)) == (20.0, 220.0)
    # Doubling Point scale doubles every bubble's area, low and high alike.
    assert (P._bubble_area(2.0, 0.0), P._bubble_area(2.0, 1.0)) == (40.0, 440.0)
    fracs = np.array([0.0, 0.5, 1.0])
    out = P._bubble_area(3.0, fracs)
    assert out[1] == (out[0] + out[2]) / 2  # linear in between


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


def test_pdf_export_exact_page_size(tmp_path):
    """save_figure_pdf(size=...) exports an exact W×H page: the MediaBox is
    w*72 × h*72 points (bbox_inches='tight' would override the size), and
    the live figure size is restored afterwards."""
    fig = P.make_scatter(_df(), x="value", y="area", color="condition")
    natural = fig.get_size_inches().tolist()

    out = tmp_path / "sized.pdf"
    P.save_figure_pdf(fig, out, size=(4.0, 3.0))
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    boxes = re.findall(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]",
        data)
    assert boxes, "PDF page has no MediaBox"
    _, _, w, h = (float(v) for v in boxes[0])
    assert (w, h) == (288, 216)  # 4 in / 3 in at 72 points per inch

    # The resize/restore must leave the on-screen figure untouched.
    assert fig.get_size_inches().tolist() == natural


def _assert_no_layout_overlap(fig):
    """Every panel title fits inside the figure and no title overlaps the
    panel above it (1 px tolerance)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = fig.axes
    height = fig.bbox.height
    # Top row (title included) stays inside the figure.
    assert axes[0].get_tightbbox(renderer).y1 <= height + 0.5
    pos = [ax.get_position() for ax in axes]
    for i, ax in enumerate(axes):
        above = None  # same column, directly above
        for j, other in enumerate(axes):
            if (j != i and abs(pos[j].x0 - pos[i].x0) < 1e-6
                    and pos[j].y0 > pos[i].y0):
                if above is None or pos[j].y0 < pos[above].y0:
                    above = j
        if above is None:
            continue
        above_bottom = axes[above].get_tightbbox(renderer).y0
        title_top = ax.title.get_window_extent(renderer).y1
        assert title_top <= above_bottom + 1.0


def test_facet_layout_clean_at_any_figure_size():
    """The view / exact-size export resize the figure AFTER the build; the
    layout must re-flow (constrained) instead of squeezing the build-time
    gaps until titles overlap the row above's tick labels and the top row
    clips — the reported Data-page bug."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "cmpd": rng.choice(["DMSO", "drug"], n),
        "cell": rng.choice(["a", "b", "c", "d", "e", "f"], n),
        "area": rng.uniform(200, 900, n),
        "score": rng.random(n),
    })
    fig = P.make_scatter(df, x="area", y="score",
                         facet_cols=["cmpd", "cell"], ncols=2)
    _assert_no_layout_overlap(fig)      # natural build size (9.2 × 20.4)
    fig.set_size_inches(8, 6)           # what the interactive view does
    _assert_no_layout_overlap(fig)      # layout must re-flow at this size


def test_hover_shows_point_values():
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    df = _df()
    fig = P.make_scatter(df, x="value", y="area", color="condition")
    # A real renderer is required: constrained layout only executes on a
    # real draw (the renderer-less default canvas skips it silently).
    canvas = FigureCanvasAgg(fig)
    P.attach_hover(fig)
    canvas.draw()
    ax = fig.axes[0]
    coll = ax.collections[0]
    x_disp, y_disp = ax.transData.transform(np.asarray(coll.get_offsets())[0])
    event = MouseEvent("motion_notify_event", canvas, x_disp, y_disp)
    canvas.callbacks.process("motion_notify_event", event)
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    assert any("value:" in t and "area:" in t for t in texts), texts


def test_hover_annotation_does_not_reflow_layout():
    """The hover tooltip is a pure overlay: appearing/disappearing must not
    re-flow the already-drawn figure (constrained layout used to squeeze
    the axes every time the annotation's box hung outside the plot)."""
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    df = _df()
    fig = P.make_scatter(df, x="value", y="area", color="condition")
    canvas = FigureCanvasAgg(fig)
    P.attach_hover(fig)
    ax = fig.axes[0]
    # Warm up the layout: the constrained solver settles after its first
    # execution (sub-pixel drift from the default starting positions).
    canvas.draw()
    canvas.draw()
    before = list(ax.get_position().bounds)

    # Hover the point nearest the top-right corner so the tooltip box hangs
    # PAST the axes edge — the case that used to squeeze the layout.
    coll = ax.collections[0]
    disp = ax.transData.transform(np.asarray(coll.get_offsets()))
    tx, ty = ax.transAxes.transform((1.0, 1.0))
    i = int(np.argmin((disp[:, 0] - tx) ** 2 + (disp[:, 1] - ty) ** 2))
    event = MouseEvent("motion_notify_event", canvas, disp[i, 0], disp[i, 1])
    canvas.callbacks.process("motion_notify_event", event)
    assert ax.texts, "hover annotation missing"
    ann_bb = ax.texts[0].get_window_extent(canvas.get_renderer())
    assert ann_bb.x1 > ax.get_window_extent().x1, "tooltip stays inside; "
    "test scenario no longer exercises the outside-the-axes case"
    canvas.draw()

    after = list(ax.get_position().bounds)
    # Float noise in the solver is ~1e-9; any real re-flow (the old bug
    # shifted the axes width by ~1.7e-2) is orders of magnitude larger.
    assert all(abs(a - b) <= 1e-7 for a, b in zip(after, before))


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
    """Legends live in the constrained-layout margins: never on the data,
    and never stacked on each other (distinct corners)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

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
        # Real renderer so constrained layout actually places the artists.
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        assert fig.legends, "expected at least one figure legend"
        leg_bboxes = [leg.get_window_extent(renderer) for leg in fig.legends]
        for i, legbb in enumerate(leg_bboxes):
            for ax in fig.axes:
                assert not legbb.overlaps(ax.get_window_extent(renderer)), (
                    f"legend {i} overlaps the plot area")
        for j in range(i + 1, len(leg_bboxes)):
            assert not legbb.overlaps(leg_bboxes[j]), (
                "two legends drawn on top of each other")


def test_facets_share_axes_scatter():
    """Shared mode (Free X/Y unchecked): wildly different ranges align."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 300.0, 400.0],
        "y": [10.0, 20.0, 3000.0, 4000.0],
        "g": ["p", "p", "q", "q"],
    })
    fig = P.make_scatter(df, x="x", y="y", facet_cols=["g"],
                         free_x=False, free_y=False)
    fig.canvas.draw()  # materialize the shared autoscale limits
    ax1, ax2 = fig.axes[0], fig.axes[1]
    assert ax1.get_xlim() == ax2.get_xlim()
    assert ax1.get_ylim() == ax2.get_ylim()


def test_facets_free_axes_scatter():
    """Free X/Y (the default): every panel scales to its own data."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 300.0, 400.0],
        "y": [10.0, 20.0, 3000.0, 4000.0],
        "g": ["p", "p", "q", "q"],
    })
    fig = P.make_scatter(df, x="x", y="y", facet_cols=["g"])
    fig.canvas.draw()
    ax1, ax2 = fig.axes[0], fig.axes[1]
    assert ax1.get_xlim() != ax2.get_xlim()
    assert ax1.get_ylim() != ax2.get_ylim()


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
    fig = P.make_boxplot(df, y="y", x="x", facet_cols=["g"], free_y=False)
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
