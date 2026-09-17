"""Plotting helpers for the microVis Data-page DB tabs.

Everything here is pure Qt-free matplotlib: the widgets load a DataFrame,
call a ``make_*`` function, and embed the returned Figure. Vector PDF export
keeps text editable by embedding Type-42 fonts (``pdf.fonttype = 42``).

Facet semantics: the selected facet variables form the full cartesian product
of their levels (one subplot per combination). Point caps apply to scatter
plots only; box/bar/line always use every row.
"""

from __future__ import annotations

import itertools
import math

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from natsort import natsort_key

# Editable text in exported vector PDFs (Type-42 TrueType embedding).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SINGLE_COLOR = "#4a90d9"


# ── Column classification ─────────────────────────────────────────────────


def is_continuous(series: pd.Series) -> bool:
    """True when a column maps to a continuous (colormap) scale."""
    return (pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_continuous(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not is_continuous(df[c])]


def _sorted_values(series: pd.Series) -> list:
    vals = [v for v in series.dropna().unique().tolist()]
    return sorted(vals, key=lambda v: natsort_key(str(v)))


def axis_values(series: pd.Series):
    """Return ``(numeric_values, tick_labels|None)`` for any variable.

    Continuous columns pass through as floats; categorical columns (meta text)
    are mapped to their natsorted level index so they can still be placed on an
    axis, with the level names returned for the axis ticks.
    """
    if is_continuous(series):
        return pd.to_numeric(series, errors="coerce"), None
    levels = _sorted_values(series)
    mapping = {v: float(i) for i, v in enumerate(levels)}
    return series.map(mapping), levels


def _apply_categorical_axis(ax, xticks, yticks) -> None:
    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, fontsize=7, rotation=30, ha="right")
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=7)


# ── Facets / sampling ─────────────────────────────────────────────────────


def facet_groups(df: pd.DataFrame, facet_cols, max_combos: int = 24):
    """Split *df* into one sub-frame per full combination of facet columns.

    Returns ``(groups, truncated)``: groups is a list of ``(label, sub_df)``;
    combinations beyond ``max_combos`` are dropped and ``truncated`` is True
    so the caller can warn. Empty combinations are skipped.
    """
    facet_cols = [c for c in (facet_cols or []) if c in df.columns]
    if not facet_cols:
        return [("", df)], False
    value_lists = [_sorted_values(df[c]) for c in facet_cols]
    combos = list(itertools.product(*value_lists))
    truncated = len(combos) > max_combos
    groups = []
    for combo in combos[:max_combos]:
        mask = pd.Series(True, index=df.index)
        for col, val in zip(facet_cols, combo):
            mask &= df[col] == val
        sub = df[mask]
        if sub.empty:
            continue
        label = ", ".join(f"{col}={val}" for col, val in zip(facet_cols, combo))
        groups.append((label, sub))
    if not groups:
        groups = [("", df)]
    return groups, truncated


def apply_point_cap(df: pd.DataFrame, cap: int, seed: int = 0):
    """Deterministic scatter row cap. ``cap <= 0`` means no cap.

    Returns ``(capped_df, sampled, total)``.
    """
    total = len(df)
    if cap and total > cap:
        return df.sample(n=int(cap), random_state=seed), True, total
    return df, False, total


def apply_filter(df: pd.DataFrame, expr: str):
    """Apply a pandas query expression to a plot DataFrame.

    Returns ``(filtered_df, None)`` on success or ``(None, error_message)``
    when the expression does not parse/evaluate — callers surface the error
    and abort the plot instead of raising.
    """
    expr = (expr or "").strip()
    if not expr:
        return df, None
    try:
        return df.query(expr, engine="python"), None
    except Exception as e:
        return None, str(e)


# ── Internals ─────────────────────────────────────────────────────────────


def _new_grid(n_panels: int, ncols: int):
    """Create a facet grid sized to n_panels panels. Returns (fig, axes).

    All panels SHARE both axes (facet panels exist to be compared, so
    limits and autoscale stay identical across the grid). The callers
    likewise build every category->position mapping from the FULL table,
    not per panel. The figure is built WITHOUT pyplot (no figure manager)
    so rendering can run on a worker thread; the Qt widget attaches the
    canvas later on the GUI thread.
    """
    ncols_eff = max(1, min(ncols, n_panels))
    nrows = int(math.ceil(n_panels / ncols_eff))
    fig = Figure(figsize=(4.6 * ncols_eff, 3.4 * nrows))
    axes = fig.subplots(nrows, ncols_eff, squeeze=False, sharex=True,
                        sharey=True)
    axes = list(axes.ravel())
    # Sharing hides the inner panels' tick labels by default — facet panels
    # exist to be compared, so EVERY panel shows its own axis numbers.
    for ax in axes:
        ax.tick_params(labelleft=True, labelbottom=True)
    return fig, axes


def _palette_map(values, palette: str) -> dict:
    """Stable value -> RGBA mapping for a categorical column."""
    uniq = _sorted_values(pd.Series(values))
    cmap = matplotlib.colormaps[palette]
    # Sequential/qualitative colormaps both work; cycle if there are more
    # categories than entries (keeps the plot drawable, warns via gray tones).
    return {v: cmap((i % max(1, cmap.N - 1)) / max(1, cmap.N - 1))
            for i, v in enumerate(uniq)}


def _group_series(df: pd.DataFrame, y: str, keys):
    """Split *df* into ``(key_tuple, y_series)`` groups in natsorted order."""
    if not keys:
        return [((), df[y].dropna())]
    out = []
    for key, group in df.groupby(keys, dropna=True):
        if not isinstance(key, tuple):
            key = (key,)
        out.append((key, group[y].dropna()))
    out.sort(key=lambda kv: tuple(natsort_key(str(v)) for v in kv[0]))
    return out


def _group_label(key) -> str:
    return " | ".join(str(v) for v in key)


def _smooth_curve(xs, ys):
    """Smooth fit through (xs, ys) with scipy's default settings.

    A smoothing spline (UnivariateSpline — automatic smoothing factor) for
    enough points, a cubic least-squares fallback, and finally the raw
    polyline. Never raises. The fit is order-invariant: x is sorted first
    (fitpack requires increasing x — raw rows arrive in arbitrary order,
    and plotting them unsorted draws a zigzag mess instead of one line).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 4:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    gx = np.linspace(float(xs.min()), float(xs.max()), 200)
    try:
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(xs, ys)
        return gx, spline(gx)
    except Exception:
        try:
            coeffs = np.polyfit(xs, ys, 3)
            return gx, np.polyval(coeffs, gx)
        except Exception:
            return xs, ys


def _axis_style(ax, title: str = ""):
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)


# Fraction of the figure width reserved on the right edge for legends, so
# they never overlap the plotted data.
_LEGEND_STRIP = 0.11


def _finish_grid(fig, axes, used, facet_cols, truncated,
                 legends=(), colorbar=None):
    """Hide unused panels and place every legend on the RIGHT edge.

    All figure legends live in a reserved strip along the figure's right
    side (stacked top/bottom), so they never sit on top of the data points.
    A continuous-color colorbar keeps its usual place just right of the
    axes, before the legend strip.
    """
    for ax in axes[used:]:
        ax.set_visible(False)
    fig.tight_layout()
    right = 1.0 - _LEGEND_STRIP if (legends or colorbar is not None) else 1.0
    if right < 1.0:
        fig.subplots_adjust(right=right)
    # Colorbar first: it needs the final axes bboxes; legends go right of it.
    legend_left = right + 0.015
    if colorbar is not None:
        cbar = fig.colorbar(colorbar["mappable"], ax=axes[:used], shrink=0.8,
                            label=colorbar.get("label") or None,
                            fraction=0.045, pad=0.015)
        legend_left = max(legend_left, cbar.ax.get_position().x1 + 0.01)
    for i, spec in enumerate(legends):
        if i == 0:
            loc, y = "upper left", 0.99
        else:
            loc, y = "lower left", 0.01
        fig.legend(handles=spec["handles"], title=spec.get("title") or None,
                   loc=loc, bbox_to_anchor=(legend_left, y),
                   fontsize=7, title_fontsize=7, frameon=False)
    if truncated:
        fig.text(right - 0.01, 0.01, "facet combinations truncated",
                 ha="right", va="bottom", fontsize=7, color="#b05050")


# ── Scatter ───────────────────────────────────────────────────────────────

# Identifier columns always shown in a hover tooltip when present, on top of
# the currently mapped x/y/color/size variables.
_HOVER_COLS = ("well", "label", "field", "stack", "timepoint")
_HOVER_RADIUS_PX = 12.0


def _hover_cols(kept: pd.DataFrame, x, y, color, size, exclude=()) -> list[str]:
    cols: list[str] = []
    for c in (x, y, color, size, *_HOVER_COLS):
        if c and c in kept.columns and c not in cols and c not in exclude:
            cols.append(c)
    return cols


def attach_hover(fig, canvas=None) -> None:
    """Enable per-point hover tooltips on scatter collections.

    ``make_scatter`` tags every scatter with the rows it drew; this connects a
    motion handler that shows the nearest point's values in an annotation box.
    Call it AFTER the figure is embedded in its real canvas (the handler binds
    to that canvas). Safe to call once per figure — repeat calls are ignored.
    """
    canvas = canvas if canvas is not None else fig.canvas
    if canvas is None or getattr(fig, "_microvis_hover_cid", None) is not None:
        return

    state = {"annotation": None, "key": None}

    def _clear():
        ann = state["annotation"]
        if ann is not None:
            ann.remove()
            state["annotation"] = None
        state["key"] = None

    def _on_motion(event):
        if event.inaxes is None or event.x is None or event.y is None:
            if state["annotation"] is not None:
                _clear()
                event.canvas.draw_idle()
            return
        best = None  # (dist2, collection, point_index)
        for coll in event.inaxes.collections:
            rows = getattr(coll, "_microvis_rows", None)
            if rows is None or len(rows) == 0:
                continue
            offsets = np.asarray(coll.get_offsets(), dtype=float)
            if offsets.size == 0:
                continue
            disp = event.inaxes.transData.transform(offsets)
            dx = disp[:, 0] - event.x
            dy = disp[:, 1] - event.y
            d2 = dx * dx + dy * dy
            i = int(np.nanargmin(d2))
            if best is None or d2[i] < best[0]:
                best = (d2[i], coll, i)
        if best is None or best[0] > _HOVER_RADIUS_PX ** 2:
            if state["annotation"] is not None:
                _clear()
                event.canvas.draw_idle()
            return
        d2, coll, i = best
        rows = coll._microvis_rows
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        row = rows.iloc[i]
        lines = []
        for c in coll._microvis_cols:
            try:
                val = row[c]
            except KeyError:
                continue
            if isinstance(val, (float, np.floating)):
                text = f"{val:.4g}"
            else:
                text = str(val)
            lines.append(f"{c}: {text}")
        text = "\n".join(lines)
        key = (id(coll), i, id(event.inaxes))
        if key == state["key"]:
            return
        state["key"] = key
        xy = (float(offsets[i, 0]), float(offsets[i, 1]))
        ann = state["annotation"]
        if ann is not None and ann.axes is event.inaxes:
            ann.xy = xy
            ann.set_text(text)
        else:
            _clear()
            ann = event.inaxes.annotate(
                text, xy=xy, xytext=(14, 14), textcoords="offset points",
                fontsize=8, zorder=20,
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0",
                          ec="#888888", alpha=0.95),
            )
            state["annotation"] = ann
            state["key"] = key
        event.canvas.draw_idle()

    fig._microvis_hover_cid = canvas.mpl_connect("motion_notify_event", _on_motion)


def attach_click(fig, canvas=None, callback=None, radius=_HOVER_RADIUS_PX):
    """Left-click picking on tagged scatter collections.

    On a left click, the NEAREST tagged point within `radius` display pixels
    wins (overlapping points included) and ``callback(row, x, y)`` fires with
    that point's DataFrame row and the widget-local click position. A click
    on empty space fires ``callback(None, x, y)`` so the caller can hide any
    popup it showed. Returns the mpl connection id (dies with the canvas).
    """
    if callback is None:
        return None
    canvas = canvas if canvas is not None else fig.canvas

    def _on_click(event):
        if (event.x is None or event.y is None
                or event.button != 1 or event.dblclick):
            return
        best = None  # (dist2, rows, point_index)
        if event.inaxes is not None:
            for coll in event.inaxes.collections:
                rows = getattr(coll, "_microvis_rows", None)
                if rows is None or len(rows) == 0:
                    continue
                offsets = np.asarray(coll.get_offsets(), dtype=float)
                if offsets.size == 0:
                    continue
                disp = event.inaxes.transData.transform(offsets)
                d2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
                i = int(np.argmin(d2))
                if best is None or d2[i] < best[0]:
                    best = (d2[i], rows, i)
        if best is None or best[0] > radius ** 2:
            callback(None, event.x, event.y)
        else:
            _, rows, i = best
            callback(rows.iloc[i], event.x, event.y)

    return canvas.mpl_connect("button_press_event", _on_click)


def make_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    facet_cols=(),
    palette: str = "Set1",
    cmap: str = "viridis",
    point_size: float = 20.0,
    max_size: float = 220.0,
    ncols: int = 3,
    max_combos: int = 24,
    hover_exclude=(),
):
    """Faceted scatter: color and size may each map one variable.

    Every drawn collection is tagged with its rows so ``attach_hover`` can
    show a point's values on mouse-over.
    """
    groups, truncated = facet_groups(df, facet_cols, max_combos)
    fig, axes = _new_grid(len(groups), ncols)

    color_map = None
    norm = None
    cmap_obj = matplotlib.colormaps[cmap]
    if color and color in df.columns:
        if is_continuous(df[color]):
            vals = pd.to_numeric(df[color], errors="coerce")
            norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        else:
            color_map = _palette_map(df[color], palette)
    if size and size in df.columns:
        s_all = pd.to_numeric(df[size], errors="coerce")
        s_min, s_max = np.nanmin(s_all), np.nanmax(s_all)
    else:
        s_min = s_max = None

    size_handles = []
    # Categorical levels come from the FULL table so every facet panel maps
    # a category to the SAME position — the grid shares its axes, and
    # per-panel mappings would scatter identical categories differently.
    def _axis_map(col):
        if not is_continuous(df[col]):
            levels = _sorted_values(df[col])
            return {v: float(i) for i, v in enumerate(levels)}, levels
        return None, None

    xmap, xlevels = _axis_map(x)
    ymap, ylevels = _axis_map(y)

    for ax, (label, sub) in zip(axes, groups):
        if xmap is not None:
            xs, xticks = sub[x].map(xmap), xlevels
        else:
            xs, xticks = pd.to_numeric(sub[x], errors="coerce"), None
        if ymap is not None:
            ys, yticks = sub[y].map(ymap), ylevels
        else:
            ys, yticks = pd.to_numeric(sub[y], errors="coerce"), None
        keep = xs.notna() & ys.notna()
        xs, ys = xs[keep], ys[keep]
        kept = sub[keep].reset_index(drop=True)
        # Point sizes
        if s_min is None:
            sizes = np.full(len(xs), point_size)
        else:
            svals = pd.to_numeric(sub[size], errors="coerce")[keep].to_numpy(dtype=float)
            if s_max > s_min:
                frac = (svals - s_min) / (s_max - s_min)
            else:
                frac = np.zeros_like(svals)
            sizes = point_size + frac * max(0.0, max_size - point_size)
        # Colors
        if color_map is not None:
            cvals = [color_map.get(v, SINGLE_COLOR) for v in sub[color][keep]]
            coll = ax.scatter(xs, ys, s=sizes, c=cvals,
                              linewidths=0.2, edgecolors="white")
        elif norm is not None:
            cvals = pd.to_numeric(sub[color], errors="coerce")[keep]
            coll = ax.scatter(xs, ys, s=sizes, c=cvals, cmap=cmap_obj, norm=norm,
                              linewidths=0.2, edgecolors="white")
        else:
            coll = ax.scatter(xs, ys, s=sizes, c=SINGLE_COLOR,
                              linewidths=0.2, edgecolors="white")
        # Hover metadata: exact plotted rows + the columns to show.
        coll._microvis_rows = kept
        coll._microvis_cols = _hover_cols(kept, x, y, color, size, hover_exclude)
        ax.set_xlabel(x, fontsize=8)
        ax.set_ylabel(y, fontsize=8)
        ax.tick_params(labelsize=7)
        _apply_categorical_axis(ax, xticks, yticks)
        _axis_style(ax, label)

    legends = []
    colorbar = None
    if color_map is not None:
        handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                          markerfacecolor=c, markeredgecolor="white", label=str(v))
                   for v, c in color_map.items()]
        legends.append({"handles": handles, "title": color})
    elif norm is not None:
        colorbar = {"mappable": ScalarMappable(norm=norm, cmap=cmap_obj),
                    "label": color}
    if s_min is not None:
        # Representative size legend (right strip, below the color legend).
        reps = np.linspace(s_min, s_max, 3) if s_max > s_min else np.array([s_min])
        size_handles = [
            Line2D([], [], marker="o", linestyle="", markersize=math.sqrt(
                point_size + (0 if s_max <= s_min else (v - s_min) / (s_max - s_min)
                              * max(0.0, max_size - point_size))) * 0.8,
                   markerfacecolor="none", markeredgecolor="#666666",
                   label=f"{v:.3g}")
            for v in reps
        ]
        legends.append({"handles": size_handles, "title": size})
    _finish_grid(fig, axes, len(groups), facet_cols, truncated,
                 legends=legends, colorbar=colorbar)
    return fig


# ── Boxplot / barplot ─────────────────────────────────────────────────────


def _distribution_groups(df, y, x, color):
    """Return (keys, labels, values_list, group_keys). Grouping keys are
    x (optionally) and a categorical color; continuous color never splits."""
    keys = []
    if x:
        keys.append(x)
    split_color = bool(color) and not is_continuous(df[color])
    if split_color:
        keys.append(color)
    raw = _group_series(df, y, keys)
    return keys, raw, split_color


def make_boxplot(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    color: str | None = None,
    facet_cols=(),
    palette: str = "Set1",
    show_points: bool = False,
    ncols: int = 3,
    max_combos: int = 24,
):
    """Faceted boxplot: one box per x (and categorical color) group.

    Y may be any variable: categorical values are placed on their level index
    with the level names as y-ticks.
    """
    yvals, yticks = axis_values(df[y])
    work = df.assign(__y__=yvals)
    groups, truncated = facet_groups(work, facet_cols, max_combos)
    fig, axes = _new_grid(len(groups), ncols)
    keys, raw_global, split_color = _distribution_groups(work, "__y__", x, color)
    color_map = None
    if split_color:
        color_map = _palette_map(work[color], palette)
    # Global group order + positions: every facet panel places the groups
    # it has at the SAME x positions (boxplot positions are 1-based), so
    # the shared axis compares like with like.
    global_keys = [k for k, _ in raw_global]
    pos_of = {k: i + 1 for i, k in enumerate(global_keys)}
    global_labels = [_group_label(k) for k in global_keys]

    for ax, (label, sub) in zip(axes, groups):
        grouped = _group_series(sub, "__y__", keys)
        data = [vals.to_numpy(dtype=float) for _, vals in grouped]
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            data, positions=[pos_of[k] for k, _ in grouped],
            patch_artist=True, showfliers=False, widths=0.6)
        for i, box in enumerate(bp["boxes"]):
            if color_map is not None and keys and keys[-1] == color:
                c = color_map.get(grouped[i][0][-1], SINGLE_COLOR)
            else:
                c = SINGLE_COLOR
            box.set_facecolor(c)
            box.set_alpha(0.85)
        if show_points:
            for i, (_, vals) in enumerate(grouped):
                jitter = np.random.default_rng(0).normal(0, 0.06, len(vals))
                ax.scatter(np.full(len(vals), pos_of[grouped[i][0]]) + jitter,
                           vals.to_numpy(dtype=float), s=3, alpha=0.4,
                           color="#333333")
        ax.set_ylabel(y, fontsize=8)
        if keys:
            ax.set_xticks(range(1, len(global_keys) + 1))
            ax.set_xticklabels(global_labels, fontsize=7, rotation=30,
                               ha="right")
        ax.tick_params(labelsize=7)
        if yticks is not None:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks, fontsize=7)
        _axis_style(ax, label)

    legends = []
    if color_map is not None:
        handles = [Line2D([], [], marker="s", linestyle="", markersize=6,
                          markerfacecolor=c, markeredgecolor="none", label=str(v))
                   for v, c in color_map.items()]
        legends.append({"handles": handles, "title": color})
    _finish_grid(fig, axes, len(groups), facet_cols, truncated,
                 legends=legends)
    return fig


def make_barplot_mean_sem(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    color: str | None = None,
    facet_cols=(),
    palette: str = "Set1",
    ncols: int = 3,
    max_combos: int = 24,
):
    """Faceted barplot: bar height = mean, error bar = SEM.

    Y may be any variable: categorical values are placed on their level index
    with the level names as y-ticks.
    """
    yvals, yticks = axis_values(df[y])
    work = df.assign(__y__=yvals)
    groups, truncated = facet_groups(work, facet_cols, max_combos)
    fig, axes = _new_grid(len(groups), ncols)
    keys, raw_global, split_color = _distribution_groups(work, "__y__", x, color)
    color_map = None
    if split_color:
        color_map = _palette_map(work[color], palette)
    # Global group order + positions (0-based bar slots): every facet panel
    # draws its groups at the SAME x positions so the shared axis compares
    # like with like.
    global_keys = [k for k, _ in raw_global]
    slot_of = {k: i for i, k in enumerate(global_keys)}
    global_labels = [_group_label(k) for k in global_keys]

    for ax, (label, sub) in zip(axes, groups):
        grouped = _group_series(sub, "__y__", keys)
        means = [vals.mean() for _, vals in grouped]
        # A single observation has no SEM — NaN (matplotlib draws no bar)
        # instead of 0.0, which would read as "mean of many, tiny variance".
        sems = [vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1
                else float("nan")
                for _, vals in grouped]
        if not means:
            ax.set_visible(False)
            continue
        positions = [slot_of[k] for k, _ in grouped]
        colors = []
        for i, (k, _) in enumerate(grouped):
            if color_map is not None and keys and keys[-1] == color:
                colors.append(color_map.get(k[-1], SINGLE_COLOR))
            else:
                colors.append(SINGLE_COLOR)
        ax.bar(positions, means, yerr=sems, capsize=3, color=colors,
               edgecolor="#333333", linewidth=0.5, alpha=0.9)
        ax.set_ylabel(y, fontsize=8)
        if keys:
            ax.set_xticks(range(len(global_keys)))
            ax.set_xticklabels(global_labels, fontsize=7, rotation=30,
                               ha="right")
        if yticks is not None:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks, fontsize=7)
        _axis_style(ax, label)

    legends = []
    if color_map is not None:
        handles = [Line2D([], [], marker="s", linestyle="", markersize=6,
                          markerfacecolor=c, markeredgecolor="none", label=str(v))
                   for v, c in color_map.items()]
        legends.append({"handles": handles, "title": color})
    _finish_grid(fig, axes, len(groups), facet_cols, truncated,
                 legends=legends)
    return fig


def make_line(
    df: pd.DataFrame,
    y: str,
    x: str,
    color: str | None = None,
    facet_cols=(),
    palette: str = "Set1",
    ncols: int = 3,
    max_combos: int = 24,
    show_points: bool = False,
):
    """Faceted smooth-line plot: a spline fit of y on x, ± SEM nodes.

    CONTINUOUS x: one smooth fit (scipy default spline) of the raw y on the
    numeric x per categorical-color group — a real numeric x axis, no
    grouping. CATEGORICAL x: one smooth curve through the per-group means
    (SEM error bars on the nodes) over the shared level positions. Raw
    observations draw as small dots behind when `show_points`. Y may be any
    variable: categorical values are placed on their level index with the
    level names as y-ticks.
    """
    yvals, yticks = axis_values(df[y])
    work = df.assign(__y__=yvals)
    groups, truncated = facet_groups(work, facet_cols, max_combos)
    fig, axes = _new_grid(len(groups), ncols)
    x_continuous = bool(x) and is_continuous(work[x])
    keys, raw_global, split_color = _distribution_groups(work, "__y__", x, color)
    color_map = None
    color_vals = None
    if split_color:
        color_map = _palette_map(work[color], palette)
        color_vals = _sorted_values(work[color].dropna())
    # Shared x axis for the CATEGORICAL branch: level order/positions come
    # from the FULL table so every facet panel draws its groups at the SAME
    # positions. In the split branch the first key element is x (or the
    # categorical color when no x was chosen).
    if not x_continuous:
        if split_color:
            global_levels = _sorted_values(work[(x or color)].dropna())
            pos_of = {v: i for i, v in enumerate(global_levels)}
        else:
            global_keys = [k for k, _ in raw_global]
            pos_of = {k: i for i, k in enumerate(global_keys)}
            global_labels = [_group_label(k) for k in global_keys]

    for ax, (label, sub) in zip(axes, groups):
        if x_continuous:
            # Numeric x axis: smooth fit of the raw y on the numeric x —
            # no grouping, so no artificial category per distinct value.
            xv = pd.to_numeric(sub[x], errors="coerce")
            yv = sub["__y__"]
            keep = (xv.notna() & yv.notna()).to_numpy()
            xv = xv.to_numpy(dtype=float)[keep]
            yv = yv.to_numpy(dtype=float)[keep]
            if xv.size == 0:
                ax.set_visible(False)
                continue
            if split_color:
                colors = sub[color].to_numpy()[keep]
                for cval in color_vals:
                    c = color_map.get(cval, SINGLE_COLOR)
                    m = colors == cval
                    if not m.any():
                        continue
                    cx, cy = _smooth_curve(xv[m], yv[m])
                    ax.plot(cx, cy, color=c, linewidth=1.4, zorder=3)
                    if show_points:
                        ax.scatter(xv[m], yv[m], s=4, alpha=0.3, color=c,
                                   edgecolors="none", zorder=2)
            else:
                cx, cy = _smooth_curve(xv, yv)
                ax.plot(cx, cy, color="#555555", linewidth=1.4, zorder=3)
                if show_points:
                    ax.scatter(xv, yv, s=4, alpha=0.3, color=SINGLE_COLOR,
                               edgecolors="none", zorder=2)
            ax.set_xlabel(x, fontsize=8)
            ax.tick_params(labelsize=7)
            if yticks is not None:
                ax.set_yticks(range(len(yticks)))
                ax.set_yticklabels(yticks, fontsize=7)
            _axis_style(ax, label)
            continue
        grouped = _group_series(sub, "__y__", keys)
        if not grouped:
            ax.set_visible(False)
            continue
        # Raw observations as dots behind the mean line (light jitter only
        # spreads identical values; the node x stays the group position).
        rng = np.random.default_rng(0)

        def _mean_sem(vals):
            m = vals.mean()
            # n=1 has no SEM — NaN keeps the node but draws no error bar
            # (0.0 would fake certainty for a single observation).
            e = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 \
                else float("nan")
            return m, e

        if split_color:
            # One smooth fit per color group over the shared x levels
            # (positions are the global ones from the full table). _group_
            # series natsorts by (x, color); within one color the nodes are
            # already in x order.
            color_vals = list(dict.fromkeys(k[1] for k, _ in grouped))
            for cval in color_vals:
                c = color_map.get(cval, SINGLE_COLOR)
                pts = [(pos_of[k[0]], vals)
                       for k, vals in grouped if k[1] == cval]
                xs = [p for p, _ in pts]
                stats = [_mean_sem(vals) for _, vals in pts]
                ms = [m for m, _ in stats]
                es = [e for _, e in stats]
                if show_points:
                    for p, vals in pts:
                        jitter = rng.normal(0, 0.04, len(vals))
                        ax.scatter(np.full(len(vals), p) + jitter,
                                   vals.to_numpy(dtype=float), s=4, alpha=0.3,
                                   color=c, edgecolors="none", zorder=2)
                cx, cy = _smooth_curve(xs, ms)
                ax.plot(cx, cy, color=c, linewidth=1.4, zorder=3)
                ax.errorbar(xs, ms, yerr=es, fmt="o", markersize=5,
                            capsize=3, color=c, ecolor=c, zorder=4)
            ax.set_xticks(range(len(global_levels)))
            ax.set_xticklabels([str(v) for v in global_levels], fontsize=7,
                               rotation=30, ha="right")
        else:
            means = [vals.mean() for _, vals in grouped]
            sems = [_mean_sem(vals)[1] for _, vals in grouped]
            positions = [pos_of[k] for k, _ in grouped]
            if show_points:
                for k, vals in grouped:
                    jitter = rng.normal(0, 0.04, len(vals))
                    ax.scatter(np.full(len(vals), pos_of[k]) + jitter,
                               vals.to_numpy(dtype=float), s=4, alpha=0.3,
                               color=SINGLE_COLOR, edgecolors="none", zorder=2)
            if len(positions) > 1:
                cx, cy = _smooth_curve(positions, means)
                ax.plot(cx, cy, color="#555555", linewidth=1.4, zorder=3)
            ax.errorbar(positions, means, yerr=sems, fmt="o", markersize=5,
                        capsize=3, color="#333333", ecolor="#333333",
                        zorder=4)
            if x:
                ax.set_xticks(range(len(global_keys)))
                ax.set_xticklabels(global_labels, fontsize=7, rotation=30,
                                   ha="right")
        ax.set_ylabel(y, fontsize=8)
        if yticks is not None:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks, fontsize=7)
        _axis_style(ax, label)

    legends = []
    if color_map is not None:
        handles = [Line2D([], [], marker="o", linestyle="-", markersize=5,
                          markerfacecolor=c, markeredgecolor=c, label=str(v))
                   for v, c in color_map.items()]
        legends.append({"handles": handles, "title": color})
    _finish_grid(fig, axes, len(groups), facet_cols, truncated,
                 legends=legends)
    return fig


# ── Export ────────────────────────────────────────────────────────────────


def save_figure_pdf(fig, path) -> None:
    """Write the figure as a vector PDF with editable text."""
    fig.savefig(str(path), format="pdf", bbox_inches="tight")
