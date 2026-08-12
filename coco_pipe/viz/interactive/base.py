"""General-purpose interactive (Plotly) visualization helpers.

Mirrors the static helpers in :mod:`coco_pipe.viz.base` (``plot_bar``,
``plot_distribution_groups``, ``plot_heatmap``) for cases where the report
or notebook wants an interactive figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import _COLORBLIND_COLORS, DIVERGING, SEQUENTIAL
from ._utils import _apply_layout

__all__ = [
    "plot_bar",
    "plot_distribution_groups",
    "plot_group_scatter_with_mean",
    "plot_grouped_bar",
    "plot_heatmap",
    "plot_ranked_bar",
    "plot_scatter",
]


def _coerce_named_series(
    values: pd.Series | Mapping[str, float] | Sequence[float],
    labels: Sequence[str] | None,
) -> pd.Series:
    """Normalize a Series / dict / sequence to a labelled Series."""
    if isinstance(values, pd.Series):
        return values.copy()
    if isinstance(values, Mapping):
        return pd.Series(dict(values))
    arr = np.asarray(list(values))
    if labels is None:
        labels = [str(i) for i in range(len(arr))]
    if len(labels) != len(arr):
        raise ValueError(
            f"`labels` length {len(labels)} does not match values length {len(arr)}."
        )
    return pd.Series(arr, index=list(labels))


def plot_bar(
    scores: pd.Series | Mapping[str, float] | Sequence[float],
    errors: pd.Series | Mapping[str, float] | Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    label_map: Mapping[str, str] | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    sort: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    color: str | Sequence[str] | None = None,
    cmap: str | None = None,
    abs_values: bool = False,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive ranked bar chart with optional error bars.

    Parameters
    ----------
    scores
        Values to plot. Accepts a ``pd.Series``, ``{label: value}`` mapping,
        or a plain sequence (then ``labels`` is required for tick labels).
    errors
        Optional symmetric error magnitudes aligned with ``scores``.
    labels
        Tick labels used when ``scores`` is a plain sequence.
    label_map
        Optional ``{raw_label: display_label}`` mapping applied to tick labels.
    top_n
        Keep only the first ``top_n`` entries after sorting.
    ascending
        Sort direction when ``sort=True``.
    sort
        Whether to sort bars by value before plotting.
    orientation
        ``"vertical"`` (default) or ``"horizontal"``.
    color
        Bar color or per-bar color sequence. Ignored when ``cmap`` is set.
    cmap
        Plotly colorscale name used to color bars by value.
    abs_values
        Sort and plot by absolute values while preserving the sign for errors.
    title, xaxis_title, yaxis_title
        Layout labels.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive bar chart.

    See Also
    --------
    coco_pipe.viz.base.plot_bar : Static Matplotlib version.
    plot_distribution_groups : Per-group distribution comparison.
    """
    series = _coerce_named_series(scores, labels)
    if errors is not None:
        err_series = _coerce_named_series(errors, list(series.index))
        err_series = err_series.reindex(series.index)
    else:
        err_series = None

    sort_key = series.abs() if abs_values else series

    if sort:
        order = sort_key.sort_values(ascending=ascending).index
        series = series.loc[order]
        if err_series is not None:
            err_series = err_series.loc[order]

    if top_n is not None:
        series = series.iloc[:top_n]
        if err_series is not None:
            err_series = err_series.iloc[:top_n]

    display_labels = [
        label_map.get(idx, idx) if label_map else idx for idx in series.index
    ]

    if cmap is not None:
        # Color by value
        marker = {
            "color": series.values.tolist(),
            "colorscale": cmap,
            "colorbar": {"title": yaxis_title or "Value"}
            if orientation == "vertical"
            else {"title": xaxis_title or "Value"},
        }
    else:
        bar_color = color if color is not None else _COLORBLIND_COLORS[0]
        marker = {"color": bar_color}

    error_kwargs: dict[str, Any] = {}
    if err_series is not None:
        error_kwargs["error_" + ("y" if orientation == "vertical" else "x")] = {
            "type": "data",
            "array": err_series.values.tolist(),
            "visible": True,
        }

    if orientation == "vertical":
        bar = go.Bar(
            x=[str(label) for label in display_labels],
            y=series.values.tolist(),
            marker=marker,
            **error_kwargs,
        )
    else:
        bar = go.Bar(
            y=[str(label) for label in display_labels],
            x=series.values.tolist(),
            orientation="h",
            marker=marker,
            **error_kwargs,
        )

    fig = go.Figure(bar)
    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    return fig


def _resolve_group_colors(
    keys: Sequence[Any],
    color_map: Mapping[Any, str] | None,
) -> list[str]:
    """Assign a color per group key, honoring ``color_map`` then the palette."""
    colors: list[str] = []
    for i, key in enumerate(keys):
        if color_map is not None and key in color_map:
            colors.append(color_map[key])
        else:
            colors.append(_COLORBLIND_COLORS[i % len(_COLORBLIND_COLORS)])
    return colors


def _apply_value_baseline(
    fig: go.Figure,
    *,
    horizontal: bool,
    baseline: float | None,
    baseline_label: str | None,
    value_range: Sequence[float] | None,
) -> None:
    """Clamp the value axis and draw a dashed reference line at *baseline*.

    ``horizontal`` selects which visual axis carries the values (x for
    horizontal bars, y otherwise). Used to anchor an axis at a meaningful floor
    (e.g. chance for balanced accuracy) and mark that level.
    """
    if value_range is not None:
        if horizontal:
            fig.update_xaxes(range=list(value_range))
        else:
            fig.update_yaxes(range=list(value_range))
    if baseline is None:
        return
    kwargs: dict[str, Any] = {
        "line_dash": "dash",
        "line_color": "#888",
        "line_width": 1.5,
    }
    if baseline_label:
        kwargs["annotation_text"] = baseline_label
        kwargs["annotation_position"] = "top left" if horizontal else "top right"
    if horizontal:
        fig.add_vline(x=baseline, **kwargs)
    else:
        fig.add_hline(y=baseline, **kwargs)


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    text: str | None = None,
    hovertemplate: str | None = None,
    mode: str = "markers",
    sort_x: bool | None = None,
    color_map: Mapping[Any, str] | None = None,
    marker_size: int = 8,
    line_width: float = 2.0,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive 2D scatter / line-scatter, optionally split into colored groups.

    Parameters
    ----------
    data
        Source DataFrame.
    x, y
        Column names for the x and y axes.
    color
        Optional column name. When given, one trace is drawn per unique value
        (each gets a legend entry and palette color); otherwise a single trace
        is drawn.
    text
        Optional column name supplying per-point hover/label text.
    hovertemplate
        Optional Plotly ``hovertemplate`` applied to every trace. ``%{text}``
        refers to the ``text`` column.
    mode
        Plotly scatter mode, e.g. ``"markers"`` (default), ``"lines+markers"``,
        or ``"lines"``.
    sort_x
        Whether to sort each trace by ``x`` before plotting. ``None`` (default)
        auto-enables sorting when ``mode`` contains ``"lines"`` so line segments
        connect in order.
    color_map
        Optional ``{group_value: color}`` overriding palette colors for groups.
    marker_size
        Marker size in pixels.
    line_width
        Line width for line modes.
    title, xaxis_title, yaxis_title
        Layout labels. Axis titles default to the ``x`` / ``y`` column names.
    legend_title
        Optional legend title (typically the ``color`` column name).
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter figure.

    See Also
    --------
    plot_bar : Ranked bar chart.
    plot_grouped_bar : Grouped bar chart from a long DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    for col in (x, y, color, text):
        if col is not None and col not in data.columns:
            raise KeyError(f"Column {col!r} not found in `data`.")

    if sort_x is None:
        sort_x = "lines" in mode

    if color is None:
        groups: list[tuple[Any, pd.DataFrame]] = [(None, data)]
    else:
        groups = list(data.groupby(color, dropna=False, sort=False))

    colors = _resolve_group_colors([key for key, _ in groups], color_map)
    is_line = "lines" in mode

    fig = go.Figure()
    for (key, group), trace_color in zip(groups, colors, strict=False):
        if sort_x:
            group = group.sort_values(x)
        trace_kwargs: dict[str, Any] = {
            "x": group[x].tolist(),
            "y": group[y].tolist(),
            "mode": mode,
            "marker": {"color": trace_color, "size": marker_size},
            "showlegend": color is not None,
        }
        if color is not None:
            trace_kwargs["name"] = str(key)
        if is_line:
            trace_kwargs["line"] = {"color": trace_color, "width": line_width}
        if text is not None:
            trace_kwargs["text"] = [str(t) for t in group[text].tolist()]
        if hovertemplate is not None:
            trace_kwargs["hovertemplate"] = hovertemplate
        fig.add_trace(go.Scatter(**trace_kwargs))

    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y,
        height=height,
    )
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def plot_grouped_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    text: str | None = None,
    text_position: str = "outside",
    color_map: Mapping[Any, str] | None = None,
    x_order: Sequence[Any] | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    baseline: float | None = None,
    baseline_label: str | None = None,
    value_range: Sequence[float] | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive grouped bar chart (``barmode="group"``) from a long DataFrame.

    Parameters
    ----------
    data
        Long-format DataFrame: one row per ``(x, group)`` bar.
    x
        Column name for the category axis (shared across groups). Placed on the
        x-axis when vertical and the y-axis when horizontal.
    y
        Column name for the bar values.
    group
        Column name defining the bar groups; one trace (legend entry) is drawn
        per unique value.
    text
        Optional column name supplying per-bar text labels.
    text_position
        Plotly ``textposition`` for bar labels (default ``"outside"``).
    color_map
        Optional ``{group_value: color}`` overriding palette colors.
    x_order
        Optional explicit ordering for the category (``x``) axis, applied to
        whichever visual axis carries the categories.
    orientation
        ``"vertical"`` (default) or ``"horizontal"``. Horizontal puts the
        category (``x``) on the y-axis and the values (``y``) on the x-axis,
        which reads better for many long category labels.
    title, xaxis_title, yaxis_title
        Layout labels. Axis titles refer to the literal visual axes and default
        to the appropriate column name for the orientation.
    legend_title
        Optional legend title (typically the ``group`` column name).
    baseline
        Optional reference value drawn as a dashed line on the value axis (e.g.
        chance for balanced accuracy).
    baseline_label
        Optional annotation for the *baseline* line.
    value_range
        Optional ``(low, high)`` range for the value axis, e.g. to anchor it at
        chance instead of zero.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive grouped bar chart.

    See Also
    --------
    plot_bar : Single-series ranked bar chart.
    plot_scatter : Scatter / line-scatter from a long DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    for col in (x, y, group, text):
        if col is not None and col not in data.columns:
            raise KeyError(f"Column {col!r} not found in `data`.")

    horizontal = orientation == "horizontal"
    groups = list(data.groupby(group, dropna=False, sort=False))
    colors = _resolve_group_colors([key for key, _ in groups], color_map)

    fig = go.Figure()
    for (key, sub), trace_color in zip(groups, colors, strict=False):
        categories = sub[x].tolist()
        values = sub[y].tolist()
        bar_kwargs: dict[str, Any] = {
            "name": str(key),
            "marker": {"color": trace_color},
        }
        if horizontal:
            bar_kwargs.update(y=categories, x=values, orientation="h")
        else:
            bar_kwargs.update(x=categories, y=values)
        if text is not None:
            bar_kwargs["text"] = [str(t) for t in sub[text].tolist()]
            bar_kwargs["textposition"] = text_position
        fig.add_trace(go.Bar(**bar_kwargs))

    # Axis titles refer to the literal visual axes; the category (`x`) column
    # names the category axis and the value (`y`) column names the value axis.
    cat_default, val_default = x, y
    _apply_layout(
        fig,
        title=title,
        xaxis_title=(
            xaxis_title
            if xaxis_title is not None
            else (val_default if horizontal else cat_default)
        ),
        yaxis_title=(
            yaxis_title
            if yaxis_title is not None
            else (cat_default if horizontal else val_default)
        ),
        height=height,
        barmode="group",
    )
    if x_order is not None:
        category_axis = fig.update_yaxes if horizontal else fig.update_xaxes
        category_axis(categoryorder="array", categoryarray=[str(v) for v in x_order])
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    _apply_value_baseline(
        fig,
        horizontal=horizontal,
        baseline=baseline,
        baseline_label=baseline_label,
        value_range=value_range,
    )
    return fig


def plot_ranked_bar(
    data: pd.DataFrame,
    *,
    value: str,
    category: str,
    color: str,
    text: str | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    orientation: Literal["vertical", "horizontal"] = "horizontal",
    color_map: Mapping[Any, str] | None = None,
    title: str | None = None,
    value_title: str | None = None,
    category_title: str | None = None,
    legend_title: str | None = None,
    baseline: float | None = None,
    baseline_label: str | None = None,
    value_range: Sequence[float] | None = None,
    height: int | None = None,
) -> go.Figure:
    """Ranked grouped bar: sort by *value*, keep *top_n*, color by *color*.

    A thin ranking layer over :func:`plot_grouped_bar`. Rows are sorted by
    *value* (descending by default), truncated to *top_n*, and colored by the
    *color* column (one legend entry per value). Each row is one bar, so
    *category* must be unique per row — it supplies the per-bar tick label.
    Horizontal orientation (the default, better for many long labels) places the
    highest-ranked bar at the top; vertical keeps rank order left-to-right.

    Parameters
    ----------
    data
        Long-format DataFrame, one row per bar.
    value
        Column with the numeric bar values driving the ranking.
    category
        Column supplying the per-bar tick label (unique per row).
    color
        Column defining the bar color / legend groups.
    text
        Optional column supplying per-bar text labels.
    top_n
        Keep only the first ``top_n`` rows after sorting.
    ascending
        Sort direction (default ``False`` — highest value first).
    orientation
        ``"horizontal"`` (default) or ``"vertical"``.
    color_map
        Optional ``{color_value: color}`` overriding palette colors.
    title, value_title, category_title
        Layout labels; axis titles map to the value / category axes for the
        chosen orientation.
    legend_title
        Optional legend title (typically the *color* column name).
    baseline, baseline_label, value_range
        Forwarded to :func:`plot_grouped_bar` — a dashed reference line on the
        value axis and an explicit value-axis range (e.g. anchored at chance).
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive ranked, grouped bar chart.

    See Also
    --------
    plot_grouped_bar : Grouped bars without the ranking / top-N layer.
    plot_bar : Single-series ranked bar chart (no color grouping).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    for col in (value, category, color, text):
        if col is not None and col not in data.columns:
            raise KeyError(f"Column {col!r} not found in `data`.")

    ranked = data.sort_values(value, ascending=ascending, na_position="last")
    if top_n is not None:
        ranked = ranked.head(top_n)

    horizontal = orientation == "horizontal"
    order = ranked[category].tolist()
    # Horizontal stacks the first category at the bottom, so reverse to lift the
    # top-ranked bar to the top; vertical keeps left-to-right rank order.
    category_order = order[::-1] if horizontal else order

    return plot_grouped_bar(
        ranked,
        x=category,
        y=value,
        group=color,
        text=text,
        color_map=color_map,
        x_order=category_order,
        orientation=orientation,
        title=title,
        xaxis_title=value_title if horizontal else category_title,
        yaxis_title=category_title if horizontal else value_title,
        legend_title=legend_title,
        baseline=baseline,
        baseline_label=baseline_label,
        value_range=value_range,
        height=height,
    )


def plot_distribution_groups(
    groups: Sequence[Sequence[float] | np.ndarray | pd.Series],
    labels: Sequence[Any],
    kind: Literal["box", "violin"] = "box",
    show_points: bool = True,
    point_opacity: float = 0.55,
    showmeans: bool = True,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str | Sequence[str] | None = None,
    height: int | None = None,
    sig_pairs: Sequence[tuple[int, int, str]] | None = None,
    baseline: float | None = None,
    baseline_label: str | None = None,
    value_range: Sequence[float] | None = None,
) -> go.Figure:
    """Interactive grouped distribution plot (box or violin) with overlaid points.

    Parameters
    ----------
    groups
        Sequence of data arrays, one per group. Non-finite values are dropped.
    labels
        Group labels aligned with ``groups``.
    kind
        ``"box"`` (default) or ``"violin"``.
    show_points
        Overlay individual points on top of each box / violin.
    point_opacity
        Opacity for overlaid individual points.
    showmeans
        Whether to mark the mean inside each box / violin.
    title, xaxis_title, yaxis_title
        Layout labels.
    color
        Single color or per-group color list. Defaults to the colorblind palette.
    height
        Figure height in pixels.
    sig_pairs
        Optional sequence of ``(idx_a, idx_b, annotation)`` triples drawing
        a horizontal significance bracket between groups at indices
        ``idx_a`` and ``idx_b`` with the given annotation (e.g., ``"*"``,
        ``"p=0.01"``).
    baseline
        Optional reference value drawn as a dashed horizontal line (e.g. chance).
    baseline_label
        Optional annotation for the *baseline* line.
    value_range
        Optional ``(low, high)`` range for the value (y) axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive distribution figure.

    See Also
    --------
    coco_pipe.viz.base.plot_distribution_groups : Static Matplotlib version.
    plot_bar : Bar chart with optional error bars.
    """
    if len(groups) != len(labels):
        raise ValueError(
            f"`groups` length {len(groups)} != `labels` length {len(labels)}."
        )
    if kind not in {"box", "violin"}:
        raise ValueError("`kind` must be 'box' or 'violin'.")

    if color is None:
        color_list = [
            _COLORBLIND_COLORS[i % len(_COLORBLIND_COLORS)] for i in range(len(groups))
        ]
    elif isinstance(color, str):
        color_list = [color] * len(groups)
    else:
        color_list = list(color)
        if len(color_list) < len(groups):
            color_list = (color_list * len(groups))[: len(groups)]

    fig = go.Figure()
    for idx, (data, label) in enumerate(zip(groups, labels, strict=False)):
        arr = np.asarray(list(data), dtype=float)
        arr = arr[np.isfinite(arr)]
        trace_color = color_list[idx]
        common = {
            "y": arr.tolist(),
            "name": str(label),
            "marker_color": trace_color,
            "line_color": trace_color,
            "showlegend": False,
        }
        if kind == "box":
            trace = go.Box(
                **common,
                boxmean=showmeans,
                boxpoints="all" if show_points else "outliers",
                jitter=0.3 if show_points else 0.0,
                pointpos=0.0,
                marker={"opacity": point_opacity, "size": 5},
            )
        else:
            trace = go.Violin(
                **common,
                meanline_visible=showmeans,
                points="all" if show_points else False,
                jitter=0.3 if show_points else 0.0,
                marker={"opacity": point_opacity, "size": 5},
                box_visible=True,
            )
        fig.add_trace(trace)

    # Significance brackets
    if sig_pairs:
        # Compute a y-position above the highest data point
        all_max = max(
            (float(np.nanmax(np.asarray(list(g)))) for g in groups if len(list(g)) > 0),
            default=0.0,
        )
        all_min = min(
            (float(np.nanmin(np.asarray(list(g)))) for g in groups if len(list(g)) > 0),
            default=0.0,
        )
        span = max(all_max - all_min, 1e-9)
        bracket_y = all_max + 0.05 * span
        bracket_step = 0.08 * span
        for i, (idx_a, idx_b, annotation) in enumerate(sig_pairs):
            y = bracket_y + i * bracket_step
            x_a = str(labels[idx_a])
            x_b = str(labels[idx_b])
            fig.add_shape(
                type="line",
                x0=x_a,
                x1=x_b,
                y0=y,
                y1=y,
                line={"color": "black", "width": 1.2},
            )
            fig.add_annotation(
                x=x_a,
                y=y,
                xshift=int(0.5 * 30),  # approximate centering
                text=annotation,
                showarrow=False,
                yshift=8,
                font={"size": 12},
            )

    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    _apply_value_baseline(
        fig,
        horizontal=False,
        baseline=baseline,
        baseline_label=baseline_label,
        value_range=value_range,
    )
    return fig


def plot_group_scatter_with_mean(
    groups: Sequence[Sequence[float] | np.ndarray | pd.Series],
    labels: Sequence[Any],
    error: Literal["sem", "sd"] = "sem",
    point_jitter: float = 0.12,
    point_labels: Sequence[Sequence[Any]] | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str | Sequence[str] | None = None,
    height: int | None = None,
    baseline: float | None = None,
    baseline_label: str | None = None,
    value_range: Sequence[float] | None = None,
) -> go.Figure:
    """Per-group scatter of individual observations plus a mean +/- error-bar marker.

    Unlike :func:`plot_distribution_groups` (box/violin summaries), this draws
    each observation (e.g. one point per subject) as a jittered marker and
    overlays a single larger marker at the group mean with an error bar —
    the "one dot per subject, plus average with error bars" comparison used to
    show a representation-to-representation enhancement.

    Parameters
    ----------
    groups
        Sequence of data arrays, one per group. Non-finite values are dropped.
    labels
        Group labels aligned with ``groups``, shown as x-axis tick labels.
    error
        Error-bar type for the mean marker: ``"sem"`` (default) or ``"sd"``.
    point_jitter
        Half-width of the horizontal jitter applied to individual points, in
        x-axis units (groups are one unit apart).
    point_labels
        Optional per-point hover labels (e.g. subject ids), aligned with
        ``groups``.
    title, xaxis_title, yaxis_title
        Layout labels.
    color
        Single color or per-group color list. Defaults to the colorblind palette.
    height
        Figure height in pixels.
    baseline
        Optional reference value drawn as a dashed horizontal line (e.g. chance).
    baseline_label
        Optional annotation for the *baseline* line.
    value_range
        Optional ``(low, high)`` range for the value (y) axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter-plus-mean figure.

    See Also
    --------
    plot_distribution_groups : Box/violin group comparison with overlaid points.
    plot_bar : Bar chart with optional error bars from pre-aggregated values.
    """
    if len(groups) != len(labels):
        raise ValueError(
            f"`groups` length {len(groups)} != `labels` length {len(labels)}."
        )
    if error not in {"sem", "sd"}:
        raise ValueError("`error` must be 'sem' or 'sd'.")
    if point_labels is not None and len(point_labels) != len(groups):
        raise ValueError(
            f"`point_labels` length {len(point_labels)} != `groups` length {len(groups)}."
        )

    if color is None:
        color_list = [
            _COLORBLIND_COLORS[i % len(_COLORBLIND_COLORS)] for i in range(len(groups))
        ]
    elif isinstance(color, str):
        color_list = [color] * len(groups)
    else:
        color_list = list(color)
        if len(color_list) < len(groups):
            color_list = (color_list * len(groups))[: len(groups)]

    rng = np.random.default_rng(0)
    fig = go.Figure()
    for idx, (data, label) in enumerate(zip(groups, labels, strict=False)):
        arr = np.asarray(list(data), dtype=float)
        finite = np.isfinite(arr)
        arr = arr[finite]
        trace_color = color_list[idx]
        if len(arr) == 0:
            continue

        hover_text = None
        if point_labels is not None:
            point_arr = np.asarray(list(point_labels[idx]), dtype=object)[finite]
            hover_text = [str(value) for value in point_arr]

        jitter = (
            rng.uniform(-point_jitter, point_jitter, size=len(arr))
            if len(arr) > 1
            else [0.0]
        )
        fig.add_trace(
            go.Scatter(
                x=(idx + np.asarray(jitter)).tolist(),
                y=arr.tolist(),
                mode="markers",
                marker={"color": trace_color, "opacity": 0.55, "size": 7},
                name=str(label),
                text=hover_text,
                hoverinfo="y+text" if hover_text is not None else "y",
                showlegend=False,
            )
        )

        mean = float(np.mean(arr))
        spread = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        err = spread / np.sqrt(len(arr)) if error == "sem" and len(arr) > 1 else spread
        fig.add_trace(
            go.Scatter(
                x=[idx],
                y=[mean],
                mode="markers",
                marker={
                    "color": trace_color,
                    "size": 13,
                    "symbol": "diamond",
                    "line": {"color": "black", "width": 1},
                },
                error_y={
                    "type": "data",
                    "array": [err],
                    "visible": True,
                    "thickness": 2,
                },
                name=f"{label} mean",
                showlegend=False,
            )
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=[str(label) for label in labels],
    )
    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    _apply_value_baseline(
        fig,
        horizontal=False,
        baseline=baseline,
        baseline_label=baseline_label,
        value_range=value_range,
    )
    return fig


def plot_heatmap(
    matrix: pd.DataFrame | Sequence[Sequence[float]] | np.ndarray,
    x_labels: Sequence[Any] | None = None,
    y_labels: Sequence[Any] | None = None,
    cmap: str | None = None,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    annotation_format: str = ".3g",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_label: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive 2D heatmap.

    Parameters
    ----------
    matrix
        2D matrix to display. A DataFrame provides default axis labels.
    x_labels, y_labels
        Tick labels. Override DataFrame columns / index when provided.
    cmap
        Plotly colorscale. Defaults to ``SEQUENTIAL`` unless ``center`` is
        provided, in which case ``DIVERGING`` is used.
    center
        Value mapped to the colormap midpoint (uses ``DIVERGING``).
    vmin, vmax
        Explicit colormap bounds.
    annotate
        Print values inside cells.
    annotation_format
        Python format spec for cell annotations.
    title, xaxis_title, yaxis_title, colorbar_label
        Layout labels.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap.

    See Also
    --------
    coco_pipe.viz.base.plot_heatmap : Static Matplotlib version.
    plot_coranking_matrix : Square coranking-matrix heatmap.
    """
    if isinstance(matrix, pd.DataFrame):
        arr = matrix.values
        x_lab = (
            [str(c) for c in matrix.columns]
            if x_labels is None
            else [str(c) for c in x_labels]
        )
        y_lab = (
            [str(c) for c in matrix.index]
            if y_labels is None
            else [str(c) for c in y_labels]
        )
    else:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"`matrix` must be 2D; got shape {arr.shape}.")
        x_lab = (
            [str(c) for c in x_labels]
            if x_labels is not None
            else [str(i) for i in range(arr.shape[1])]
        )
        y_lab = (
            [str(c) for c in y_labels]
            if y_labels is not None
            else [str(i) for i in range(arr.shape[0])]
        )

    if cmap is None:
        cmap = DIVERGING if center is not None else SEQUENTIAL

    heatmap_kwargs: dict[str, Any] = {
        "z": arr,
        "x": x_lab,
        "y": y_lab,
        "colorscale": cmap,
        "colorbar": {"title": colorbar_label or ""},
    }
    if center is not None:
        heatmap_kwargs["zmid"] = center
    if vmin is not None:
        heatmap_kwargs["zmin"] = vmin
    if vmax is not None:
        heatmap_kwargs["zmax"] = vmax
    if annotate:
        heatmap_kwargs["text"] = [
            [format(v, annotation_format) for v in row] for row in arr
        ]
        heatmap_kwargs["texttemplate"] = "%{text}"
        heatmap_kwargs["textfont"] = {"size": 10}

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))
    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
    )
    # Heatmaps usually look better with a square aspect; let Plotly default for now.
    return fig


def plot_timecourses(
    data: np.ndarray | pd.DataFrame,
    times: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    rois: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    group_labels: Sequence[Any] | None = None,
    group_name_map: Mapping[Any, str] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | None = None,
    linestyle_map: Mapping[Any, str] | None = None,
    n_cols: int | None = None,
    error_style: str = "band",
    xlabel: str = "Time",
    ylabel: str = "Amplitude",
    title: str | None = "Grouped Timecourses",
    sharex: bool = True,
    sharey: bool = True,
    base_height: int = 300,
    row_height: int = 220,
    showlegend: bool = True,
    line_width: float = 2.0,
    opacity: float = 1.0,
    add_zero: bool = False,
) -> go.Figure:
    """Plot interactive timecourses (ERPs, ERFs, source activations)
    across channels or ROIs.

    Parameters
    ----------
    data
        Shape ``(n_trials, n_channels, n_times)`` or ``(n_channels, n_times)``.
    times
        Time axis values, shape ``(n_times,)``.
    channel_names
        Names of the channels in ``data``. Required if ``rois`` specifies channel names.
        If ``data`` is ``(n_channels, n_times)`` and ``channel_names`` is omitted,
        channels are numbered.
    rois
        Mapping of ``{roi_name: [channel_names]}`` to average over. If a sequence
        of strings is provided, each string is treated as a single-channel ROI.
        If None, all channels are plotted individually.
    group_labels
        Shape ``(n_trials,)`` assigning each trial to a group. If None, all trials are
        averaged together.
    group_name_map
        Mapping to rename groups for the legend.
    palette
        Color mapping for groups. Can be a dictionary or a list of colors.
    linestyle_map
        Linestyle mapping for groups.
    n_cols
        Maximum number of columns for the subplot grid.
    error_style
        "band" (shaded SEM) or None to plot only means.
    xlabel
        X-axis label.
    ylabel
        Y-axis label.
    title
        Figure title.
    sharex, sharey
        Whether to share axes across subplots.
    base_height
        Base figure height before row scaling.
    row_height
        Additional height per plotted row.
    showlegend
        Whether to show the legend.
    line_width
        Trace line width.
    opacity
        Trace opacity.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-row timecourse figure.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # (1, n_channels, n_times)
    elif data.ndim != 3:
        raise ValueError(
            "data must be 2D (channels, times) or 3D (trials, channels, times)."
        )

    n_trials, n_channels, n_times = data.shape
    x_values = np.arange(n_times) if times is None else np.asarray(times)
    if len(x_values) != n_times:
        raise ValueError(
            f"Length of times ({len(x_values)}) must match the last "
            f"dimension of data ({n_times})."
        )

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(
            "Length of channel_names must match the second dimension of data."
        )

    # Resolve ROIs
    if rois is None:
        roi_dict = {ch: [ch] for ch in channel_names}
    elif isinstance(rois, Mapping):
        roi_dict = dict(rois)
    else:
        roi_dict = {ch: [ch] for ch in rois}

    # Resolve groups
    if group_labels is None:
        group_labels = np.zeros(n_trials, dtype=int)
        unique_groups = [0]
    else:
        group_labels = np.asarray(group_labels)
        if len(group_labels) != n_trials:
            raise ValueError(
                "Length of group_labels must match the first dimension of data."
            )
        unique_groups = []
        for g in group_labels:
            if g not in unique_groups:
                unique_groups.append(g)

    # Setup grid
    n_plots = len(roi_dict)
    if n_cols is None:
        n_cols = min(n_plots, 4)
    n_cols = min(n_cols, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    subplot_titles = list(roi_dict.keys())

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=sharex,
        shared_yaxes=sharey,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles,
    )

    from ..theme import _COLORBLIND_COLORS

    def color_to_rgba(color_val, alpha):
        try:
            rgb = mcolors.to_rgb(mcolors.cnames.get(str(color_val), str(color_val)))
        except ValueError:
            return f"rgba(100,100,100,{alpha})"
        r, g, b = (int(channel * 255) for channel in rgb[:3])
        return f"rgba({r}, {g}, {b}, {alpha})"

    for idx, (_roi_name, roi_channels) in enumerate(roi_dict.items()):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        ch_idx = [j for j, ch in enumerate(channel_names) if ch in roi_channels]
        if not ch_idx:
            continue

        data_roi = np.nanmean(data[:, ch_idx, :], axis=1)  # shape: (n_trials, n_times)

        for g_idx, grp in enumerate(unique_groups):
            grp_mask = group_labels == grp
            grp_data = data_roi[grp_mask, :]

            if grp_data.shape[0] == 0:
                continue

            mean_erp = np.nanmean(grp_data, axis=0)
            if grp_data.shape[0] > 1 and error_style == "band":
                sem_erp = np.nanstd(grp_data, axis=0) / np.sqrt(grp_data.shape[0])
            else:
                sem_erp = None

            display_name = (
                group_name_map.get(grp, str(grp)) if group_name_map else str(grp)
            )

            color = None
            if isinstance(palette, Mapping):
                color = palette.get(grp, palette.get(str(grp)))
            elif isinstance(palette, Sequence):
                color = palette[g_idx % len(palette)]
            if color is None:
                color = _COLORBLIND_COLORS[g_idx % len(_COLORBLIND_COLORS)]

            dash = "solid"
            if linestyle_map:
                dash = linestyle_map.get(grp, linestyle_map.get(str(grp), "solid"))

            show_lg = (idx == 0) and showlegend and len(unique_groups) > 1

            # Draw SEM band first
            if sem_erp is not None:
                rgba_color = color_to_rgba(color, 0.15)
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=mean_erp + sem_erp,
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        legendgroup=str(grp),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=mean_erp - sem_erp,
                        mode="lines",
                        fill="tonexty",
                        fillcolor=rgba_color,
                        line={"width": 0},
                        showlegend=False,
                        legendgroup=str(grp),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            # Draw Mean line
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=mean_erp,
                    mode="lines",
                    name=display_name,
                    legendgroup=str(grp),
                    line={"color": color, "width": line_width, "dash": dash},
                    opacity=opacity,
                    showlegend=show_lg,
                ),
                row=row,
                col=col,
            )

        if col == 1 or not sharey:
            fig.update_yaxes(title_text=ylabel, row=row, col=col)
        if row == n_rows or not sharex:
            fig.update_xaxes(title_text=xlabel, row=row, col=col)

        if add_zero:
            fig.add_hline(
                y=0,
                line_width=1,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                row=row,
                col=col,
            )
            fig.add_vline(
                x=0,
                line_width=1,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                row=row,
                col=col,
            )

    _apply_layout(
        fig,
        title=title,
        height=base_height + row_height * n_rows,
        legend_horizontal=True,
    )
    fig.update_layout(margin={"l": 60, "r": 40, "b": 60, "t": 70})
    return fig
