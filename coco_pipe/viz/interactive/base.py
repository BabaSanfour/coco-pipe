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
    "plot_grouped_bar",
    "plot_heatmap",
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
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Interactive grouped bar chart (``barmode="group"``) from a long DataFrame.

    Parameters
    ----------
    data
        Long-format DataFrame: one row per ``(x, group)`` bar.
    x
        Column name for the category axis (shared across groups).
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
        Optional explicit ordering for the category axis.
    title, xaxis_title, yaxis_title
        Layout labels. Axis titles default to the ``x`` / ``y`` column names.
    legend_title
        Optional legend title (typically the ``group`` column name).
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

    groups = list(data.groupby(group, dropna=False, sort=False))
    colors = _resolve_group_colors([key for key, _ in groups], color_map)

    fig = go.Figure()
    for (key, sub), trace_color in zip(groups, colors, strict=False):
        bar_kwargs: dict[str, Any] = {
            "name": str(key),
            "x": sub[x].tolist(),
            "y": sub[y].tolist(),
            "marker": {"color": trace_color},
        }
        if text is not None:
            bar_kwargs["text"] = [str(t) for t in sub[text].tolist()]
            bar_kwargs["textposition"] = text_position
        fig.add_trace(go.Bar(**bar_kwargs))

    _apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y,
        height=height,
        barmode="group",
    )
    if x_order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=[str(v) for v in x_order])
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


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
