"""General-purpose interactive (Plotly) visualization helpers.

Mirrors the static helpers in :mod:`coco_pipe.viz.base` (``plot_bar``,
``plot_distribution_groups``, ``plot_heatmap``) for cases where the report
or notebook wants an interactive figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..theme import _COLORBLIND_COLORS, DIVERGING, SEQUENTIAL
from ._utils import _apply_layout

__all__ = [
    "plot_bar",
    "plot_distribution_groups",
    "plot_heatmap",
]


def _coerce_named_series(
    values: Union[pd.Series, Mapping[str, float], Sequence[float]],
    labels: Optional[Sequence[str]],
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
    scores: Union[pd.Series, Mapping[str, float], Sequence[float]],
    errors: Optional[Union[pd.Series, Mapping[str, float], Sequence[float]]] = None,
    labels: Optional[Sequence[str]] = None,
    label_map: Optional[Mapping[str, str]] = None,
    top_n: Optional[int] = None,
    ascending: bool = False,
    sort: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    color: Optional[Union[str, Sequence[str]]] = None,
    cmap: Optional[str] = None,
    abs_values: bool = False,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: Optional[int] = None,
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

    if abs_values:
        sort_key = series.abs()
    else:
        sort_key = series

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
        marker = dict(
            color=series.values.tolist(),
            colorscale=cmap,
            colorbar=dict(title=yaxis_title or "Value")
            if orientation == "vertical"
            else dict(title=xaxis_title or "Value"),
        )
    else:
        bar_color = color if color is not None else _COLORBLIND_COLORS[0]
        marker = dict(color=bar_color)

    error_kwargs: dict[str, Any] = {}
    if err_series is not None:
        error_kwargs["error_" + ("y" if orientation == "vertical" else "x")] = dict(
            type="data", array=err_series.values.tolist(), visible=True
        )

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


def plot_distribution_groups(
    groups: Sequence[Union[Sequence[float], np.ndarray, pd.Series]],
    labels: Sequence[Any],
    kind: Literal["box", "violin"] = "box",
    show_points: bool = True,
    point_opacity: float = 0.55,
    showmeans: bool = True,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    height: Optional[int] = None,
    sig_pairs: Optional[Sequence[tuple[int, int, str]]] = None,
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
    for idx, (data, label) in enumerate(zip(groups, labels)):
        arr = np.asarray(list(data), dtype=float)
        arr = arr[np.isfinite(arr)]
        trace_color = color_list[idx]
        common = dict(
            y=arr.tolist(),
            name=str(label),
            marker_color=trace_color,
            line_color=trace_color,
            showlegend=False,
        )
        if kind == "box":
            trace = go.Box(
                **common,
                boxmean=showmeans,
                boxpoints="all" if show_points else "outliers",
                jitter=0.3 if show_points else 0.0,
                pointpos=0.0,
                marker=dict(opacity=point_opacity, size=5),
            )
        else:
            trace = go.Violin(
                **common,
                meanline_visible=showmeans,
                points="all" if show_points else False,
                jitter=0.3 if show_points else 0.0,
                marker=dict(opacity=point_opacity, size=5),
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
                line=dict(color="black", width=1.2),
            )
            fig.add_annotation(
                x=x_a,
                y=y,
                xshift=int(0.5 * 30),  # approximate centering
                text=annotation,
                showarrow=False,
                yshift=8,
                font=dict(size=12),
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
    matrix: Union[pd.DataFrame, Sequence[Sequence[float]], np.ndarray],
    x_labels: Optional[Sequence[Any]] = None,
    y_labels: Optional[Sequence[Any]] = None,
    cmap: Optional[str] = None,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = False,
    annotation_format: str = ".3g",
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    height: Optional[int] = None,
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
        "colorbar": dict(title=colorbar_label or ""),
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
