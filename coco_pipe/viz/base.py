from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

from ._utils import _coerce_series, coerce_sensor_layout, finalize_axes
from .theme import DIVERGING, QUALITATIVE, SEQUENTIAL, coco_theme, figure_size


def _colored_line_collection(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: str,
    linewidth: float = 2.5,
    linestyle: str = "-",
    norm: plt.Normalize | None = None,
) -> LineCollection:
    """Build a 2D line collection whose segment color encodes values."""
    n = min(len(x), len(y), len(values))
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    values = np.asarray(values[:n], dtype=float)
    if n < 2:
        raise ValueError("Need at least 2 points to build a LineCollection.")

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)
    lc.set_array(values[:-1])
    lc.set_linewidth(linewidth)
    lc.set_linestyle(linestyle)
    lc.set_joinstyle("round")
    lc.set_capstyle("round")
    return lc


def _plot_alpha_encoded_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    base_color: Any,
    label: str | None = None,
    linewidth: float = 3.2,
    linestyle: str = "-",
    alpha_min: float = 0.20,
    alpha_max: float = 1.00,
) -> None:
    """Draw a 2D line whose segment alpha and lightness encode values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(x) < 2:
        return

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    values = values[: len(segments)]

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        normalized = np.zeros_like(values)
    else:
        vmin, vmax = np.nanpercentile(finite, [5, 95])
        normalized = (
            np.zeros_like(values)
            if np.isclose(vmin, vmax)
            else np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        )

    try:
        base_rgb = np.array(
            mcolors.to_rgb(mcolors.cnames.get(str(base_color), str(base_color)))
        )
    except ValueError:
        base_rgb = np.array([0.2, 0.2, 0.2])

    white = np.ones(3)
    segment_colors = []
    for value in normalized:
        value = 0.0 if not np.isfinite(value) else value
        rgb = base_rgb + (white - base_rgb) * 0.75 * (1.0 - value)
        alpha = alpha_min + (alpha_max - alpha_min) * value
        segment_colors.append((*rgb, alpha))

    lc = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=linewidth,
        linestyles=linestyle,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(lc)
    if label is not None:
        ax.plot([], [], color=base_color, linewidth=linewidth, label=label)


def plot_bar(
    scores: pd.Series | Mapping[str, float] | Sequence[float],
    errors: pd.Series | Mapping[str, float] | Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    label_map: Mapping[str, str] | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    sort: bool = True,
    orientation: str = "vertical",
    color: str | Sequence[str] | None = None,
    cmap: str | None = None,
    abs_values: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a ranked bar chart with optional error bars.

    Parameters
    ----------
    scores
        Values to plot. Accepts a Series, ``{label: value}`` mapping, or a
        plain sequence.
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
        ``"vertical"`` or ``"horizontal"``.
    color
        Bar color or per-bar color sequence. Ignored when ``cmap`` is set.
    cmap
        Colormap name used to derive per-bar colors from values.
    abs_values
        Plot and sort by absolute values while preserving the sign for errors.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    s = _coerce_series(scores, index=labels)
    if s.empty:
        raise ValueError("scores must contain at least one non-null value.")
    if sort:
        order_key = s.abs() if abs_values else s
        s = s.loc[order_key.sort_values(ascending=ascending).index]
    if top_n is not None and top_n > 0:
        s = s.head(top_n)
    e = (
        _coerce_series(errors, index=s.index).reindex(s.index)
        if errors is not None
        else None
    )
    values = s.abs() if abs_values else s

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or figure_size(columns=2, aspect_ratio=4 / 7),
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    if cmap is not None and color is None:
        cmap_obj = plt.get_cmap(cmap)
        finite = np.asarray(values, dtype=float)
        vmin, vmax = np.nanmin(finite), np.nanmax(finite)
        denom = vmax - vmin if vmax != vmin else 1.0
        color = [cmap_obj((val - vmin) / denom) for val in finite]

    display_labels = [
        label_map.get(str(label), str(label)) if label_map else str(label)
        for label in values.index
    ]
    positions = np.arange(len(values))
    if orientation == "horizontal":
        ax.barh(
            positions,
            values.to_numpy(dtype=float),
            xerr=None if e is None else e,
            color=color,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(display_labels)
    elif orientation == "vertical":
        ax.bar(
            positions,
            values.to_numpy(dtype=float),
            yerr=None if e is None else e,
            color=color,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=45, ha="right")
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    finalize_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_heatmap(
    matrix: pd.DataFrame | Sequence[Sequence[float]] | np.ndarray,
    x_labels: Sequence[Any] | None = None,
    y_labels: Sequence[Any] | None = None,
    cmap: str = SEQUENTIAL,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    aspect: str = "auto",
    origin: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    annotate: bool = False,
    annotation_format: str = ".3g",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xtick_rotation: float = 45,
    xtick_ha: str = "right",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D numeric matrix as a heatmap.

    Parameters
    ----------
    matrix
        Values to display. A DataFrame provides default axis labels.
    x_labels
        Column tick labels. Overrides DataFrame column names.
    y_labels
        Row tick labels. Overrides DataFrame index.
    cmap
        Colormap name.
    center
        When set, a ``TwoSlopeNorm`` is applied so this value maps to the
        colormap midpoint.
    vmin
        Explicit colormap lower bound.
    vmax
        Explicit colormap upper bound.
    aspect
        Passed directly to ``imshow``.
    origin
        Passed directly to ``imshow`` when not ``None``.
    colorbar
        Whether to attach a colorbar.
    colorbar_label
        Optional colorbar label.
    annotate
        Overlay each cell with its numeric value.
    annotation_format
        Format string used for cell annotations.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    xtick_rotation
        X-tick label rotation in degrees.
    xtick_ha
        X-tick label horizontal alignment.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy(dtype=float)
        x_labels = list(matrix.columns) if x_labels is None else x_labels
        y_labels = list(matrix.index) if y_labels is None else y_labels
    else:
        values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or figure_size(columns=2, aspect_ratio=5 / 7),
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    kwargs: dict[str, Any] = {"cmap": cmap, "aspect": aspect}
    if origin is not None:
        kwargs["origin"] = origin
    if center is not None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            lower = float(np.nanmin(finite_values)) if vmin is None else float(vmin)
            upper = float(np.nanmax(finite_values)) if vmax is None else float(vmax)
            if lower < center < upper:
                kwargs["norm"] = TwoSlopeNorm(
                    vmin=lower, vcenter=float(center), vmax=upper
                )
            else:
                kwargs["vmin"] = lower
                kwargs["vmax"] = upper
    else:
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
    im = ax.imshow(values, **kwargs)

    if x_labels is not None:
        ax.set_xticks(np.arange(values.shape[1]))
        ax.set_xticklabels([str(value) for value in x_labels], rotation=xtick_rotation)
        for tick in ax.get_xticklabels():
            tick.set_ha(xtick_ha)
    if y_labels is not None:
        ax.set_yticks(np.arange(values.shape[0]))
        ax.set_yticklabels([str(value) for value in y_labels])

    if annotate:
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    format(values[row_idx, col_idx], annotation_format),
                    ha="center",
                    va="center",
                )

    finalize_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xtick_ha=xtick_ha if x_labels is not None else None,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=colorbar_label)
    return fig, ax


def plot_line(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    yerr: pd.Series | Sequence[float] | np.ndarray | None = None,
    error_style: str = "band",
    label: str | None = None,
    color: Any | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    band_alpha: float = 0.15,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D line with optional uncertainty band or error bars.

    Parameters
    ----------
    x
        Horizontal axis values.
    y
        Vertical axis values aligned with ``x``.
    yerr
        Optional symmetric error magnitudes. Rendered as a shaded band or
        error bars depending on ``error_style``.
    error_style
        ``"band"`` for a filled ``fill_between`` region, ``"bar"`` for
        ``errorbar`` caps.
    label
        Legend label.
    color
        Line and band color.
    marker
        Matplotlib marker string.
    linestyle
        Matplotlib linestyle string.
    linewidth
        Line width in points.
    alpha
        Line opacity.
    band_alpha
        Opacity of the uncertainty band when ``error_style="band"``.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    legend
        Whether to show the legend.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or figure_size(columns=2, aspect_ratio=4 / 7),
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    line_kwargs = {
        "label": label,
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "alpha": alpha,
    }
    line_kwargs = {
        key: value for key, value in line_kwargs.items() if value is not None
    }
    ax.plot(x_arr, y_arr, **line_kwargs)

    if yerr is not None:
        err = yerr.values if isinstance(yerr, pd.Series) else np.asarray(yerr)
        if error_style == "band":
            ax.fill_between(
                x_arr, y_arr - err, y_arr + err, alpha=band_alpha, color=color
            )
        elif error_style == "bar":
            ax.errorbar(x_arr, y_arr, yerr=err, fmt="none", color=color, alpha=alpha)
        else:
            raise ValueError("error_style must be 'band' or 'bar'.")

    finalize_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
    return fig, ax


def plot_error_points(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    xerr: Any | None = None,
    yerr: Any | None = None,
    labels: Sequence[Any] | None = None,
    marker: str = "o",
    color: Any | None = None,
    label: str | None = None,
    capsize: float = 3.0,
    reference_x: float | None = None,
    reference_y: float | None = None,
    reference_style: Mapping[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xtick_rotation: float = 45,
    xtick_ha: str = "right",
    legend: bool = False,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot point estimates with optional x/y error bars and references.

    Parameters
    ----------
    x
        Horizontal positions of the points.
    y
        Vertical positions of the points.
    xerr
        Optional horizontal error magnitudes passed to ``errorbar``.
    yerr
        Optional vertical error magnitudes passed to ``errorbar``.
    labels
        Tick labels placed at each ``x`` position.
    marker
        Matplotlib marker string.
    color
        Point and error bar color.
    label
        Legend label for the point series.
    capsize
        Error bar cap size in points.
    reference_x
        Draw a vertical dashed reference line at this x value.
    reference_y
        Draw a horizontal dashed reference line at this y value.
    reference_style
        Optional style overrides for the reference lines.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    xtick_rotation
        X-tick label rotation in degrees.
    xtick_ha
        X-tick label horizontal alignment.
    legend
        Whether to show the legend.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or figure_size(columns=2, aspect_ratio=4 / 7),
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    ref_kwargs = {"color": "0.5", "linestyle": "--", "linewidth": 1}
    if reference_style:
        ref_kwargs.update(reference_style)
    if reference_x is not None:
        ax.axvline(reference_x, **ref_kwargs)
    if reference_y is not None:
        ax.axhline(reference_y, **ref_kwargs)

    ax.errorbar(
        x_arr,
        y_arr,
        xerr=xerr,
        yerr=yerr,
        fmt=marker,
        color=color,
        label=label,
        capsize=capsize,
    )
    if labels is not None:
        ax.set_xticks(x_arr)
        ax.set_xticklabels([str(value) for value in labels], rotation=xtick_rotation)
        for tick in ax.get_xticklabels():
            tick.set_ha(xtick_ha)
    finalize_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        xtick_ha=xtick_ha if labels is not None else None,
    )
    return fig, ax


def plot_distribution_groups(
    groups: Sequence[Sequence[float] | np.ndarray | pd.Series],
    labels: Sequence[Any],
    kind: str = "box",
    show_points: bool = True,
    jitter: float = 0.15,
    point_color: str = "black",
    point_alpha: float = 0.55,
    point_size: float = 18,
    showmeans: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xtick_rotation: float = 45,
    xtick_ha: str = "right",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot grouped scalar distributions as box or violin plots.

    Parameters
    ----------
    groups
        Sequence of data arrays, one per group. Non-finite values are dropped.
    labels
        Group labels aligned with ``groups``.
    kind
        ``"box"`` or ``"violin"``.
    show_points
        Overlay jittered individual points on top of the box or violin.
    jitter
        Half-width of the jitter band around each group position.
    point_color
        Color of the overlaid individual points.
    point_alpha
        Opacity of the overlaid individual points.
    point_size
        Size of the overlaid individual points.
    showmeans
        Whether to show the mean marker inside each box or violin.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    xtick_rotation
        X-tick label rotation in degrees.
    xtick_ha
        X-tick label horizontal alignment.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    values = [np.asarray(group, dtype=float).reshape(-1) for group in groups]
    values = [group[np.isfinite(group)] for group in values]
    if len(values) != len(labels):
        raise ValueError("groups and labels must have the same length.")
    if not values or all(len(group) == 0 for group in values):
        raise ValueError("groups must contain at least one finite value.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize
            or figure_size(columns=2, aspect_ratio=max(4 / 7, len(values) * 0.08)),
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    positions = np.arange(1, len(values) + 1)
    if kind == "box":
        ax.boxplot(values, positions=positions, tick_labels=labels, showmeans=showmeans)
    elif kind == "violin":
        ax.violinplot(values, positions=positions, showmeans=showmeans)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
    else:
        raise ValueError("kind must be 'box' or 'violin'.")

    if show_points:
        for position, group in zip(positions, values):
            if len(group) == 0:
                continue
            x = np.linspace(position - jitter, position + jitter, len(group))
            ax.scatter(
                x,
                group,
                alpha=point_alpha,
                color=point_color,
                zorder=3,
                s=point_size,
            )

    finalize_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xtick_rotation=xtick_rotation,
        xtick_ha=xtick_ha,
    )
    return fig, ax


def plot_scatter2d(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    labels: pd.Series | Sequence[Any] | np.ndarray | None = None,
    label_map: Mapping[Any, str] | None = None,
    palette: Sequence[str] | None = None,
    c: pd.Series | Sequence[float] | np.ndarray | None = None,
    color: Any | None = None,
    cmap: str | None = None,
    colorbar: bool = False,
    colorbar_label: str = "Value",
    xerr: Any | None = None,
    yerr: Any | None = None,
    reference_x: float | None = None,
    reference_y: float | None = None,
    reference_style: Mapping[str, Any] | None = None,
    error_color: str = "0.35",
    capsize: float = 0.0,
    alpha: float = 0.8,
    s: float = 25.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot with optional class coloring.

    Parameters
    ----------
    x
        Horizontal coordinates.
    y
        Vertical coordinates.
    labels
        Discrete class labels used to color points. When provided, ``color``
        and ``c`` are ignored and each class gets its own legend entry.
    label_map
        Optional ``{raw_label: display_label}`` mapping for the legend.
    palette
        Explicit per-class color sequence, overriding the default qualitative
        palette.
    c
        Continuous scalar values used for colormap encoding when ``labels`` is
        ``None``.
    color
        Single color for all points when neither ``labels`` nor ``c`` is set.
    cmap
        Colormap name used with ``c``.
    colorbar
        Whether to attach a colorbar when ``c`` is provided.
    colorbar_label
        Optional colorbar label.
    xerr
        Optional horizontal error magnitudes.
    yerr
        Optional vertical error magnitudes.
    reference_x
        Draw a vertical dashed reference line at this x value.
    reference_y
        Draw a horizontal dashed reference line at this y value.
    reference_style
        Optional style overrides for the reference lines.
    error_color
        Color of the error bars.
    capsize
        Error bar cap size.
    alpha
        Point opacity.
    s
        Point marker size.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    legend
        Whether to show the class legend.
    legend_title
        Optional legend title.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    ref_kwargs = {"color": "0.5", "linestyle": "--", "linewidth": 1}
    if reference_style:
        ref_kwargs.update(reference_style)
    if reference_x is not None:
        ax.axvline(reference_x, **ref_kwargs)
    if reference_y is not None:
        ax.axhline(reference_y, **ref_kwargs)

    if labels is None:
        scatter_kwargs = {"s": s, "alpha": alpha}
        if c is not None:
            scatter_kwargs["c"] = (
                c.values if isinstance(c, pd.Series) else np.asarray(c)
            )
            if cmap is not None:
                scatter_kwargs["cmap"] = cmap
        elif color is not None:
            scatter_kwargs["color"] = color
        scatter = ax.scatter(x_arr, y_arr, **scatter_kwargs)
        if colorbar and c is not None:
            fig.colorbar(scatter, ax=ax, label=colorbar_label)
    else:
        lab_arr = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)
        unique = pd.unique(lab_arr)
        colors = (
            list(palette)
            if palette is not None
            else [plt.get_cmap(QUALITATIVE)(i % 10) for i in range(len(unique))]
        )
        for color, label in zip(colors, unique):
            mask = lab_arr == label
            display = label_map.get(label, label) if label_map else label
            ax.scatter(
                x_arr[mask], y_arr[mask], s=s, alpha=alpha, color=color, label=display
            )
        if legend:
            finalize_axes(ax, legend=True, legend_title=legend_title)

    if xerr is not None or yerr is not None:
        ax.errorbar(
            x_arr,
            y_arr,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=error_color,
            capsize=capsize,
            zorder=2,
        )

    finalize_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_scatter3d(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    z: pd.Series | Sequence[float] | np.ndarray,
    labels: pd.Series | Sequence[Any] | np.ndarray | None = None,
    label_map: Mapping[Any, str] | None = None,
    palette: Sequence[str] | None = None,
    c: pd.Series | Sequence[float] | np.ndarray | None = None,
    color: Any | None = None,
    cmap: str | None = None,
    colorbar: bool = False,
    colorbar_label: str = "Value",
    alpha: float = 0.8,
    s: float = 25.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """3D scatter plot with optional class coloring.

    Parameters
    ----------
    x
        X-axis coordinates.
    y
        Y-axis coordinates.
    z
        Z-axis coordinates.
    labels
        Discrete class labels used to color points. Each class gets its own
        legend entry.
    label_map
        Optional ``{raw_label: display_label}`` mapping for the legend.
    palette
        Explicit per-class color sequence, overriding the default qualitative
        palette.
    c
        Continuous scalar values used for colormap encoding when ``labels`` is
        ``None``.
    color
        Single color for all points when neither ``labels`` nor ``c`` is set.
    cmap
        Colormap name used with ``c``.
    colorbar
        Whether to attach a colorbar when ``c`` is provided.
    colorbar_label
        Optional colorbar label.
    alpha
        Point opacity.
    s
        Point marker size.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    zlabel
        Optional z-axis label.
    legend
        Whether to show the class legend.
    legend_title
        Optional legend title.
    figsize
        Figure size used when creating new axes.
    ax
        Existing 3D Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    z_arr = z.values if isinstance(z, pd.Series) else np.asarray(z)
    if ax is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if labels is None:
        scatter_kwargs = {"s": s, "alpha": alpha}
        if c is not None:
            scatter_kwargs["c"] = (
                c.values if isinstance(c, pd.Series) else np.asarray(c)
            )
            if cmap is not None:
                scatter_kwargs["cmap"] = cmap
        elif color is not None:
            scatter_kwargs["color"] = color
        scatter = ax.scatter(x_arr, y_arr, z_arr, **scatter_kwargs)
        if colorbar and c is not None:
            fig.colorbar(scatter, ax=ax, label=colorbar_label, pad=0.1)
    else:
        lab_arr = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)
        unique = pd.unique(lab_arr)
        colors = (
            list(palette)
            if palette is not None
            else [plt.get_cmap(QUALITATIVE)(i % 10) for i in range(len(unique))]
        )
        for color_value, label in zip(colors, unique):
            mask = lab_arr == label
            display = label_map.get(label, label) if label_map else label
            ax.scatter(
                x_arr[mask],
                y_arr[mask],
                z_arr[mask],
                s=s,
                alpha=alpha,
                color=color_value,
                label=display,
            )
        if legend:
            finalize_axes(ax, legend=True, legend_title=legend_title)

    finalize_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
    )
    return fig, ax


def plot_hexbin(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    gridsize: int = 40,
    cmap: str = SEQUENTIAL,
    mincnt: int | None = 1,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    reference_identity: bool = False,
    reference_style: Mapping[str, Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Hexbin density plot for paired numeric values.

    Parameters
    ----------
    x
        Horizontal values.
    y
        Vertical values aligned with ``x``.
    gridsize
        Number of hexagons in the horizontal direction.
    cmap
        Colormap name encoding bin counts.
    mincnt
        Minimum count per bin required to display a hexagon.
    colorbar
        Whether to attach a colorbar.
    colorbar_label
        Optional colorbar label.
    reference_identity
        Overlay a dashed red identity line (``y = x``).
    reference_style
        Optional style overrides for the identity reference line.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    legend
        Whether to show the legend.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    hb = ax.hexbin(
        x_arr,
        y_arr,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        edgecolors="none",
    )
    if colorbar:
        fig.colorbar(hb, ax=ax, label=colorbar_label)

    if reference_identity:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ref_kwargs = {"color": "red", "linestyle": "--", "alpha": 0.8, "lw": 2}
        if reference_style:
            ref_kwargs.update(reference_style)
        ax.plot(lims, lims, label="Ideal", **ref_kwargs)

    finalize_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
    )
    return fig, ax


def plot_streamfield(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    points: np.ndarray | None = None,
    density: float = 1.5,
    cmap: str = SEQUENTIAL,
    colorbar_label: str | None = None,
    point_color: str = "#DDDDDD",
    point_size: float = 15,
    point_alpha: float = 0.6,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
    **_: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D stream field from gridded vector components.

    Parameters
    ----------
    x_grid
        2D grid of x-coordinates produced by ``np.meshgrid``.
    y_grid
        2D grid of y-coordinates produced by ``np.meshgrid``.
    u_grid
        Horizontal velocity component on the same grid.
    v_grid
        Vertical velocity component on the same grid.
    points
        Optional ``(N, 2)`` array of raw data points scattered beneath the
        streamlines.
    density
        Streamline density passed to ``matplotlib.axes.Axes.streamplot``.
    cmap
        Colormap used to encode local flow speed along streamlines.
    colorbar_label
        Optional colorbar label.
    point_color
        Color for the optional raw data scatter points.
    point_size
        Marker size for raw data scatter points.
    point_alpha
        Opacity of raw data scatter points.
    title
        Optional axes title.
    xlabel
        Optional x-axis label.
    ylabel
        Optional y-axis label.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    if not (
        x_grid.shape == y_grid.shape == u_grid.shape == v_grid.shape
        and x_grid.ndim == 2
    ):
        raise ValueError(
            "x_grid, y_grid, u_grid, and v_grid must be matching 2D arrays."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    if points is not None:
        point_values = np.asarray(points, dtype=float)
        if point_values.ndim != 2 or point_values.shape[1] != 2:
            raise ValueError("points must be an Nx2 array.")
        ax.scatter(
            point_values[:, 0],
            point_values[:, 1],
            c=point_color,
            s=point_size,
            alpha=point_alpha,
        )

    speed = np.sqrt(u_grid**2 + v_grid**2)
    stream = ax.streamplot(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        color=speed,
        cmap=cmap,
        density=density,
    )
    fig.colorbar(stream.lines, ax=ax, label=colorbar_label)
    finalize_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_topomap(
    values: pd.Series | Mapping[str, float] | Sequence[float],
    coords=None,
    index: Sequence[str] | None = None,
    info=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    sensors: str = "markers",
    outlines: bool = True,
    contours: int = 0,
    symmetric: bool = True,
    title: str | None = None,
    cbar: bool = True,
    cbar_label: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
    **_: object,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a topographic map for sensor values using MNE.

    Parameters
    ----------
    values
        Sensor values to plot. A Series or mapping must be keyed by sensor
        name; a plain sequence requires ``index`` or ``coords``/``info`` to
        supply the names.
    coords
        Sensor coordinate source: an MNE-style ``(N, 2+)`` array, a
        ``{name: (x, y)}`` mapping, or a DataFrame with ``x`` and ``y``
        columns and sensor names in the index or a ``FeatureName`` / ``Sensor``
        column.
    index
        Explicit sensor name list used when ``values`` is a plain sequence.
    info
        MNE ``Info`` object from which electrode positions are extracted.
        Takes priority over ``coords``.
    vmin
        Colormap lower bound. Computed symmetrically when ``symmetric=True``.
    vmax
        Colormap upper bound. Computed symmetrically when ``symmetric=True``.
    cmap
        Colormap name. Defaults to the diverging theme map when
        ``symmetric=True``, otherwise the sequential map.
    sensors
        ``"markers"`` to show sensor dots, ``"labels"`` to annotate each
        sensor, or ``"none"`` to hide sensors.
    outlines
        Whether to draw the head outline.
    contours
        Number of contour lines drawn over the topomap.
    symmetric
        Derive ``vmin``/``vmax`` symmetrically around zero.
    title
        Optional axes title.
    cbar
        Whether to attach a colorbar.
    cbar_label
        Optional colorbar label.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.
    """
    try:
        import mne
    except Exception as exc:
        raise ImportError(
            "plot_topomap requires 'mne'. Install the neuroviz extra."
        ) from exc

    vals = _coerce_series(values, index=index)
    layout = coerce_sensor_layout(
        info=info, coords=coords, names=vals.index.astype(str).tolist()
    )
    positions = pd.DataFrame(
        layout.positions[:, :2],
        index=layout.names,
        columns=["x", "y"],
    )
    common = vals.index.astype(str).intersection(positions.index)
    if common.empty:
        raise ValueError("No overlapping sensor names between values and layout.")

    vals = vals.loc[common]
    data = vals.to_numpy(dtype=float)
    pos = positions.loc[common, ["x", "y"]].to_numpy(dtype=float)

    if vmin is None or vmax is None:
        if symmetric:
            amp = float(np.nanmax(np.abs(data))) if len(data) else 1.0
            amp = 1.0 if not np.isfinite(amp) or amp == 0 else amp
            vmin = -amp if vmin is None else vmin
            vmax = amp if vmax is None else vmax
        else:
            vmin = float(np.nanmin(data)) if vmin is None else vmin
            vmax = float(np.nanmax(data)) if vmax is None else vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    sensors_opt: bool | str
    if sensors == "none":
        sensors_opt = False
    elif sensors == "labels":
        sensors_opt = True
    else:
        sensors_opt = True

    im, _ = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        vlim=(vmin, vmax),
        cmap=cmap or (DIVERGING if symmetric else SEQUENTIAL),
        sensors=sensors_opt,
        names=list(common) if sensors == "labels" else None,
        contours=contours,
        outlines="head" if outlines else "none",
        show=False,
    )
    finalize_axes(ax, title=title)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)
    return fig, ax


def plot_timecourses(
    data: np.ndarray | pd.DataFrame,
    times: np.ndarray,
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
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    sharey: bool = True,
    sharex: bool = True,
    add_zero: bool = False,
    axes_kws: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot timecourses (ERPs, ERFs, source activations) across channels or ROIs.

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
        "band" (shaded SEM) or "bar" (errorbars). Passed to `plot_line`.
    xlabel
        X-axis label.
    ylabel
        Y-axis label.
    title
        Figure title.
    figsize
        Custom figure size.
    sharey, sharex
        Whether to share axes across subplots.
    add_zero
        Whether to add reference lines at Time=0 and Amplitude=0.
    axes_kws
        Dictionary of keyword arguments passed to `finalize_axes` (e.g. `grid=False`).
    kwargs
        Additional arguments passed to `plot_line`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and axes array.
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
    if len(times) != n_times:
        raise ValueError("Length of times must match the last dimension of data.")

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

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    if title:
        fig.suptitle(title, fontsize=14)

    with coco_theme():
        for idx, (roi_name, roi_channels) in enumerate(roi_dict.items()):
            ax = axes_flat[idx]
            ch_idx = [j for j, ch in enumerate(channel_names) if ch in roi_channels]

            if not ch_idx:
                finalize_axes(ax, title=f"{roi_name} (No channels)", **(axes_kws or {}))
                continue

            # Average over channels in this ROI
            data_roi = np.nanmean(
                data[:, ch_idx, :], axis=1
            )  # shape: (n_trials, n_times)

            for g_idx, grp in enumerate(unique_groups):
                grp_mask = group_labels == grp
                grp_data = data_roi[grp_mask, :]

                if grp_data.shape[0] == 0:
                    continue

                mean_erp = np.nanmean(grp_data, axis=0)
                if grp_data.shape[0] > 1:
                    sem_erp = np.nanstd(grp_data, axis=0) / np.sqrt(grp_data.shape[0])
                else:
                    sem_erp = None

                # Styling
                color = None
                if isinstance(palette, Mapping):
                    color = palette.get(grp, palette.get(str(grp)))
                elif isinstance(palette, Sequence):
                    color = palette[g_idx % len(palette)]

                linestyle = None
                if linestyle_map:
                    linestyle = linestyle_map.get(
                        grp, linestyle_map.get(str(grp), "solid")
                    )
                    if linestyle == "dash":
                        linestyle = "dashed"

                label = None
                if group_name_map:
                    label = group_name_map.get(
                        grp, group_name_map.get(str(grp), str(grp))
                    )
                else:
                    label = str(grp) if len(unique_groups) > 1 else None

                # Only add legend to the first ax if not None
                if idx > 0 and label is not None:
                    label = "_nolegend_"

                # Use plot_line helper
                plot_line(
                    x=times,
                    y=mean_erp,
                    yerr=sem_erp,
                    error_style=error_style,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    ax=ax,
                    title=None,
                    xlabel=None,
                    ylabel=None,
                    legend=False,
                    **kwargs,
                )

            show_ylabel = (idx % n_cols == 0) or not sharey
            show_xlabel = (idx >= n_plots - n_cols) or not sharex
            finalize_axes(
                ax,
                title=roi_name,
                xlabel=xlabel if show_xlabel else None,
                ylabel=ylabel if show_ylabel else None,
                legend=(idx == 0 and len(unique_groups) > 1),
                **(axes_kws or {}),
            )

            # Add reference lines for time = 0 and amplitude = 0 since the grid is off
            if add_zero:
                ax.axvline(0, color="black", linestyle="--", alpha=0.3, zorder=0)
                ax.axhline(0, color="black", linestyle="--", alpha=0.3, zorder=0)

        # Hide any unused subplots
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)

    return fig, axes_flat


def plot_roi_sensors(
    container: Any,
    rois: Mapping[str, Sequence[str]],
    palette: Sequence[Any] | None = None,
    montage: str | Any = "standard_1005",
    figsize: tuple[float, float] | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot scalp topomaps highlighting specific Regions of Interest (ROIs).

    Parameters
    ----------
    container
        The DataContainer containing the channel coordinates.
    rois
        A dictionary mapping ROI names to lists of channel names.
    palette
        A list of colors to use for highlighting each ROI. If None, uses the
        default colorblind palette.
    montage
        The MNE montage to use for plotting sensors. Can be a string name of a
        standard montage (e.g. 'standard_1005') or an mne.channels.DigMontage object.
    figsize
        Figure size. If None, automatically scaled based on the number of ROIs.
    axes_kws
        Additional keyword arguments passed to `finalize_axes`.

    Returns
    -------
    fig, axes
        The matplotlib Figure and Axes array.
    """
    from itertools import cycle

    import matplotlib.pyplot as plt
    import mne

    ch_names = list(np.asarray(container.coords["channel"]).astype(str))
    info = mne.create_info(ch_names=ch_names, sfreq=160, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage(montage)
        if isinstance(montage, str)
        else montage
    )

    n_rois = len(rois)
    if palette is None:
        palette = [f"C{i % 10}" for i in range(n_rois)]

    fig, axes = plt.subplots(
        1, n_rois, figsize=figsize or (5 * n_rois, 5), squeeze=False
    )
    fig.patch.set_facecolor("white")

    for ax, (region_name, region_chs), color in zip(
        axes.flat, rois.items(), cycle(palette)
    ):
        # Plot the base MNE Topomap head (without names so it's clean)
        mne.viz.plot_sensors(
            info, kind="topomap", show_names=False, axes=ax, show=False
        )

        # Find the MNE scatter object and update its colors to highlight the region
        for collection in ax.collections:
            if len(collection.get_offsets()) == len(ch_names):
                # Default everything to a faint background dot
                face_colors = np.array(["#eeeeee"] * len(ch_names), dtype="object")
                edge_colors = np.array(["white"] * len(ch_names), dtype="object")
                sizes = np.full(len(ch_names), 40)

                # Find the index of the sensors in our ROI
                idx = [ch_names.index(ch) for ch in region_chs if ch in ch_names]

                # Highlight the ROI sensors!
                if idx:
                    face_colors[idx] = color
                    edge_colors[idx] = "black"
                    sizes[idx] = 120

                collection.set_facecolors(face_colors)
                collection.set_edgecolors(edge_colors)
                collection.set_sizes(sizes)
                break

        finalize_axes(ax, title=f"{region_name} ROI", **(axes_kws or {}))

    plt.subplots_adjust(wspace=0.1)
    return fig, axes.flatten()
