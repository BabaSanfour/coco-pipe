#!/usr/bin/env python3
"""
coco_pipe/viz/plots.py
----------------------
Minimal plotting helpers:

- plot_topomap: 2D topographic map using MNE's plot_topomap for EEG layouts.
- plot_bar:     Generic ranked bar plots with optional error bars.

Both functions are data-agnostic: pass any metric (e.g., accuracy, importance)
for any analysis unit (sensors, features, regions).
"""
from __future__ import annotations

import math
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _coerce_series(
    data: Union[pd.Series, Mapping[str, float], Sequence[float]],
    index: Optional[Sequence[str]] = None,
) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    if isinstance(data, Mapping):
        return pd.Series(data).dropna()
    values = np.asarray(list(data), dtype=float)
    if index is None:
        index = [str(i) for i in range(len(values))]
    return pd.Series(values, index=index).dropna()


def _coerce_coords(
    coords: Union[
        pd.DataFrame, Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]
    ],
    index: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if isinstance(coords, pd.DataFrame):
        if coords.shape[1] < 2:
            raise ValueError("coords DataFrame must have at least two columns (x, y)")
        df = coords.iloc[:, :2].copy()
        df.columns = ["x", "y"]
        return df
    if isinstance(coords, Mapping):
        items = [(k, v[0], v[1]) for k, v in coords.items()]
        return pd.DataFrame(items, columns=["name", "x", "y"]).set_index("name")
    # assume sequence of (x,y)
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("coords must be Nx2 sequence for (x,y) pairs")
    idx = list(index) if index is not None else [str(i) for i in range(arr.shape[0])]
    return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]}, index=idx)


def plot_topomap(
    values: Union[pd.Series, Mapping[str, float], Sequence[float]],
    coords: Union[
        pd.DataFrame, Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]
    ],
    *,
    index: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    head_radius: float = 0.5,  # kept for API compatibility; not used by MNE
    levels: int = 64,  # kept for API compatibility; not used by MNE
    fill: bool = True,  # kept for API compatibility; not used by MNE
    sensors: str = "markers",  # "markers"->True, "labels"->'labels', "none"->False
    sensor_size: float = 30.0,  # kept for API compatibility
    outlines: bool = True,  # True->'head', False->'none'
    contours: int = 0,
    symmetric: bool = True,
    title: Optional[str] = None,
    cbar: bool = True,
    cbar_label: Optional[str] = None,
    text_size: Optional[float] = None,
    title_size: Optional[float] = None,
    title_loc: Optional[str] = None,
    tick_size: Optional[float] = None,
    cbar_size: Optional[float] = None,
    figsize: Tuple[float, float] = (5, 5),
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a topographic map for arbitrary sensor values using MNE.

    Parameters
    ----------
    values : Series | Mapping | Sequence
        Sensor values. If Series or Mapping, keys should match coords names.
    coords : DataFrame | Mapping | Sequence
        Sensor coordinates. Accepts:
        - DataFrame with columns ['x','y'] and index = sensor names
        - Mapping[name] -> (x,y)
        - Sequence[(x,y)], plus `index` to name them
    index : sequence, optional
        Names for values/coords when passing sequences.
    sensors : str
        One of {"markers", "labels", "none"} to show sensor markers/labels.
    contours : int
        Number of contour lines to draw on top (0 disables).
    symmetric : bool, default=True
        If True, use symmetric color limits around 0 (vlim = [-a, +a]) where
        a is a rounded-up amplitude from the data range. Set False to use
        asymmetric nice-rounded limits.

    Notes
    -----
    - Requires the optional dependency 'mne'. Install with `pip install mne`.
    - Uses mne.viz.plot_topomap for robust EEG topographies.
    - vmin/vmax are rounded to "nice" values: we floor vmin and ceil vmax to the
      nearest power-of-10 step (e.g., 0.87 -> 0.9, 0.012 -> 0.02).
    - Use `text_size` to scale all text; override with `title_size`, `tick_size`,
      or `cbar_size` for finer control.
    - Use `title_loc` to set title alignment: 'left', 'center', or 'right'.
    """
    try:
        import mne  # type: ignore
    except Exception as e:
        raise ImportError(
            "plot_topomap requires 'mne'. Install it via 'pip install mne'."
        ) from e

    vals = _coerce_series(values, index=index)
    cdf = _coerce_coords(coords, index=index)
    # Align on intersection of names
    common = vals.index.intersection(cdf.index)
    if common.empty:
        raise ValueError("No overlapping sensor names between values and coords")
    vals = vals.loc[common]
    cdf = cdf.loc[common]

    # Prepare arrays for MNE
    pos = cdf[["x", "y"]].to_numpy(dtype=float)
    data = vals.to_numpy(dtype=float)

    # Determine and round vmin/vmax (even if provided) to nice boundaries
    def _nice_step(x: float) -> float:
        ax = abs(x)
        if ax == 0 or not np.isfinite(ax):
            return 1.0
        e = math.floor(math.log10(ax))
        return 10.0**e

    def _nice_floor(x: float) -> float:
        s = _nice_step(x)
        return math.floor(x / s) * s

    def _nice_ceil(x: float) -> float:
        s = _nice_step(x)
        return math.ceil(x / s) * s

    raw_vmin = float(np.nanmin(data)) if vmin is None else float(vmin)
    raw_vmax = float(np.nanmax(data)) if vmax is None else float(vmax)
    if raw_vmin > raw_vmax:
        raw_vmin, raw_vmax = raw_vmax, raw_vmin
    if symmetric:
        amp = max(abs(raw_vmin), abs(raw_vmax))
        if not np.isfinite(amp) or amp == 0:
            amp = 1.0
        vmax_r = _nice_ceil(amp)
        vmin_r = -vmax_r
    else:
        vmin_r = _nice_floor(raw_vmin)
        vmax_r = _nice_ceil(raw_vmax)
        if not np.isfinite(vmin_r) or not np.isfinite(vmax_r) or vmin_r == vmax_r:
            span = max(abs(raw_vmin), abs(raw_vmax), 1e-12)
            step = _nice_step(span)
            vmin_r = _nice_floor(raw_vmin - step)
            vmax_r = _nice_ceil(raw_vmax + step)

    # Map sensors option
    if sensors == "none":
        sensors_opt: Union[bool, str] = False
    elif sensors == "labels":
        sensors_opt = "labels"
    else:
        sensors_opt = True  # markers

    outlines_opt: Union[str, dict] = "head" if outlines else "none"

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Call MNE's plot_topomap
    im, cn = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        vlim=(vmin_r, vmax_r),
        cmap=cmap,
        sensors=sensors_opt,
        names=list(common) if sensors == "labels" else None,
        contours=contours,
        outlines=outlines_opt,
        show=False,
    )

    # Resolve text sizes
    _title_fs = title_size if title_size is not None else text_size
    _tick_fs = tick_size if tick_size is not None else text_size
    _cbar_fs = cbar_size if cbar_size is not None else text_size
    _title_loc = (
        None
        if title_loc is None
        else (
            "center"
            if str(title_loc).lower() in {"center", "centre"}
            else str(title_loc).lower()
        )
    )

    if title:
        ax.set_title(title, fontsize=_title_fs, loc=_title_loc)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if _cbar_fs is not None:
            cb.ax.tick_params(labelsize=_cbar_fs)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=_cbar_fs)

    if _tick_fs is not None:
        ax.tick_params(axis="both", labelsize=_tick_fs)

    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_bar(
    scores: Union[pd.Series, Mapping[str, float], Sequence[float]],
    *,
    errors: Optional[Union[pd.Series, Mapping[str, float], Sequence[float]]] = None,
    labels: Optional[Sequence[str]] = None,
    label_map: Optional[Mapping[str, str]] = None,
    top_n: Optional[int] = None,
    ascending: bool = False,
    orientation: str = "vertical",  # "vertical" or "horizontal"
    color: Optional[Union[str, Sequence[str]]] = None,
    cmap: Optional[str] = None,
    abs_values: bool = False,
    nice_axis_limits: bool = False,
    axis_break_orders: Optional[float] = None,
    axis_break_pad: float = 0.05,
    remove_spines: Optional[Union[str, Sequence[str]]] = None,
    remove_ticks: Optional[Union[str, Sequence[str]]] = None,
    text_size: Optional[float] = None,
    title_size: Optional[float] = None,
    title_loc: Optional[str] = None,
    axis_label_size: Optional[float] = None,
    tick_size: Optional[float] = None,
    grid_axis: Optional[str] = "y",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    axis_lim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (7, 4),
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a ranked bar chart of scores with optional error bars.

    Parameters
    ----------
    scores : Series | Mapping | Sequence
        Values to plot. If Sequence, provide `labels`.
    errors : Series | Mapping | Sequence, optional
        Error bars (std/sem). Keys or order must match `scores`.
    label_map : Mapping[str, str], optional
        Mapping from original item names (index of `scores`) to custom display labels
        for tick names. Unmapped labels fall back to their original names.
    top_n : int, optional
        Show only the top-N items after sorting by value.
    ascending : bool
        Sort ascending (False for descending/top highest).
    orientation : str
        "vertical" or "horizontal".
    axis_lim : tuple(float, float), optional
        Limits for the numeric axis (y for vertical, x for horizontal). If provided,
        applies via `ax.set_ylim(axis_lim)` (vertical) or `ax.set_xlim(axis_lim)` (horizontal).
    color : str | list, optional
        Single color or list per bar. If not provided and `cmap` is set, colors
        are mapped from values.
    cmap : str, optional
        Colormap name to map bar colors by value.
    abs_values : bool, default=False
        If True, plot absolute values of `scores` and also sort/select top_n by
        absolute magnitude.
    nice_axis_limits : bool, default=False
        If True and `axis_lim` is not provided, set the numeric axis limits to
        "nice-rounded" floor(min(values)) and ceil(max(values)). Applies to the
        y-axis for vertical orientation and x-axis for horizontal orientation.
    remove_spines : str | sequence, optional
        Remove axis spines by name. Accepts any of {"left","right","top","bottom"},
        or the shortcut "right_top" to remove both right and top. You can also
        pass a list like ["right", "bottom"].
    remove_ticks : str | sequence, optional
        Hide tick marks (not labels) on the x and/or y axes. Accepts "x", "y",
        or "both". You can combine as a space/comma-separated string (e.g.,
        "x y" or "x,y") or pass a list like ["x", "y"].
    text_size : float, optional
        Base font size for all text in the plot (title, axis labels, ticks).
    title_size : float, optional
        Overrides title font size; falls back to `text_size`.
    axis_label_size : float, optional
        Overrides axis label font size; falls back to `text_size`.
    tick_size : float, optional
        Overrides tick label font size; falls back to `text_size`.
    title_loc : str, optional
        Title alignment: one of {"left", "center", "right"}. Defaults to matplotlib's
        default (center) if not provided.
    grid_axis : str | None, default="y"
        Which axis to draw grid lines on: one of {"x", "y", "both", "auto"}.
        Use "none" or None to disable the grid. "auto" chooses 'y' for vertical
        bars and 'x' for horizontal bars.
    axis_break_orders : float, optional
        If provided, automatically create a broken numeric axis (y for vertical,
        x for horizontal) when the largest value is at least 10**axis_break_orders
        times the second largest. Only applied when values are non-negative and
        `axis_lim` is not set. Keeps API compatible by returning the primary
        axis (bottom/left) of the broken pair.
    axis_break_pad : float, default=0.05
        Fractional padding around the split limits for the broken axes.
    """
    s = _coerce_series(scores, index=labels)
    # Sort and optionally take top-N. When abs_values=True, order by |scores|.
    if abs_values:
        order_idx = s.abs().sort_values(ascending=ascending).index
        s = s.loc[order_idx]
    else:
        s = s.sort_values(ascending=ascending)
    if top_n is not None and top_n > 0:
        s = s.head(top_n)

    if errors is not None:
        e = _coerce_series(errors, index=s.index)
        e = e.reindex(s.index)
    else:
        e = None

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        created_fig = True
    else:
        fig = ax.figure

    labels = s.index.tolist()
    if label_map:
        disp_labels = [label_map.get(lbl, lbl) for lbl in labels]
    else:
        disp_labels = labels
    # Optionally transform values for plotting
    s_plot = s.abs() if abs_values else s
    vals = s_plot.values.astype(float)

    if cmap is not None and color is None:
        cmap_obj = plt.get_cmap(cmap)
        # normalize by min/max
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax == vmin:
            colors = [cmap_obj(0.5)] * len(vals)
        else:
            colors = [cmap_obj((v - vmin) / (vmax - vmin)) for v in vals]
    else:
        if isinstance(color, (list, tuple, np.ndarray)):
            colors = list(color)
        else:
            colors = color

    # helper to compute nice-rounded bounds
    def _nice_step(x: float) -> float:
        ax = abs(x)
        if ax == 0 or not np.isfinite(ax):
            return 1.0
        e = math.floor(math.log10(ax))
        return 10.0**e

    def _nice_floor(x: float) -> float:
        s = _nice_step(x)
        return math.floor(x / s) * s

    def _nice_ceil(x: float) -> float:
        s = _nice_step(x)
        return math.ceil(x / s) * s

    # Potential broken-axis branch (applies only when explicit axis limits are not set)
    # Conditions: axis_break_orders set, at least two values, all non-negative after optional abs,
    # and axis_lim is None. Uses simple two-panel broken axis with diagonal marks.
    do_axis_break = False
    if (
        axis_break_orders is not None
        and axis_lim is None
        and len(vals) >= 2
        and np.all(np.isfinite(vals))
        and np.all(vals >= 0)
    ):
        # Consider errors in determining extremes if provided
        errs = e.values if e is not None else np.zeros_like(vals)
        vals_with_err = vals + np.maximum(0.0, errs)
        # Find top-1 and top-2
        order = np.argsort(vals_with_err)[::-1]
        vmax1 = float(vals_with_err[order[0]])
        vmax2 = float(vals_with_err[order[1]]) if len(order) > 1 else 0.0
        ratio = (vmax1 / max(vmax2, 1e-12)) if vmax1 > 0 else 1.0
        threshold = 10.0 ** float(axis_break_orders)
        if ratio >= threshold and np.isfinite(ratio):
            do_axis_break = True

    if do_axis_break:
        # Compute split limits
        errs = e.values if e is not None else np.zeros_like(vals)
        vals_with_err = vals + np.maximum(0.0, errs)
        order = np.argsort(vals_with_err)[::-1]
        vmax1 = float(vals_with_err[order[0]])
        vmax2 = float(vals_with_err[order[1]]) if len(order) > 1 else 0.0
        # Lower panel upper bound based on second max (or fraction of top if others are 0)
        lower_upper = (
            vmax2 if vmax2 > 0 else (vmax1 / (10.0 ** float(axis_break_orders)))
        )
        lower_upper = max(lower_upper, 0.0) * (1.0 + float(axis_break_pad))
        upper_lower = lower_upper  # join point
        upper_upper = vmax1 * (1.0 + float(axis_break_pad))

        # Create two axes either stacked (vertical) or side-by-side (horizontal)
        if orientation == "horizontal":
            # Side-by-side axes sharing y
            if created_fig and ax is not None and len(fig.axes) == 1:
                plt.close(
                    fig
                )  # close single-axes fig before creating new layout to avoid duplicates
                fig, (ax_left, ax_right) = plt.subplots(
                    1,
                    2,
                    sharey=True,
                    figsize=figsize,
                    constrained_layout=True,
                    gridspec_kw={"width_ratios": [3, 1]},
                )
            else:
                # Replace provided ax with two new axes in the same position
                bbox = ax.get_position()
                ax.set_visible(False)
                w_total = bbox.width
                gap = 0.02 * w_total
                w_left = w_total * 0.72
                w_right = w_total * 0.28
                ax_left = fig.add_axes(
                    [bbox.x0, bbox.y0, w_left - gap / 2, bbox.height]
                )
                ax_right = fig.add_axes(
                    [
                        bbox.x0 + w_left + gap / 2,
                        bbox.y0,
                        w_right - gap / 2,
                        bbox.height,
                    ]
                )
                # share y manually
                ax_right.get_shared_y_axes().join(ax_right, ax_left)

            # Draw bars; left shows all, right shows only the outlier (largest)
            y_pos = np.arange(len(labels))
            idx_out = int(order[0])
            mask = np.zeros(len(vals), dtype=bool)
            mask[idx_out] = True

            # color handling for subset
            def _subset_colors(mask_bool):
                if isinstance(colors, (list, tuple, np.ndarray)):
                    return [colors[i] for i, m in enumerate(mask_bool) if m]
                return colors

            ax_left.barh(
                y_pos,
                vals,
                xerr=e.values if e is not None else None,
                color=colors,
            )
            ax_right.barh(
                y_pos[mask],
                vals[mask],
                xerr=(e.values[mask] if e is not None else None),
                color=_subset_colors(mask),
            )
            # Ticks/labels
            ax_left.set_yticks(y_pos, labels=disp_labels)
            ax_right.tick_params(axis="y", labelleft=False)
            # Limits per panel
            ax_left.set_xlim((0.0, lower_upper))
            ax_right.set_xlim((upper_lower, upper_upper))
            # Diagonal break marks (only on primary/left axis)
            d = 0.015
            kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False)
            ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            # No start lines on the secondary/right axis

            # Titles/labels and text sizes
            _title_fs = title_size if title_size is not None else text_size
            _label_fs = axis_label_size if axis_label_size is not None else text_size
            _tick_fs = tick_size if tick_size is not None else text_size
            if title:
                _title_loc = (
                    None
                    if title_loc is None
                    else (
                        "center"
                        if str(title_loc).lower() in {"center", "centre"}
                        else str(title_loc).lower()
                    )
                )
                ax_left.set_title(title, fontsize=_title_fs, loc=_title_loc)
            if xlabel:
                ax_left.set_xlabel(xlabel, fontsize=_label_fs)
            if ylabel:
                ax_left.set_ylabel(ylabel, fontsize=_label_fs)
            if _tick_fs is not None:
                ax_left.tick_params(axis="both", labelsize=_tick_fs)
                ax_right.tick_params(axis="both", labelsize=_tick_fs)

            # Grid configuration
            axis_opt = None if grid_axis is None else str(grid_axis).lower()
            if axis_opt not in {None, "none", "off", "false"}:
                if axis_opt in {"xy", "yx", "all"}:
                    axis_val = "both"
                elif axis_opt == "auto":
                    axis_val = "x"  # numeric axis for horizontal bars
                elif axis_opt in {"x", "y", "both"}:
                    axis_val = axis_opt
                else:
                    axis_val = "x"
                ax_left.grid(True, axis=axis_val, linestyle="--", alpha=0.3)
                ax_right.grid(True, axis=axis_val, linestyle="--", alpha=0.3)

            # Apply spines/ticks removal to both axes if requested
            def _apply_styling_axes(ax_list: List[plt.Axes]):
                # Spines
                if remove_spines is not None:
                    if isinstance(remove_spines, str):
                        opts = [t for t in re.split(r"[\s,]+", remove_spines) if t]
                    else:
                        opts = list(remove_spines)
                    spines_to_remove = set()
                    for opt in opts:
                        if opt in ("right_top", "top_right"):
                            spines_to_remove.update(["right", "top"])
                        elif opt == "all":
                            spines_to_remove.update(["left", "right", "top", "bottom"])
                        else:
                            spines_to_remove.add(opt)
                    for axx in ax_list:
                        for sp in ("left", "right", "top", "bottom"):
                            if sp in spines_to_remove:
                                axx.spines[sp].set_visible(False)
                # Ticks
                if remove_ticks is not None:
                    if isinstance(remove_ticks, str):
                        toks = [t for t in re.split(r"[\s,]+", remove_ticks) if t]
                    else:
                        toks = list(remove_ticks)
                    axes_rm = set()
                    for t in toks:
                        tl = str(t).lower()
                        if tl in ("both", "xy", "yx"):
                            axes_rm.update(["x", "y"])
                        elif tl in ("x", "y"):
                            axes_rm.add(tl)
                    if "x" in axes_rm:
                        ax_left.tick_params(axis="x", which="both", length=0, width=0)
                        ax_right.tick_params(axis="x", which="both", length=0, width=0)
                    if "y" in axes_rm:
                        ax_left.tick_params(axis="y", which="both", length=0, width=0)
                        ax_right.tick_params(axis="y", which="both", length=0, width=0)

            _apply_styling_axes([ax_left, ax_right])

            if save:
                fig.savefig(save, dpi=150)
            return fig, ax_left

        else:
            # Vertical bars: stacked axes sharing x
            if created_fig and ax is not None and len(fig.axes) == 1:
                plt.close(fig)  # close single-axes fig
                fig, (ax_top, ax_bottom) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=figsize,
                    constrained_layout=True,
                    gridspec_kw={"height_ratios": [1, 3]},
                )
            else:
                bbox = ax.get_position()
                ax.set_visible(False)
                h_total = bbox.height
                gap = 0.02 * h_total
                h_top = h_total * 0.35
                h_bottom = h_total * 0.65
                ax_top = fig.add_axes(
                    [bbox.x0, bbox.y0 + h_bottom + gap / 2, bbox.width, h_top - gap / 2]
                )
                ax_bottom = fig.add_axes(
                    [bbox.x0, bbox.y0, bbox.width, h_bottom - gap / 2]
                )
                # share x manually
                ax_top.get_shared_x_axes().join(ax_top, ax_bottom)

            x_pos = np.arange(len(labels))
            idx_out = int(order[0])
            mask = np.zeros(len(vals), dtype=bool)
            mask[idx_out] = True

            def _subset_colors(mask_bool):
                if isinstance(colors, (list, tuple, np.ndarray)):
                    return [colors[i] for i, m in enumerate(mask_bool) if m]
                return colors

            # Top shows only outlier; bottom shows all
            ax_top.bar(
                x_pos[mask],
                vals[mask],
                yerr=(e.values[mask] if e is not None else None),
                color=_subset_colors(mask),
            )
            ax_bottom.bar(
                x_pos,
                vals,
                yerr=e.values if e is not None else None,
                color=colors,
            )
            # Tick labels only on bottom
            ax_bottom.set_xticks(x_pos, labels=disp_labels, rotation=45, ha="right")
            ax_top.tick_params(axis="x", labelbottom=False)
            # Limits per panel
            ax_bottom.set_ylim((0.0, lower_upper))
            ax_top.set_ylim((upper_lower, upper_upper))
            # Diagonal marks (only on primary/top axis)
            d = 0.015
            kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            # No start lines on the secondary/bottom axis

            # Titles/labels and text sizes
            _title_fs = title_size if title_size is not None else text_size
            _label_fs = axis_label_size if axis_label_size is not None else text_size
            _tick_fs = tick_size if tick_size is not None else text_size
            if title:
                _title_loc = (
                    None
                    if title_loc is None
                    else (
                        "center"
                        if str(title_loc).lower() in {"center", "centre"}
                        else str(title_loc).lower()
                    )
                )
                ax_top.set_title(title, fontsize=_title_fs, loc=_title_loc)
            if xlabel:
                ax_bottom.set_xlabel(xlabel, fontsize=_label_fs)
            if ylabel:
                ax_bottom.set_ylabel(ylabel, fontsize=_label_fs)
            if _tick_fs is not None:
                ax_top.tick_params(axis="both", labelsize=_tick_fs)
                ax_bottom.tick_params(axis="both", labelsize=_tick_fs)

            # Grid configuration
            axis_opt = None if grid_axis is None else str(grid_axis).lower()
            if axis_opt not in {None, "none", "off", "false"}:
                if axis_opt in {"xy", "yx", "all"}:
                    axis_val = "both"
                elif axis_opt == "auto":
                    axis_val = "y"  # numeric axis for vertical bars
                elif axis_opt in {"x", "y", "both"}:
                    axis_val = axis_opt
                else:
                    axis_val = "y"
                ax_top.grid(True, axis=axis_val, linestyle="--", alpha=0.3)
                ax_bottom.grid(True, axis=axis_val, linestyle="--", alpha=0.3)

            # Apply spines/ticks removal to both axes if requested
            def _apply_styling_axes(ax_list: List[plt.Axes]):
                if remove_spines is not None:
                    if isinstance(remove_spines, str):
                        opts = [t for t in re.split(r"[\s,]+", remove_spines) if t]
                    else:
                        opts = list(remove_spines)
                    spines_to_remove = set()
                    for opt in opts:
                        if opt in ("right_top", "top_right"):
                            spines_to_remove.update(["right", "top"])
                        elif opt == "all":
                            spines_to_remove.update(["left", "right", "top", "bottom"])
                        else:
                            spines_to_remove.add(opt)
                    for axx in ax_list:
                        for sp in ("left", "right", "top", "bottom"):
                            if sp in spines_to_remove:
                                axx.spines[sp].set_visible(False)
                # Ticks removal
                if remove_ticks is not None:
                    if isinstance(remove_ticks, str):
                        toks = [t for t in re.split(r"[\s,]+", remove_ticks) if t]
                    else:
                        toks = list(remove_ticks)
                    axes_rm = set()
                    for t in toks:
                        tl = str(t).lower()
                        if tl in ("both", "xy", "yx"):
                            axes_rm.update(["x", "y"])
                        elif tl in ("x", "y"):
                            axes_rm.add(tl)
                    if "x" in axes_rm:
                        ax_top.tick_params(axis="x", which="both", length=0, width=0)
                        ax_bottom.tick_params(axis="x", which="both", length=0, width=0)
                    if "y" in axes_rm:
                        ax_top.tick_params(axis="y", which="both", length=0, width=0)
                        ax_bottom.tick_params(axis="y", which="both", length=0, width=0)

            _apply_styling_axes([ax_top, ax_bottom])

            if save:
                fig.savefig(save, dpi=150)
            return fig, ax_bottom

    # Standard single-axis branch
    if orientation == "horizontal":
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, xerr=e.values if e is not None else None, color=colors)
        ax.set_yticks(y_pos, labels=disp_labels)
        # Resolve sizes
        _title_fs = title_size if title_size is not None else text_size
        _label_fs = axis_label_size if axis_label_size is not None else text_size
        _tick_fs = tick_size if tick_size is not None else text_size

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=_label_fs)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=_label_fs)
        if axis_lim is not None:
            ax.set_xlim(axis_lim)
        elif nice_axis_limits:
            if e is not None:
                vmin = float(np.nanmin(vals - e.values)) if len(vals) else 0.0
                vmax = float(np.nanmax(vals + e.values)) if len(vals) else 1.0
            else:
                vmin = float(np.nanmin(vals)) if len(vals) else 0.0
                vmax = float(np.nanmax(vals)) if len(vals) else 1.0
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            ax.set_xlim((_nice_floor(vmin), _nice_ceil(vmax)))
    else:
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, vals, yerr=e.values if e is not None else None, color=colors)
        ax.set_xticks(x_pos, labels=disp_labels, rotation=45, ha="right")
        # Resolve sizes
        _title_fs = title_size if title_size is not None else text_size
        _label_fs = axis_label_size if axis_label_size is not None else text_size
        _tick_fs = tick_size if tick_size is not None else text_size

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=_label_fs)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=_label_fs)
        if axis_lim is not None:
            ax.set_ylim(axis_lim)
        elif nice_axis_limits:
            if e is not None:
                vmin = float(np.nanmin(vals - e.values)) if len(vals) else 0.0
                vmax = float(np.nanmax(vals + e.values)) if len(vals) else 1.0
            else:
                vmin = float(np.nanmin(vals)) if len(vals) else 0.0
                vmax = float(np.nanmax(vals)) if len(vals) else 1.0
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            ax.set_ylim((_nice_floor(vmin), _nice_ceil(vmax)))

    if title:
        _title_fs = title_size if title_size is not None else text_size
        _title_loc = (
            None
            if title_loc is None
            else (
                "center"
                if str(title_loc).lower() in {"center", "centre"}
                else str(title_loc).lower()
            )
        )
        ax.set_title(title, fontsize=_title_fs, loc=_title_loc)

    # Grid configuration
    axis_opt = None if grid_axis is None else str(grid_axis).lower()
    if axis_opt not in {None, "none", "off", "false"}:
        if axis_opt in {"xy", "yx", "all"}:
            axis_val = "both"
        elif axis_opt == "auto":
            axis_val = "x" if orientation == "horizontal" else "y"
        elif axis_opt in {"x", "y", "both"}:
            axis_val = axis_opt
        else:
            axis_val = "y"  # fallback to previous default
        ax.grid(True, axis=axis_val, linestyle="--", alpha=0.3)

    # Apply tick label size if requested
    _tick_fs = tick_size if tick_size is not None else text_size
    if _tick_fs is not None:
        ax.tick_params(axis="both", labelsize=_tick_fs)

    # Optionally remove specific spines
    if remove_spines is not None:
        if isinstance(remove_spines, str):
            # split on commas/whitespace so inputs like "right top" or "right,top" work
            opts = [t for t in re.split(r"[\s,]+", remove_spines) if t]
        else:
            opts = list(remove_spines)
        spines_to_remove = set()
        for opt in opts:
            if opt in ("right_top", "top_right"):
                spines_to_remove.update(["right", "top"])
            elif opt == "all":
                spines_to_remove.update(["left", "right", "top", "bottom"])
            else:
                spines_to_remove.add(opt)
        for sp in ("left", "right", "top", "bottom"):
            if sp in spines_to_remove:
                ax.spines[sp].set_visible(False)

    # Optionally remove ticks on x and/or y axes
    if remove_ticks is not None:
        if isinstance(remove_ticks, str):
            toks = [t for t in re.split(r"[\s,]+", remove_ticks) if t]
        else:
            toks = list(remove_ticks)
        axes_rm = set()
        for t in toks:
            tl = str(t).lower()
            if tl in ("both", "xy", "yx"):
                axes_rm.update(["x", "y"])
            elif tl in ("x", "y"):
                axes_rm.add(tl)
        if "x" in axes_rm:
            # Set tick mark length/width to 0 to hide marks but keep labels
            ax.tick_params(axis="x", which="both", length=0, width=0)
        if "y" in axes_rm:
            ax.tick_params(axis="y", which="both", length=0, width=0)

    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def plot_scatter2d(
    x: Union[pd.Series, Sequence[float], np.ndarray],
    y: Union[pd.Series, Sequence[float], np.ndarray],
    *,
    labels: Optional[Union[pd.Series, Sequence[Any], np.ndarray]] = None,
    label_map: Optional[Mapping[Any, str]] = None,
    palette: Optional[Sequence[str]] = None,
    alpha: float = 0.8,
    s: float = 25.0,
    text_size: Optional[float] = None,
    title_size: Optional[float] = None,
    title_loc: Optional[str] = None,
    axis_label_size: Optional[float] = None,
    tick_size: Optional[float] = None,
    legend_size: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    figsize: Tuple[float, float] = (5, 5),
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot with class-based coloring.

    Parameters
    ----------
    x, y : array-like
        Coordinates for points.
    labels : array-like, optional
        Class labels for coloring. If None, a single color is used.
    label_map : mapping, optional
        Mapping from raw labels to display names in legend.
    palette : sequence of colors, optional
        Colors to cycle over classes (defaults to matplotlib tab10).
    legend : bool
        Whether to show legend for classes.
    """
    x_arr = x.values if isinstance(x, pd.Series) else np.asarray(x)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if labels is None:
        ax.scatter(x_arr, y_arr, s=s, alpha=alpha)
    else:
        lab_arr = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)
        uniq = pd.unique(lab_arr)
        # Colors
        if palette is None:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(uniq))]
        else:
            colors = list(palette)
            if len(colors) < len(uniq):
                # repeat if needed
                k = int(np.ceil(len(uniq) / len(colors)))
                colors = (colors * k)[: len(uniq)]
        for c, u in zip(colors, uniq):
            mask = lab_arr == u
            ax.scatter(
                x_arr[mask],
                y_arr[mask],
                s=s,
                alpha=alpha,
                label=label_map.get(u, u) if label_map else u,
                color=c,
            )

    # Resolve sizes
    _title_fs = title_size if title_size is not None else text_size
    _label_fs = axis_label_size if axis_label_size is not None else text_size
    _tick_fs = tick_size if tick_size is not None else text_size
    _legend_fs = legend_size if legend_size is not None else text_size

    if title:
        _title_loc = (
            None
            if title_loc is None
            else (
                "center"
                if str(title_loc).lower() in {"center", "centre"}
                else str(title_loc).lower()
            )
        )
        ax.set_title(title, fontsize=_title_fs, loc=_title_loc)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_label_fs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_label_fs)
    if legend and labels is not None:
        ax.legend(frameon=False, fontsize=_legend_fs)

    if _tick_fs is not None:
        ax.tick_params(axis="both", labelsize=_tick_fs)

    if save:
        fig.savefig(save, dpi=150)
    return fig, ax
