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

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _coerce_series(data: Union[pd.Series, Mapping[str, float], Sequence[float]],
                   index: Optional[Sequence[str]] = None) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    if isinstance(data, Mapping):
        return pd.Series(data).dropna()
    values = np.asarray(list(data), dtype=float)
    if index is None:
        index = [str(i) for i in range(len(values))]
    return pd.Series(values, index=index).dropna()


def _coerce_coords(coords: Union[pd.DataFrame, Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]],
                   index: Optional[Sequence[str]] = None) -> pd.DataFrame:
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
    coords: Union[pd.DataFrame, Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]],
    *,
    index: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    head_radius: float = 0.5,  # kept for API compatibility; not used by MNE
    levels: int = 64,          # kept for API compatibility; not used by MNE
    fill: bool = True,         # kept for API compatibility; not used by MNE
    sensors: str = "markers",  # "markers"->True, "labels"->'labels', "none"->False
    sensor_size: float = 30.0, # kept for API compatibility
    outlines: bool = True,     # True->'head', False->'none'
    contours: int = 0,
    symmetric: bool = True,
    title: Optional[str] = None,
    cbar: bool = True,
    cbar_label: Optional[str] = None,
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
        return 10.0 ** e

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
        sensors_opt = 'labels'
    else:
        sensors_opt = True  # markers

    outlines_opt: Union[str, dict] = 'head' if outlines else 'none'

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
        names=list(common) if sensors == 'labels' else None,
        contours=contours,
        outlines=outlines_opt,
        show=False,
    )

    if title:
        ax.set_title(title)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)

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
    """
    s = _coerce_series(scores, index=labels)
    if top_n is not None and top_n > 0:
        s = s.sort_values(ascending=ascending).head(top_n)
    else:
        s = s.sort_values(ascending=ascending)

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
    vals = s.values.astype(float)

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

    if orientation == "horizontal":
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, xerr=e.values if e is not None else None, color=colors)
        ax.set_yticks(y_pos, labels=disp_labels)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if axis_lim is not None:
            ax.set_xlim(axis_lim)
    else:
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, vals, yerr=e.values if e is not None else None, color=colors)
        ax.set_xticks(x_pos, labels=disp_labels, rotation=45, ha='right')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if axis_lim is not None:
            ax.set_ylim(axis_lim)

    if title:
        ax.set_title(title)

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
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

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        created_fig = True
    else:
        fig = ax.figure

    if labels is None:
        ax.scatter(x_arr, y_arr, s=s, alpha=alpha)
    else:
        lab_arr = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)
        uniq = pd.unique(lab_arr)
        # Colors
        if palette is None:
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(uniq))]
        else:
            colors = list(palette)
            if len(colors) < len(uniq):
                # repeat if needed
                k = int(np.ceil(len(uniq) / len(colors)))
                colors = (colors * k)[:len(uniq)]
        for c, u in zip(colors, uniq):
            mask = (lab_arr == u)
            ax.scatter(x_arr[mask], y_arr[mask], s=s, alpha=alpha, label=label_map.get(u, u) if label_map else u, color=c)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and labels is not None:
        ax.legend(frameon=False)

    if save:
        fig.savefig(save, dpi=150)
    return fig, ax
