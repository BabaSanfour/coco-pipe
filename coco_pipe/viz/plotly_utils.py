"""
Dim-Reduction Plotly Visualization
==================================

Interactive plotting helpers for explicit dim-reduction embeddings, tidy metric
records, trajectory tensors, and interpretation payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    extract_interpretation_matrix,
    filter_metric_frame,
    infer_metric_plot_type,
    is_categorical,
    prepare_embedding_frame,
    prepare_feature_scores,
    prepare_interpretation_frame,
    prepare_metrics_frame,
)

__all__ = [
    "plot_channel_traces_interactive",
    "plot_embedding_interactive",
    "plot_loss_history_interactive",
    "plot_metric_details",
    "plot_scree_interactive",
    "plot_radar_comparison",
    "plot_raw_preview",
    "plot_shepard_interactive",
    "plot_feature_importance_interactive",
    "plot_feature_correlation_heatmap_interactive",
    "plot_interpretation_interactive",
    "plot_streamlines_interactive",
    "plot_trajectory_metric_series_interactive",
    "plot_trajectory_interactive",
]


def plot_channel_traces_interactive(
    data: np.ndarray,
    times: Optional[np.ndarray] = None,
    group_labels: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str] | np.ndarray] = None,
    selected_channels: Optional[Sequence[int] | Sequence[str]] = None,
    group_name_map: Optional[dict[Any, str]] = None,
    color_map: Optional[dict[Any, str]] = None,
    title: str = "Grouped Channel Time Series",
    xaxis_title: str = "Time",
    yaxis_title: str = "Amplitude",
    template: str = "plotly_white",
    shared_xaxes: bool = True,
    vertical_spacing: float = 0.05,
    line_width: float = 2.0,
    opacity: float = 1.0,
    base_height: int = 300,
    row_height: int = 220,
    showlegend: bool = True,
) -> go.Figure:
    """
    Plot grouped channel traces as stacked interactive subplots.

    Parameters
    ----------
    data : np.ndarray
        Three-dimensional array with shape ``(n_groups, n_channels, n_times)``.
    times : np.ndarray, optional
        Explicit time axis aligned with the last dimension of ``data``.
    group_labels : np.ndarray, optional
        Labels aligned with the first axis of ``data``.
    channel_names : sequence of str or np.ndarray, optional
        Channel names aligned with the channel axis.
    selected_channels : sequence of int or sequence of str, optional
        Channel indices or names to plot. When omitted, all channels are shown.
    group_name_map : dict, optional
        Optional mapping from raw group labels to display names.
    color_map : dict, optional
        Optional mapping from raw group labels to trace colors.
    title : str, default="Grouped Channel Time Series"
        Figure title.
    xaxis_title : str, default="Time"
        X-axis label for the final row.
    yaxis_title : str, default="Amplitude"
        Y-axis label per subplot row.
    template : str, default="plotly_white"
        Plotly layout template.
    shared_xaxes : bool, default=True
        Whether subplot rows share the same x-axis.
    vertical_spacing : float, default=0.05
        Vertical spacing between subplot rows.
    line_width : float, default=2.0
        Trace line width.
    opacity : float, default=1.0
        Trace opacity.
    base_height : int, default=300
        Base figure height before row scaling.
    row_height : int, default=220
        Additional height per plotted row.
    showlegend : bool, default=True
        Whether to show the legend.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-row channel trace figure.

    Raises
    ------
    ValueError
        If the input shape or aligned labels/names are invalid.
    TypeError
        If ``selected_channels`` contains unsupported selector types.
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(
            "`data` must be 3D with shape (n_groups, n_channels, n_times)."
            f" Got {arr.shape}."
        )
    n_groups, n_channels, n_times = arr.shape

    if times is None:
        x_values = np.arange(n_times)
    else:
        x_values = np.asarray(times)
        if len(x_values) != n_times:
            raise ValueError(
                f"`times` length ({len(x_values)}) must match n_times ({n_times})."
            )

    if group_labels is None:
        groups = np.arange(n_groups)
    else:
        groups = np.asarray(group_labels)
        if len(groups) != n_groups:
            raise ValueError(
                f"`group_labels` length ({len(groups)}) must match n_groups."
                f" Got {n_groups}."
            )

    ch_names = None
    if channel_names is not None:
        ch_names = np.asarray(channel_names).astype(str)
        if len(ch_names) != n_channels:
            raise ValueError(
                f"`channel_names` length ({len(ch_names)}) must match"
                f" n_channels ({n_channels})."
            )

    if selected_channels is None:
        ch_indices = list(range(n_channels))
    else:
        ch_indices = []
        for ch in selected_channels:
            if isinstance(ch, (int, np.integer)):
                idx = int(ch)
            elif isinstance(ch, str):
                if ch_names is None:
                    raise ValueError(
                        "String-based `selected_channels` requires `channel_names`."
                    )
                matches = np.where(ch_names == ch)[0]
                if len(matches) == 0:
                    raise ValueError(f"Channel '{ch}' not found in `channel_names`.")
                idx = int(matches[0])
            else:
                raise TypeError(
                    "`selected_channels` entries must be int indices or str names."
                )
            if idx < 0 or idx >= n_channels:
                raise ValueError(
                    f"Channel index {idx} out of bounds for n_channels={n_channels}."
                )
            ch_indices.append(idx)

    if len(ch_indices) == 0:
        raise ValueError("No channels selected for plotting.")

    subplot_titles = [
        f"Channel: {ch_names[idx]}" if ch_names is not None else f"Channel: {idx}"
        for idx in ch_indices
    ]

    fig = make_subplots(
        rows=len(ch_indices),
        cols=1,
        shared_xaxes=shared_xaxes,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    for row_idx, ch_idx in enumerate(ch_indices, start=1):
        for grp_idx, grp in enumerate(groups):
            display_name = (
                group_name_map.get(grp, str(grp))
                if group_name_map is not None
                else str(grp)
            )
            line_dict = {"width": line_width}
            if color_map is not None and grp in color_map:
                line_dict["color"] = color_map[grp]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=arr[grp_idx, ch_idx, :],
                    mode="lines",
                    name=display_name,
                    legendgroup=str(grp),
                    line=line_dict,
                    opacity=opacity,
                    showlegend=showlegend and row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title_text=yaxis_title, row=row_idx, col=1)

    fig.update_xaxes(title_text=xaxis_title, row=len(ch_indices), col=1)
    fig.update_layout(
        title=title,
        template=template,
        height=base_height + row_height * len(ch_indices),
        margin=dict(l=60, r=40, b=60, t=70),
    )
    return fig


def _discrete_colorscale(
    categories: Sequence[Any], palette: Optional[str | Sequence[str]] = None
):
    colors = (
        list(palette)
        if isinstance(palette, Sequence) and not isinstance(palette, str)
        else getattr(px.colors.qualitative, str(palette), px.colors.qualitative.Plotly)
        if palette is not None
        else px.colors.qualitative.Plotly
    )
    n_categories = max(1, len(categories))
    actual_colors = [colors[i % len(colors)] for i in range(n_categories)]
    scale = []
    step = 1.0 / n_categories
    for i, color in enumerate(actual_colors):
        scale.append([i * step, color])
        scale.append([(i + 1) * step, color])
    return actual_colors, scale


def _marker_payload(
    df: pd.DataFrame,
    column: str,
    cmap: str,
    palette: Optional[str | Sequence[str]],
    *,
    restyle: bool,
):
    values = df[column]
    if is_categorical(values):
        if hasattr(values, "cat"):
            categories = values.cat.categories.tolist()
        else:
            categories = sorted(
                pd.Series(values).dropna().astype(str).unique().tolist()
            )
        cat_map = {cat: i for i, cat in enumerate(categories)}
        mapped = [cat_map.get(v, np.nan) for v in values]
        _, colorscale = _discrete_colorscale(categories, palette=palette)
        payload = {
            "color": mapped,
            "colorscale": colorscale,
            "colorbar": {
                "title": column,
                "tickmode": "array",
                "tickvals": list(range(len(categories))),
                "ticktext": [str(cat) for cat in categories],
            },
            "cmin": 0,
            "cmax": max(1, len(categories) - 1),
        }
        if restyle:
            return {
                "marker.color": [payload["color"]],
                "marker.colorscale": [payload["colorscale"]],
                "marker.colorbar.title": column,
                "marker.colorbar.tickmode": "array",
                "marker.colorbar.tickvals": [payload["colorbar"]["tickvals"]],
                "marker.colorbar.ticktext": [payload["colorbar"]["ticktext"]],
                "marker.cmin": payload["cmin"],
                "marker.cmax": payload["cmax"],
            }
        return payload

    payload = {
        "color": values,
        "colorscale": cmap,
        "colorbar": {"title": column},
        "cmin": None,
        "cmax": None,
    }
    if restyle:
        return {
            "marker.color": [payload["color"]],
            "marker.colorscale": [payload["colorscale"]],
            "marker.colorbar.title": column,
            "marker.colorbar.tickmode": "auto",
            "marker.colorbar.tickvals": None,
            "marker.colorbar.ticktext": None,
            "marker.cmin": None,
            "marker.cmax": None,
        }
    return payload


def plot_embedding_interactive(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
    title: str = "Embedding",
    dimensions: int = 2,
    cmap: str = "Viridis",
    palette: Optional[str | Sequence[str]] = None,
    random_state: Optional[int] = None,
) -> go.Figure:
    """
    Create an interactive 2D or 3D scatter plot of an embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding array with shape ``(n_samples, n_dimensions)``.
    labels : np.ndarray, optional
        Optional values aligned with the sample axis.
    metadata : dict, optional
        Optional column-oriented metadata aligned with the sample axis.
    title : str, default="Embedding"
        Figure title.
    dimensions : int, default=2
        Number of embedding dimensions to plot. Must be 2 or 3.
    cmap : str, default="Viridis"
        Continuous colormap name.
    palette : str or sequence of str, optional
        Discrete color palette used for categorical columns.
    random_state : int, optional
        Reserved for compatibility with data-first static/interactive APIs.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive embedding scatter plot.

    See Also
    --------
    prepare_embedding_frame
    coco_pipe.viz.dim_reduction.plot_embedding
    """
    df = prepare_embedding_frame(
        embedding,
        labels=labels,
        metadata=metadata,
        dimensions=dimensions,
    )

    color_columns: list[str] = []
    if "Label" in df.columns:
        color_columns.append("Label")
    if metadata:
        color_columns.extend(
            [str(key) for key in metadata.keys() if str(key) in df.columns]
        )

    hover_cols = [col for col in df.columns if col not in {"x", "y", "z"}]
    custom_data = df[hover_cols].values if hover_cols else None
    hovertemplate = (
        "<br>".join(
            [
                f"<b>{col}:</b> %{{customdata[{idx}]}}"
                for idx, col in enumerate(hover_cols)
            ]
        )
        if hover_cols
        else None
    )

    marker = {"size": 4 if dimensions == 2 else 3, "opacity": 0.75}
    if color_columns:
        marker.update(
            _marker_payload(
                df, color_columns[0], cmap=cmap, palette=palette, restyle=False
            )
        )

    if dimensions == 3 and "z" in df.columns:
        trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=marker,
            customdata=custom_data,
            hovertemplate=hovertemplate,
            name="Embedding",
        )
    else:
        trace_class = go.Scattergl if len(df) > 15000 else go.Scatter
        trace = trace_class(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=marker,
            customdata=custom_data,
            hovertemplate=hovertemplate,
            name="Embedding",
        )

    fig = go.Figure([trace])
    if len(color_columns) > 1:
        buttons = [
            dict(
                label=column,
                method="restyle",
                args=[
                    _marker_payload(
                        df, column, cmap=cmap, palette=palette, restyle=True
                    )
                ],
            )
            for column in color_columns
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.15,
                    yanchor="top",
                )
            ]
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def plot_loss_history_interactive(
    loss_history: list, title: str = "Training Loss"
) -> go.Figure:
    """Plot training loss history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode="lines", name="Loss"))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        margin=dict(l=40, r=40, b=40, t=40),
        height=300,
        template="plotly_white",
    )
    return fig


def plot_metric_details(
    metrics_df: Any,
    title: str = "Metric Details",
    plot_type: str = "auto",
    metric: Optional[str] = None,
    scope: Optional[str] = None,
    method: Optional[str | Sequence[str]] = None,
) -> go.Figure:
    """
    Create an interactive metric plot from tidy metric observations.

    Parameters
    ----------
    metrics_df : Any
        Metric mapping, tidy metric frame, list of records, or object exposing
        ``to_frame()``.
    title : str, default="Metric Details"
        Figure title.
    plot_type : str, default="auto"
        Plot style to use. ``"auto"`` infers a suitable view from the filtered
        metric records.
    metric : str, optional
        Restrict plotting to one metric.
    scope : str, optional
        Restrict plotting to one scope.
    method : str or sequence of str, optional
        Restrict plotting to one or more methods.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive metric plot.

    See Also
    --------
    prepare_metrics_frame
    infer_metric_plot_type
    coco_pipe.viz.dim_reduction.plot_metrics
    """
    df = filter_metric_frame(
        prepare_metrics_frame(metrics_df),
        metric=metric,
        scope=scope,
        method=method,
    )
    if df.empty:
        raise ValueError("No metrics available to plot.")

    resolved = infer_metric_plot_type(df, requested=plot_type)
    fig = go.Figure()

    if resolved in {"bar", "grouped_bar", "lollipop"}:
        x_col = "method" if df["metric"].nunique() == 1 else "metric"
        hue_groups = ["method"] if x_col == "metric" else []
        if hue_groups:
            grouped = df.pivot_table(
                index=x_col, columns="method", values="value", aggfunc="mean"
            )
            for method_name in grouped.columns:
                values = grouped[method_name].values
                fig.add_trace(
                    go.Bar(
                        name=str(method_name),
                        x=grouped.index.astype(str).tolist(),
                        y=values,
                        text=[f"{val:.3f}" for val in values],
                        textposition="auto",
                    )
                )
        else:
            grouped = df.groupby(x_col, dropna=False)["value"].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=grouped[x_col].astype(str).tolist(),
                    y=grouped["value"].tolist(),
                    text=[f"{val:.3f}" for val in grouped["value"]],
                    textposition="auto",
                    name="value",
                )
            )
        fig.update_layout(
            barmode="group", xaxis_title=x_col.title(), yaxis_title="Score"
        )

    elif resolved in {"box", "boxen", "violin", "raincloud", "strip", "swarm"}:
        x_col = "method" if df["metric"].nunique() == 1 else "metric"
        trace_cls = (
            go.Box if resolved in {"box", "boxen", "strip", "swarm"} else go.Violin
        )
        for name, sub_df in df.groupby("method", dropna=False):
            if trace_cls is go.Box:
                fig.add_trace(
                    go.Box(
                        name=str(name),
                        x=sub_df[x_col].astype(str),
                        y=sub_df["value"],
                        boxpoints="all"
                        if resolved in {"box", "strip", "swarm"}
                        else False,
                        jitter=0.25 if resolved in {"box", "strip", "swarm"} else 0.0,
                        pointpos=0,
                    )
                )
            else:
                fig.add_trace(
                    go.Violin(
                        name=str(name),
                        x=sub_df[x_col].astype(str),
                        y=sub_df["value"],
                        box_visible=resolved == "raincloud",
                        meanline_visible=True,
                        points="all" if resolved == "raincloud" else False,
                        jitter=0.12 if resolved == "raincloud" else 0.0,
                    )
                )
        fig.update_layout(xaxis_title=x_col.title(), yaxis_title="Score")

    elif resolved == "heatmap":
        scope_values = df["scope_value"].astype(str).nunique()
        if scope_values > 1 and df["metric"].nunique() == 1:
            heatmap_df = df.pivot_table(
                index="method", columns="scope_value", values="value", aggfunc="mean"
            )
            x_title = df["scope"].iloc[0].replace("_", " ").title()
        else:
            heatmap_df = df.pivot_table(
                index="method", columns="metric", values="value", aggfunc="mean"
            )
            x_title = "Metric"
        fig.add_trace(
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.astype(str).tolist(),
                y=heatmap_df.index.astype(str).tolist(),
                colorscale="Viridis",
                colorbar=dict(title="Score"),
            )
        )
        fig.update_layout(xaxis_title=x_title, yaxis_title="Method")

    elif resolved == "line":
        group_cols = ["method"]
        if df["metric"].nunique() > 1:
            group_cols.append("metric")
        summary = (
            df.groupby(group_cols + ["scope", "scope_value"], dropna=False)["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for keys, sub_df in summary.groupby(group_cols, dropna=False):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            label = " / ".join(str(k) for k in keys)
            sub_df = sub_df.copy()
            sub_df["scope_numeric"] = pd.to_numeric(
                sub_df["scope_value"], errors="coerce"
            )
            use_numeric = sub_df["scope_numeric"].notna().all()
            sort_col = "scope_numeric" if use_numeric else "scope_value"
            sub_df = sub_df.sort_values(sort_col)
            x_vals = (
                sub_df["scope_numeric"]
                if use_numeric
                else sub_df["scope_value"].astype(str)
            )
            fig.add_trace(
                go.Scatter(x=x_vals, y=sub_df["mean"], mode="lines+markers", name=label)
            )
        fig.update_layout(
            xaxis_title=df["scope"].iloc[0].replace("_", " ").title(),
            yaxis_title="Score",
        )

    elif resolved in {"dumbbell", "slopegraph"}:
        wide = df.pivot_table(
            index="metric", columns="method", values="value", aggfunc="mean"
        )
        if wide.shape[1] != 2:
            raise ValueError("Dumbbell plots require exactly two methods.")
        left_method, right_method = wide.columns.tolist()
        for metric_name, row in wide.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row[left_method], row[right_method]],
                    y=[metric_name, metric_name],
                    mode="lines+markers",
                    marker=dict(size=10),
                    name=str(metric_name),
                    showlegend=False,
                )
            )
        fig.update_layout(xaxis_title="Score", yaxis_title="Metric")
    else:
        raise ValueError(f"Unsupported plot_type: {resolved}")

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, b=40, t=40),
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scree_interactive(explained_variance_ratio: np.ndarray) -> go.Figure:
    """
    Plot explained variance and cumulative variance interactively.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        One-dimensional array of explained variance ratios.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scree plot.
    """
    explained_variance_ratio = np.asarray(explained_variance_ratio)
    components = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=components, y=explained_variance_ratio, name="Individual", opacity=0.7)
    )
    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Scree Plot",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        yaxis2=dict(
            title="Cumulative Variance", overlaying="y", side="right", range=[0, 1.1]
        ),
        legend=dict(x=0.5, y=1.1, orientation="h"),
        margin=dict(l=40, r=40, b=40, t=40),
        height=300,
        template="plotly_white",
    )
    return fig


def plot_radar_comparison(
    metrics_df: pd.DataFrame,
    normalize: bool = True,
    title: str = "Method Comparison",
) -> go.Figure:
    """
    Create a radar chart comparing methods across scalar metrics.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Wide comparison table indexed by method with numeric metric columns.
    normalize : bool, default=True
        Whether to normalize each numeric metric column to ``[0, 1]`` before
        plotting.
    title : str, default="Method Comparison"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive radar comparison figure.

    Notes
    -----
    Radar charts are overview visuals. They are less precise than line plots,
    tables, or heatmaps for detailed method comparisons.
    """
    fig = go.Figure()
    df = metrics_df.copy()
    cols = df.select_dtypes(include=[np.number]).columns
    if normalize:
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if not np.isclose(max_val, min_val):
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 1.0
    categories = list(cols)
    for method_name, row in df.iterrows():
        values = row[categories].values.tolist()
        values += [values[0]]
        cats = categories + [categories[0]]
        fig.add_trace(
            go.Scatterpolar(r=values, theta=cats, fill="toself", name=str(method_name))
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1] if normalize else None)),
        title=title,
        showlegend=True,
        margin=dict(l=40, r=40, b=40, t=40),
        height=400,
        template="plotly_white",
    )
    return fig


def plot_raw_preview(
    data: np.ndarray,
    names: Optional[list] = None,
    title: str = "Raw Data Preview",
    max_points: int = 50000,
) -> go.Figure:
    """
    Create a scrollable preview of multichannel raw traces.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array with shape ``(n_samples, n_channels)``.
    names : list, optional
        Optional channel names aligned with the channel axis.
    title : str, default="Raw Data Preview"
        Figure title.
    max_points : int, default=50000
        Soft limit used to subsample very large inputs for display.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive raw-trace preview with a range slider.
    """
    fig = go.Figure()
    n_samples, n_channels = data.shape
    total_points = n_samples * n_channels
    step = 1
    if total_points > max_points:
        step = int(np.ceil(total_points / max_points))
        if n_samples // step < 100:
            step = 1
    x_axis = np.arange(0, n_samples, step)
    display_channels = min(n_channels, 20)
    for i in range(display_channels):
        trace_data = data[::step, i]
        name = names[i] if names and i < len(names) else f"Ch {i}"
        fig.add_trace(
            go.Scattergl(
                x=x_axis,
                y=trace_data,
                mode="lines",
                name=name,
                opacity=0.8,
                line=dict(width=1),
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(rangeslider=dict(visible=True), title="Sample / Time"),
        yaxis=dict(title="Amplitude"),
        margin=dict(l=40, r=40, b=40, t=40),
        height=450,
        showlegend=True,
        template="plotly_white",
    )
    return fig


def plot_shepard_interactive(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    sample_size: int = 1000,
    title: str = "Shepard Diagram",
    random_state: Optional[int] = None,
    distances: Optional[dict[str, np.ndarray]] = None,
    clip_quantiles: Optional[tuple[float, float]] = (0.01, 0.99),
    scatter_max_points: int = 4000,
    scatter_opacity: float = 0.14,
) -> go.Figure:
    """Create an interactive Shepard diagram using Plotly."""
    from ..dim_reduction.evaluation.metrics import shepard_diagram_data

    if isinstance(distances, dict) and {"original", "embedded"} <= set(distances):
        dist_high = np.asarray(distances["original"])
        dist_low = np.asarray(distances["embedded"])
    else:
        dist_high, dist_low = shepard_diagram_data(
            X_orig, X_emb, sample_size=sample_size, random_state=random_state
        )
    valid = np.isfinite(dist_high) & np.isfinite(dist_low)
    dist_high = dist_high[valid]
    dist_low = dist_low[valid]
    if dist_high.size == 0:
        raise ValueError("No valid pairwise distances to plot in Shepard diagram.")

    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        x_q = np.quantile(dist_high, [q_low, q_high])
        y_q = np.quantile(dist_low, [q_low, q_high])
        data_min = float(min(x_q[0], y_q[0]))
        data_max = float(max(x_q[1], y_q[1]))
    else:
        data_min = float(min(dist_high.min(), dist_low.min()))
        data_max = float(max(dist_high.max(), dist_low.max()))

    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        data_min = float(min(dist_high.min(), dist_low.min()))
        data_max = float(max(dist_high.max(), dist_low.max()))
    if data_max <= data_min:
        data_max = data_min + 1e-6

    pad = 0.03 * (data_max - data_min)
    axis_min = max(0.0, data_min - pad)
    axis_max = data_max + pad

    in_window = (
        (dist_high >= axis_min)
        & (dist_high <= axis_max)
        & (dist_low >= axis_min)
        & (dist_low <= axis_max)
    )
    dist_high_plot = dist_high[in_window]
    dist_low_plot = dist_low[in_window]
    if dist_high_plot.size < 200:
        dist_high_plot = dist_high
        dist_low_plot = dist_low

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=dist_high_plot,
            y=dist_low_plot,
            colorscale="Blues",
            reversescale=False,
            contours=dict(coloring="heatmap"),
            ncontours=12,
            showscale=True,
            colorbar=dict(title="Pair density"),
            name="Density",
        )
    )
    n_pairs = dist_high_plot.size
    if n_pairs > 0:
        if n_pairs > scatter_max_points:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n_pairs, size=scatter_max_points, replace=False)
            x_sc = dist_high_plot[idx]
            y_sc = dist_low_plot[idx]
        else:
            x_sc = dist_high_plot
            y_sc = dist_low_plot
        fig.add_trace(
            go.Scattergl(
                x=x_sc,
                y=y_sc,
                mode="markers",
                marker=dict(size=3, color=f"rgba(0,0,0,{scatter_opacity})"),
                name="Pairs",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal",
        )
    )
    corr = np.corrcoef(dist_high, dist_low)[0, 1] if len(dist_high) > 1 else np.nan
    fig.update_layout(
        title=f"{title}<br>Pearson Corr: {corr:.3f}",
        xaxis=dict(title="Original Distances", range=[axis_min, axis_max]),
        yaxis=dict(title="Embedded Distances", range=[axis_min, axis_max]),
        margin=dict(l=40, r=40, b=40, t=40),
        height=400,
        showlegend=True,
        template="plotly_white",
    )
    return fig


def plot_feature_importance_interactive(
    scores: Any,
    title: str = "Feature Importance",
    top_n: int = 20,
    analysis: Optional[str] = None,
    method: Optional[str] = None,
    dimension: Optional[str] = None,
) -> go.Figure:
    """
    Plot feature importance as an interactive horizontal bar chart.

    Parameters
    ----------
    scores : Any
        Raw ``feature -> score`` mapping, interpretation payload, or
        interpretation record table.
    title : str, default="Feature Importance"
        Figure title.
    top_n : int, default=20
        Maximum number of features to show.
    analysis : str, optional
        Interpretation analysis to select when multiple analyses are present.
    method : str, optional
        Method name to select when multiple methods are present.
    dimension : str, optional
        Dimension label to select when multiple dimensions are present.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive feature-importance bar chart.

    See Also
    --------
    prepare_feature_scores
    plot_interpretation_interactive
    coco_pipe.viz.dim_reduction.plot_feature_importance
    """
    feature_scores = prepare_feature_scores(
        scores,
        analysis=analysis,
        method=method,
        dimension=dimension,
    ).head(top_n)
    fig = go.Figure(
        [
            go.Bar(
                x=feature_scores.values[::-1],
                y=feature_scores.index.astype(str).tolist()[::-1],
                orientation="h",
                marker_color="#348ABD",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=40, r=40, b=40, t=40),
        height=max(400, top_n * 20),
        template="plotly_white",
    )
    return fig


def plot_feature_correlation_heatmap_interactive(
    correlations: Any,
    title: str = "Feature Correlation",
    top_n: Optional[int] = 25,
    method: Optional[str] = None,
) -> go.Figure:
    """
    Plot feature-to-dimension correlations as an interactive heatmap.

    Parameters
    ----------
    correlations : Any
        Correlation interpretation payload or records.
    title : str, default="Feature Correlation"
        Figure title.
    top_n : int, optional
        Maximum number of features to show. Features are ranked by the maximum
        absolute correlation across dimensions.
    method : str, optional
        Method name to select when multiple methods are present.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive feature-correlation heatmap.

    See Also
    --------
    prepare_interpretation_frame
    plot_interpretation_interactive
    coco_pipe.viz.dim_reduction.plot_feature_correlation_heatmap
    """
    frame = prepare_interpretation_frame(correlations)
    frame = frame[frame["analysis"] == "correlation"]
    if method is not None:
        frame = frame[frame["method"] == method]
    elif frame["method"].dropna().nunique() > 1:
        raise ValueError("Specify `method` when multiple methods are present.")
    if frame.empty:
        raise ValueError("No correlation records available to plot.")

    heatmap = frame.pivot_table(
        index="feature", columns="dimension", values="value", aggfunc="mean"
    ).fillna(0.0)
    if top_n is not None and len(heatmap.index) > top_n:
        ranking = heatmap.abs().max(axis=1).sort_values(ascending=False)
        heatmap = heatmap.loc[ranking.head(top_n).index]

    fig = go.Figure(
        [
            go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns.astype(str).tolist(),
                y=heatmap.index.astype(str).tolist(),
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="Correlation"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dimension",
        yaxis_title="Feature",
        template="plotly_white",
    )
    return fig


def plot_interpretation_interactive(
    interpretation: Any,
    *,
    analysis: str,
    title: Optional[str] = None,
    method: Optional[str] = None,
    dimension: Optional[str] = None,
    top_n: int = 20,
) -> go.Figure:
    """
    Plot one interpretation analysis using an interactive Plotly view.

    Parameters
    ----------
    interpretation : Any
        Interpretation payload or interpretation records.
    analysis : str
        Interpretation analysis to plot.
    title : str, optional
        Figure title. Defaults to a title derived from ``analysis``.
    method : str, optional
        Method name to select when multiple methods are present.
    dimension : str, optional
        Dimension label to select when multiple dimensions are present.
    top_n : int, default=20
        Maximum number of features to show in bar or heatmap views.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive interpretation figure.

    See Also
    --------
    plot_feature_importance_interactive
    plot_feature_correlation_heatmap_interactive
    coco_pipe.viz.dim_reduction.plot_interpretation
    """
    if analysis == "correlation":
        return plot_feature_correlation_heatmap_interactive(
            interpretation,
            title=title or "Feature Correlation",
            top_n=top_n,
            method=method,
        )

    matrix = extract_interpretation_matrix(interpretation, analysis=analysis)
    if matrix is not None:
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            scores = {
                f"Feature {i + 1}": float(value) for i, value in enumerate(matrix)
            }
            return plot_feature_importance_interactive(
                scores,
                title=title or analysis.replace("_", " ").title(),
                top_n=top_n,
            )
        fig = go.Figure(
            [
                go.Heatmap(
                    z=matrix,
                    colorscale="Magma",
                    colorbar=dict(title="Score"),
                )
            ]
        )
        fig.update_layout(
            title=title or analysis.replace("_", " ").title(),
            xaxis_title="Feature Index",
            yaxis_title="Feature Axis",
            template="plotly_white",
        )
        return fig

    return plot_feature_importance_interactive(
        interpretation,
        title=title or analysis.replace("_", " ").title(),
        top_n=top_n,
        analysis=analysis,
        method=method,
        dimension=dimension,
    )


def plot_streamlines_interactive(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
    title: str = "Velocity Streamlines",
    random_state: Optional[int] = None,
) -> go.Figure:
    """Plot a velocity vector field using Plotly line segments."""
    if X_emb.shape[1] != 2:
        raise ValueError("Streamlines currently only supported for 2D.")

    if X_emb.shape[0] > 1000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_emb.shape[0], 1000, replace=False)
        X_sub = X_emb[idx]
        V_sub = V_emb[idx]
    else:
        X_sub = X_emb
        V_sub = V_emb

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=X_emb[:, 0],
            y=X_emb[:, 1],
            mode="markers",
            marker=dict(color="#DDDDDD", size=3),
            name="Points",
            hoverinfo="skip",
        )
    )

    scale = 1.0
    span_x = X_emb[:, 0].max() - X_emb[:, 0].min()
    max_v = np.max(np.abs(V_sub))
    if max_v > 0:
        scale = (span_x / 50.0) / max_v

    x_lines = []
    y_lines = []
    for i in range(len(X_sub)):
        x, y = X_sub[i]
        u, v = V_sub[i]
        x_lines.extend([x, x + u * scale, None])
        y_lines.extend([y, y + v * scale, None])

    fig.add_trace(
        go.Scattergl(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color="orange", width=1.5),
            name="Velocity",
            opacity=0.8,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        margin=dict(l=40, r=40, b=40, t=40),
        height=500,
        showlegend=True,
        template="plotly_white",
    )
    return fig


def plot_trajectory_metric_series_interactive(
    series: Any,
    *,
    times: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "Trajectory Metric",
    ylabel: str = "Value",
) -> go.Figure:
    """
    Plot evaluated trajectory metric time series interactively.

    Parameters
    ----------
    series : Any
        One-dimensional series, two-dimensional ``(trajectory, time)`` array,
        or mapping of ``name -> timecourse``.
    times : np.ndarray, optional
        Explicit time axis aligned with the time dimension.
    labels : np.ndarray, optional
        Optional trajectory labels aligned with the first axis of 2D inputs.
    title : str, default="Trajectory Metric"
        Figure title.
    ylabel : str, default="Value"
        Y-axis label.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive trajectory metric series figure.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_trajectory_metric_series
    """
    fig = go.Figure()
    if isinstance(series, Mapping):
        if not series:
            raise ValueError("No trajectory series available to plot.")
        lengths = {len(np.asarray(values).reshape(-1)) for values in series.values()}
        if len(lengths) != 1:
            raise ValueError("All trajectory series must share the same length.")
        n_times = lengths.pop()
        x_vals = np.arange(n_times) if times is None else np.asarray(times)
        if len(x_vals) != n_times:
            raise ValueError("`times` must align with the trajectory time axis.")
        for name, values in series.items():
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.asarray(values).reshape(-1),
                    mode="lines",
                    name=str(name),
                )
            )
    else:
        arr = np.asarray(series)
        if arr.ndim == 1:
            x_vals = np.arange(arr.shape[0]) if times is None else np.asarray(times)
            if len(x_vals) != arr.shape[0]:
                raise ValueError("`times` must align with the trajectory time axis.")
            fig.add_trace(go.Scatter(x=x_vals, y=arr, mode="lines", name=ylabel))
        elif arr.ndim == 2:
            x_vals = np.arange(arr.shape[1]) if times is None else np.asarray(times)
            if len(x_vals) != arr.shape[1]:
                raise ValueError("`times` must align with the trajectory time axis.")
            if labels is not None:
                labels = np.asarray(labels)
                if labels.shape[0] != arr.shape[0]:
                    raise ValueError("`labels` must align with the series axis.")
                for label in np.unique(labels):
                    subset = arr[labels == label]
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=subset.mean(axis=0),
                            mode="lines",
                            name=str(label),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(x=x_vals, y=arr.mean(axis=0), mode="lines", name=ylabel)
                )
        else:
            raise ValueError("Trajectory metric series must be 1D, 2D, or a dict.")
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    return fig


def plot_trajectory_interactive(
    X: np.ndarray,
    times: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
    smooth_window: Optional[int] = None,
) -> go.Figure:
    """
    Plot already-prepared native trajectory tensors interactively.

    Parameters
    ----------
    X : np.ndarray
        Trajectory tensor with shape ``(n_trajectories, n_times, n_dimensions)``.
    times : np.ndarray, optional
        Explicit time axis aligned with the time dimension.
    labels : np.ndarray, optional
        Optional label per trajectory.
    values : np.ndarray, optional
        Optional scalar overlay with shape ``(n_trajectories, n_times)``.
    title : str, default="Trajectory Plot"
        Figure title.
    dimensions : int, default=2
        Number of embedding dimensions to display. Must be 2 or 3.
    smooth_window : int, optional
        Moving-average window applied independently to each already-valid
        trajectory when greater than 1.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive trajectory plot.

    Raises
    ------
    ValueError
        If the input is not a native 3D trajectory tensor or if aligned arrays
        do not match the trajectory/time axes.

    See Also
    --------
    plot_trajectory_metric_series_interactive
    coco_pipe.viz.dim_reduction.plot_trajectory
    """
    trajectories = np.asarray(X)
    if trajectories.ndim != 3:
        raise ValueError(
            "`X` must be a 3D trajectory tensor with shape "
            "(n_trajectories, n_times, n_dimensions)."
        )
    if dimensions not in {2, 3}:
        raise ValueError("`dimensions` must be 2 or 3.")
    if trajectories.shape[2] < dimensions:
        msg = (
            f"`X` has only {trajectories.shape[2]} dimensions; "
            f"cannot plot {dimensions}."
        )
        raise ValueError(msg)

    n_trajectories, n_times, _ = trajectories.shape
    times = np.arange(n_times) if times is None else np.asarray(times)
    if len(times) != n_times:
        raise ValueError("`times` must align with the trajectory time axis.")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != n_trajectories:
            raise ValueError("`labels` must align with the trajectory axis.")
    if values is not None:
        values = np.asarray(values)
        if values.shape != (n_trajectories, n_times):
            raise ValueError("`values` must have shape (n_trajectories, n_times).")

    if smooth_window is not None and smooth_window > 1:
        from ..dim_reduction.evaluation.geometry import moving_average

        trajectories = np.asarray(
            [
                np.stack(
                    [
                        moving_average(traj[:, dim], smooth_window)
                        for dim in range(traj.shape[1])
                    ],
                    axis=1,
                )
                for traj in trajectories
            ]
        )
        times = moving_average(times, smooth_window)
        if values is not None:
            values = np.asarray([moving_average(v, smooth_window) for v in values])

    fig = go.Figure()
    if values is not None:
        for idx, traj in enumerate(trajectories[:, :, :dimensions]):
            if dimensions == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2],
                        mode="lines",
                        line=dict(color="rgba(150,150,150,0.35)", width=4),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=values[idx],
                            colorscale="Viridis",
                            colorbar=dict(title="Value") if idx == 0 else None,
                            showscale=idx == 0,
                        ),
                        name=str(labels[idx])
                        if labels is not None
                        else f"Trajectory {idx + 1}",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        mode="lines",
                        line=dict(color="rgba(150,150,150,0.35)", width=4),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        mode="markers",
                        marker=dict(
                            size=7,
                            color=values[idx],
                            colorscale="Viridis",
                            colorbar=dict(title="Value") if idx == 0 else None,
                            showscale=idx == 0,
                        ),
                        name=str(labels[idx])
                        if labels is not None
                        else f"Trajectory {idx + 1}",
                    )
                )
    else:
        palette = px.colors.qualitative.Plotly
        label_color_map = None
        if labels is not None:
            unique_labels = list(dict.fromkeys(labels.tolist()))
            label_color_map = {
                label: palette[idx % len(palette)]
                for idx, label in enumerate(unique_labels)
            }
        for idx, traj in enumerate(trajectories[:, :, :dimensions]):
            color = (
                label_color_map[labels[idx]]
                if label_color_map is not None
                else palette[idx % len(palette)]
            )
            name = str(labels[idx]) if labels is not None else f"Trajectory {idx + 1}"
            if dimensions == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2],
                        mode="lines+markers",
                        line=dict(color=color, width=4),
                        marker=dict(size=4, color=color),
                        name=name,
                        legendgroup=name,
                        showlegend=name
                        not in {trace.name for trace in fig.data if trace.name},
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        mode="lines+markers",
                        line=dict(color=color, width=3),
                        marker=dict(size=6, color=color),
                        name=name,
                        legendgroup=name,
                        showlegend=name
                        not in {trace.name for trace in fig.data if trace.name},
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Dimension 1" if dimensions == 2 else None,
        yaxis_title="Dimension 2" if dimensions == 2 else None,
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
        )
        if dimensions == 3
        else None,
    )
    return fig
