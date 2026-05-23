"""Interactive Plotly visualization helpers for dimensionality reduction outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._utils import (
    filter_metric_frame,
    prepare_component_loadings_frame,
    prepare_eigenvalue_curves,
    prepare_embedding_frame,
    prepare_feature_scores,
    prepare_interpretation_frame,
    prepare_loss_history,
    prepare_metrics_frame,
    prepare_shepard_distances,
    prepare_streamline_inputs,
    prepare_trajectory_data,
    prepare_trajectory_metric_series,
    prepare_trajectory_separation_series,
)
from ..theme import _COLORBLIND_COLORS, DIVERGING, SEQUENTIAL, ColorKind
from ._utils import COCO_TEMPLATE, _apply_layout, _marker_payload

__all__ = [
    "plot_channel_traces",
    "plot_coranking_matrix",
    "plot_component_loadings",
    "plot_eigenvalues",
    "plot_embedding",
    "plot_feature_correlation_heatmap",
    "plot_feature_importance",
    "plot_loss_history",
    "plot_metrics",
    "plot_radar_comparison",
    "plot_raw_preview",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_trajectory",
    "plot_trajectory_metric_series",
    "plot_trajectory_separation",
]


def plot_channel_traces(
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
    template: str = COCO_TEMPLATE,
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
    template : str, default="coco"
        Plotly layout template. Defaults to the registered CoCo template.
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

    See Also
    --------
    plot_raw_preview : Scrollable preview of multichannel raw traces.
    plot_radar_comparison : Radar chart comparing methods across scalar metrics.
    plot_embedding : Interactive 2D or 3D scatter plot of an embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> data = np.random.default_rng(42).normal(size=(2, 4, 50))
    >>> fig = viz.plot_channel_traces(data)
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(
            "`data` must be 3D with shape "
            f"(n_groups, n_channels, n_times). Got {arr.shape}."
        )
    n_groups, n_channels, n_times = arr.shape

    x_values = np.arange(n_times) if times is None else np.asarray(times)
    if len(x_values) != n_times:
        raise ValueError(
            f"`times` length ({len(x_values)}) must match n_times ({n_times})."
        )

    groups = np.arange(n_groups) if group_labels is None else np.asarray(group_labels)
    if len(groups) != n_groups:
        raise ValueError(
            f"`group_labels` length ({len(groups)}) must match n_groups ({n_groups})."
        )

    ch_names = None
    if channel_names is not None:
        ch_names = np.asarray(channel_names).astype(str)
        if len(ch_names) != n_channels:
            raise ValueError(
                f"`channel_names` length ({len(ch_names)}) must match "
                f"n_channels ({n_channels})."
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
            color = (
                color_map[grp]
                if color_map is not None and grp in color_map
                else _COLORBLIND_COLORS[grp_idx % len(_COLORBLIND_COLORS)]
            )
            line_dict: dict[str, Any] = {"color": color, "width": line_width}
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
    _apply_layout(
        fig,
        title=title,
        template=template,
        height=base_height + row_height * len(ch_indices),
        legend_horizontal=True,
    )
    fig.update_layout(margin=dict(l=60, r=40, b=60, t=70))
    return fig


def plot_embedding(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
    title: str = "Embedding",
    dimensions: int = 2,
    cmap: str = SEQUENTIAL,
    palette: Optional[str | Sequence[str]] = None,
    color_kind: ColorKind = "categorical",
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
    cmap : str, default=SEQUENTIAL
        Continuous colormap name.
    palette : str or sequence of str, optional
        Discrete color palette used for categorical columns.
    color_kind : {"categorical", "continuous"}, default="categorical"
        How to color the first available label or metadata column.
    random_state : int, optional
        Accepted for API compatibility; not used internally.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive embedding scatter plot.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_embedding : Static Matplotlib version.
    plot_shepard_diagram : Validates embedding structure via distance preservation.
    plot_metrics : Metric summary for the same embedding.
    plot_trajectory : Overlay temporal trajectories on an embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> X_emb = np.random.default_rng(42).normal(size=(50, 2))
    >>> fig = viz.plot_embedding(X_emb)
    """
    _ = random_state
    df = prepare_embedding_frame(
        embedding,
        labels=labels,
        metadata=metadata,
        dimensions=dimensions,
        label_kind=color_kind,
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
            f"<b>{col}:</b> %{{customdata[{idx}]}}"
            for idx, col in enumerate(hover_cols)
        )
        if hover_cols
        else None
    )
    marker: dict[str, Any] = {"size": 4 if dimensions == 2 else 3, "opacity": 0.75}
    if color_columns:
        marker.update(
            _marker_payload(
                df,
                color_columns[0],
                cmap=cmap,
                palette=palette,
                color_kind=color_kind,
                restyle=False,
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
                        df,
                        column,
                        cmap=cmap,
                        palette=palette,
                        color_kind=color_kind,
                        restyle=True,
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
    _apply_layout(fig, title=title)
    return fig


def plot_loss_history(
    loss_history: list,
    title: str = "Training Loss",
) -> go.Figure:
    """
    Plot training loss history as an interactive Plotly line chart.

    Parameters
    ----------
    loss_history
        Sequence of per-epoch loss values.
    title
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive loss curve figure.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_loss_history : Static Matplotlib version.
    plot_eigenvalues : Scree plot of explained variance ratios.
    plot_metrics : Metric bar chart for post-fit quality evaluation.

    Examples
    --------
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> fig = viz.plot_loss_history([1.0, 0.7, 0.4, 0.25, 0.18])
    """
    losses = prepare_loss_history(loss_history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(losses.size),
            y=losses,
            mode="lines",
            name="Loss",
        )
    )
    _apply_layout(fig, title=title, xaxis_title="Epoch", yaxis_title="Loss", height=300)
    return fig


def plot_metrics(
    metrics_df: Any,
    title: str = "Metric Details",
    plot_type: Literal[
        "bar",
        "grouped_bar",
        "lollipop",
        "box",
        "boxen",
        "violin",
        "raincloud",
        "strip",
        "swarm",
        "heatmap",
        "line",
        "dumbbell",
        "slopegraph",
    ] = "bar",
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
    plot_type : str, default="bar"
        Explicit plot style to use.
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
    coco_pipe.viz.dim_reduction.plot_metrics : Static Matplotlib version.
    plot_eigenvalues : Scree plot complementing variance-based metrics.
    plot_shepard_diagram : Distance-preservation diagnostic.
    plot_coranking_matrix : Rank-based quality matrix.

    Examples
    --------
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> fig = viz.plot_metrics({"trustworthiness": 0.9, "continuity": 0.85})
    """
    df = filter_metric_frame(
        prepare_metrics_frame(metrics_df),
        metric=metric,
        scope=scope,
        method=method,
    )
    if df.empty:
        raise ValueError("No metrics available to plot.")

    fig = go.Figure()

    if plot_type in {"bar", "grouped_bar", "lollipop"}:
        x_col = "Method" if df["Metric"].nunique() == 1 else "Metric"
        if x_col == "Metric":
            grouped = df.pivot_table(
                index="Metric", columns="Method", values="Value", aggfunc="mean"
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
            grouped = df.groupby("Method", dropna=False)["Value"].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=grouped["Method"].astype(str).tolist(),
                    y=grouped["Value"].tolist(),
                    text=[f"{val:.3f}" for val in grouped["Value"]],
                    textposition="auto",
                    name="Value",
                )
            )
        _apply_layout(
            fig,
            title=title,
            xaxis_title=x_col,
            yaxis_title="Score",
            barmode="group",
            height=420,
        )

    elif plot_type in {"box", "boxen", "violin", "raincloud", "strip", "swarm"}:
        x_col = "Method" if df["Metric"].nunique() == 1 else "Metric"
        for name, sub_df in df.groupby("Method", dropna=False):
            if plot_type in {"box", "boxen", "strip", "swarm"}:
                fig.add_trace(
                    go.Box(
                        name=str(name),
                        x=sub_df[x_col].astype(str),
                        y=sub_df["Value"],
                        boxpoints="all"
                        if plot_type in {"box", "strip", "swarm"}
                        else False,
                        jitter=0.25 if plot_type in {"box", "strip", "swarm"} else 0.0,
                        pointpos=0,
                    )
                )
            else:
                fig.add_trace(
                    go.Violin(
                        name=str(name),
                        x=sub_df[x_col].astype(str),
                        y=sub_df["Value"],
                        box_visible=plot_type == "raincloud",
                        meanline_visible=True,
                        points="all" if plot_type == "raincloud" else False,
                        jitter=0.12 if plot_type == "raincloud" else 0.0,
                    )
                )
        _apply_layout(
            fig, title=title, xaxis_title=x_col, yaxis_title="Score", height=420
        )

    elif plot_type == "heatmap":
        scope_values = df["ScopeValue"].astype(str).nunique()
        if scope_values > 1 and df["Metric"].nunique() == 1:
            heatmap_df = df.pivot_table(
                index="Method", columns="ScopeValue", values="Value", aggfunc="mean"
            )
            x_title = df["Scope"].iloc[0].replace("_", " ").title()
        else:
            heatmap_df = df.pivot_table(
                index="Method", columns="Metric", values="Value", aggfunc="mean"
            )
            x_title = "Metric"
        fig.add_trace(
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.astype(str).tolist(),
                y=heatmap_df.index.astype(str).tolist(),
                colorscale=SEQUENTIAL,
                colorbar=dict(title="Score"),
            )
        )
        _apply_layout(
            fig, title=title, xaxis_title=x_title, yaxis_title="Method", height=420
        )

    elif plot_type == "line":
        group_cols = ["Method"]
        if df["Metric"].nunique() > 1:
            group_cols.append("Metric")
        summary = (
            df.groupby(group_cols + ["Scope", "ScopeValue"], dropna=False)["Value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for keys, sub_df in summary.groupby(group_cols, dropna=False):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            label = " / ".join(str(k) for k in keys)
            sub_df = sub_df.copy()
            sub_df["scope_numeric"] = pd.to_numeric(
                sub_df["ScopeValue"], errors="coerce"
            )
            use_numeric = sub_df["scope_numeric"].notna().all()
            sort_col = "scope_numeric" if use_numeric else "ScopeValue"
            sub_df = sub_df.sort_values(sort_col)
            x_vals = (
                sub_df["scope_numeric"]
                if use_numeric
                else sub_df["ScopeValue"].astype(str)
            )
            fig.add_trace(
                go.Scatter(x=x_vals, y=sub_df["mean"], mode="lines+markers", name=label)
            )
        _apply_layout(
            fig,
            title=title,
            xaxis_title=df["Scope"].iloc[0].replace("_", " ").title(),
            yaxis_title="Score",
            height=420,
        )

    elif plot_type in {"dumbbell", "slopegraph"}:
        wide = df.pivot_table(
            index="Metric", columns="Method", values="Value", aggfunc="mean"
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
        _apply_layout(
            fig, title=title, xaxis_title="Score", yaxis_title="Metric", height=420
        )
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    return fig


def plot_eigenvalues(
    explained_variance_ratio: np.ndarray,
) -> go.Figure:
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

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_eigenvalues : Static Matplotlib version.
    plot_loss_history : Training loss curve for iterative methods.
    plot_metrics : Broader metric quality summary.
    plot_component_loadings : Component loading heatmap for linear reducers.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> fig = viz.plot_eigenvalues(np.array([0.5, 0.3, 0.2]))
    """
    curve = prepare_eigenvalue_curves(explained_variance_ratio)[0]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=curve["components"],
            y=curve["mean"],
            name="Individual",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["components"],
            y=curve["cumulative"],
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
        margin=dict(l=50, r=50, b=50, t=55),
        height=350,
        template="coco",
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

    See Also
    --------
    plot_metrics : Tidy metric bar/box/line charts for the same data.
    plot_channel_traces : Grouped channel traces for raw exploration.
    plot_raw_preview : Scrollable raw data preview.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> df = pd.DataFrame(
    ...     {"trustworthiness": [0.9, 0.85], "continuity": [0.88, 0.82]},
    ...     index=["UMAP", "t-SNE"],
    ... )
    >>> fig = viz.plot_radar_comparison(df)
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
        margin=dict(l=40, r=40, b=40, t=55),
        height=420,
        template="coco",
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

    See Also
    --------
    plot_channel_traces : Grouped channel traces with stacked subplots.
    plot_radar_comparison : Radar chart comparing methods across metrics.
    plot_embedding : Embedding scatter after dimensionality reduction.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> data = np.random.default_rng(42).normal(size=(200, 8))
    >>> fig = viz.plot_raw_preview(data)
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
        margin=dict(l=50, r=40, b=50, t=55),
        height=450,
        showlegend=True,
        template="coco",
    )
    return fig


def plot_shepard_diagram(
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
    """
    Create an interactive Shepard diagram.

    Parameters
    ----------
    X_orig
        Original high-dimensional data. Ignored when ``distances`` is given.
    X_emb
        Low-dimensional embedding. Ignored when ``distances`` is given.
    sample_size
        Number of point pairs sampled for distance computation.
    title
        Figure title.
    random_state
        Random seed for reproducible distance sampling.
    distances
        Pre-computed dict with ``"original"`` and ``"embedded"`` keys.
        When both keys are present, ``X_orig`` and ``X_emb`` are not used.
    clip_quantiles
        ``(low, high)`` quantile pair used to clip axis ranges for display.
        Set to ``None`` to use the full distance range.
    scatter_max_points
        Maximum number of individual pair points shown as a scatter overlay.
    scatter_opacity
        Opacity of the scatter overlay points.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Shepard diagram with density contours and scatter overlay.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_shepard_diagram : Static Matplotlib version.
    plot_embedding : Embedding scatter plot for visual inspection.
    plot_coranking_matrix : Rank-order quality matrix complementing Shepard analysis.
    plot_metrics : Summary of trustworthiness and continuity metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> fig = viz.plot_shepard_diagram(
    ...     rng.normal(size=(30, 5)), rng.normal(size=(30, 2))
    ... )
    """
    dist_high, dist_low, corr = prepare_shepard_distances(
        X_orig,
        X_emb,
        sample_size=sample_size,
        random_state=random_state,
        distances=distances,
    )

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
            colorscale=SEQUENTIAL,
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
    _apply_layout(
        fig,
        title=f"{title}<br>Pearson Corr: {corr:.3f}",
        xaxis_title="Original Distances",
        yaxis_title="Embedded Distances",
        height=420,
    )
    fig.update_xaxes(range=[axis_min, axis_max])
    fig.update_yaxes(range=[axis_min, axis_max])
    return fig


def plot_feature_importance(
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
    coco_pipe.viz.dim_reduction.plot_feature_importance : Static Matplotlib version.
    plot_feature_correlation_heatmap : Feature-to-dimension correlation heatmap.
    plot_component_loadings : Component loading matrix for linear reducers.

    Examples
    --------
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> fig = viz.plot_feature_importance({"F1": 0.6, "F2": 0.4, "F3": 0.2})
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
                marker_color=_COLORBLIND_COLORS[0],
            )
        ]
    )
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, top_n * 22),
    )
    return fig


def plot_feature_correlation_heatmap(
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
    coco_pipe.viz.dim_reduction.plot_feature_correlation_heatmap :
        Static Matplotlib version.
    plot_feature_importance : Feature importance bar chart.
    plot_component_loadings : Linear component loadings heatmap.

    Examples
    --------
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> corr = {"correlation": {"D1": {"F1": 0.3,
    ...                           "F2": -0.1},
    ...                           "D2": {"F1": 0.5,
    ...                           "F2": 0.2}}}
    >>> fig = viz.plot_feature_correlation_heatmap(corr)
    """
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be a positive integer or None.")
    frame = prepare_interpretation_frame(correlations)
    frame = frame[frame["Analysis"] == "correlation"]
    if method is not None:
        frame = frame[frame["Method"] == method]
    elif frame["Method"].dropna().nunique() > 1:
        raise ValueError("Specify `method` when multiple methods are present.")
    if frame.empty:
        raise ValueError("No correlation records available to plot.")
    heatmap = frame.pivot_table(
        index="Feature", columns="Dimension", values="Value", aggfunc="mean"
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
                colorscale=DIVERGING,
                zmid=0.0,
                colorbar=dict(title="Correlation"),
            )
        ]
    )
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Dimension",
        yaxis_title="Feature",
        height=max(400, len(heatmap.index) * 18 + 100),
    )
    return fig


def plot_streamlines(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
    title: str = "Velocity Streamlines",
    random_state: Optional[int] = None,
) -> go.Figure:
    """
    Plot a velocity vector field using Plotly line segments.

    Parameters
    ----------
    X_emb
        2D embedding coordinates with shape ``(n_samples, 2)``.
    V_emb
        Velocity vectors with the same shape as ``X_emb``.
    grid_density
        Accepted for API parity with the static version; not used internally.
    title
        Figure title.
    random_state
        Random seed used when subsampling large point clouds.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive velocity field figure.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_streamlines : Static Matplotlib version.
    plot_embedding : Embedding scatter that provides context for the velocity field.
    plot_trajectory : Trajectory lines overlaid on an embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> fig = viz.plot_streamlines(rng.normal(size=(50, 2)), rng.normal(size=(50, 2)))
    """
    _ = grid_density
    X_emb, V_emb = prepare_streamline_inputs(X_emb, V_emb)
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
    x_lines: list[Any] = []
    y_lines: list[Any] = []
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
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=500,
    )
    return fig


def plot_trajectory_metric_series(
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
    coco_pipe.viz.dim_reduction.plot_trajectory_metric_series :
        Static Matplotlib version.
    plot_trajectory : Trajectory geometry in 2D or 3D space.
    plot_trajectory_separation : Pairwise label-separation timecourses.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> series = np.random.default_rng(42).normal(size=(3, 20))
    >>> fig = viz.plot_trajectory_metric_series(series)
    """
    frame = prepare_trajectory_metric_series(series, times=times, labels=labels)
    fig = go.Figure()
    groups = list(frame.groupby("Series", sort=False))
    for name, group in groups:
        errors = group["Error"].to_numpy(dtype=float)
        error_y = None if np.isnan(errors).all() else {"type": "data", "array": errors}
        fig.add_trace(
            go.Scatter(
                x=group["Time"],
                y=group["Value"],
                error_y=error_y,
                mode="lines",
                name=str(name) if len(groups) > 1 else ylabel,
            )
        )
    _apply_layout(fig, title=title, xaxis_title="Time", yaxis_title=ylabel)
    return fig


def plot_trajectory(
    X: np.ndarray,
    times: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
    smooth_window: Optional[int] = None,
    downsample: int = 1,
) -> go.Figure:
    """
    Plot native trajectory tensors interactively.

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
        Moving-average window applied to each trajectory when greater than 1.
    downsample : int, default=1
        Keep every ``downsample``-th time point after smoothing.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive trajectory plot.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_trajectory : Static Matplotlib version.
    plot_trajectory_separation : Pairwise label-separation timecourses.
    plot_trajectory_metric_series : Scalar metric timecourses per trajectory.
    plot_embedding : Static embedding scatter for context.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> X = np.random.default_rng(42).normal(size=(3, 20, 2))
    >>> fig = viz.plot_trajectory(X)
    """
    trajectories, _, labels, values, dimensions = prepare_trajectory_data(
        X,
        times=times,
        labels=labels,
        values=values,
        dimensions=dimensions,
        smooth_window=smooth_window,
        downsample=downsample,
    )
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
                            colorscale=SEQUENTIAL,
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
                            colorscale=SEQUENTIAL,
                            colorbar=dict(title="Value") if idx == 0 else None,
                            showscale=idx == 0,
                        ),
                        name=str(labels[idx])
                        if labels is not None
                        else f"Trajectory {idx + 1}",
                    )
                )
    else:
        palette = list(_COLORBLIND_COLORS)
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
            show = name not in {trace.name for trace in fig.data if trace.name}
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
                        showlegend=show,
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
                        showlegend=show,
                    )
                )
    _apply_layout(fig, title=title)
    if dimensions == 2:
        fig.update_layout(xaxis_title="Dimension 1", yaxis_title="Dimension 2")
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
            )
        )
    return fig


def plot_coranking_matrix(
    coranking_matrix: np.ndarray,
    title: str = "Co-Ranking Matrix",
    max_k: Optional[int] = None,
) -> go.Figure:
    """
    Plot a co-ranking matrix as an interactive heatmap.

    Parameters
    ----------
    coranking_matrix
        Square co-ranking matrix with shape ``(n_samples-1, n_samples-1)``.
    title
        Figure title.
    max_k
        Crop the matrix to the top-left ``max_k × max_k`` corner. Defaults
        to ``min(n, 50)``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive co-ranking matrix heatmap.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_coranking_matrix : Static Matplotlib version.
    plot_shepard_diagram : Distance-level quality diagnostic.
    plot_metrics : Scalar trustworthiness and continuity summary.
    plot_embedding : Embedding scatter for visual inspection.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> fig = viz.plot_coranking_matrix(np.eye(8))
    """
    if coranking_matrix is None:
        raise ValueError("coranking_matrix is required.")
    matrix = np.asarray(coranking_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coranking_matrix must be a 2D square array.")
    k = min(matrix.shape[0], 50 if max_k is None else max_k)
    matrix = matrix[:k, :k]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            colorscale=SEQUENTIAL,
            colorbar=dict(title="Count"),
        )
    )
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Embedding Rank",
        yaxis_title="Original Rank",
        height=500,
    )
    return fig


def plot_trajectory_separation(
    separation: dict,
    times: Optional[np.ndarray] = None,
    top_n: Optional[int] = None,
    title: str = "Trajectory Separation",
) -> go.Figure:
    """
    Plot pairwise label-separation timecourses interactively.

    Parameters
    ----------
    separation
        Mapping of ``(label_a, label_b)`` tuples (or any hashable key) to
        1D separation timecourses.
    times
        Explicit time axis aligned with the separation arrays.
    top_n
        Keep only the ``top_n`` pairs ranked by peak separation.
    title
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive trajectory separation figure.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_trajectory_separation : Static Matplotlib version.
    plot_trajectory : Trajectory geometry in embedding space.
    plot_trajectory_metric_series : Scalar metric timecourses per trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> sep = {("A", "B"): np.arange(10, dtype=float)}
    >>> fig = viz.plot_trajectory_separation(sep)
    """
    items = prepare_trajectory_separation_series(
        separation,
        times=times,
        top_n=top_n,
    )
    fig = go.Figure()
    for item in items:
        fig.add_trace(
            go.Scatter(x=item["x"], y=item["y"], mode="lines", name=item["label"])
        )
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Time",
        yaxis_title="Separation",
        height=420,
    )
    return fig


def plot_component_loadings(
    components: np.ndarray,
    feature_names: Optional[list[str]] = None,
    n_components: Optional[int] = None,
    title: str = "Component Loadings",
) -> go.Figure:
    """
    Plot component loadings from linear reducers as an interactive heatmap.

    Parameters
    ----------
    components
        Loading matrix with shape ``(n_features, n_components)``.
    feature_names
        Optional feature names for row labels.
    n_components
        Crop to this many components (columns).
    title
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive component-loadings heatmap.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_component_loadings : Static Matplotlib version.
    plot_feature_importance : Feature importance bar chart.
    plot_feature_correlation_heatmap : Feature-to-dimension correlation heatmap.
    plot_eigenvalues : Scree plot of explained variance per component.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz.interactive import dim_reduction as viz
    >>> components = np.random.default_rng(42).normal(size=(10, 3))
    >>> fig = viz.plot_component_loadings(components)
    """
    loadings = prepare_component_loadings_frame(
        components,
        feature_names=feature_names,
        n_components=n_components,
    )
    fig = go.Figure(
        go.Heatmap(
            z=loadings.values,
            x=loadings.columns.astype(str).tolist(),
            y=loadings.index.astype(str).tolist(),
            colorscale=DIVERGING,
            zmid=0.0,
            colorbar=dict(title="Loading"),
        )
    )
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Component",
        yaxis_title="Feature",
        height=max(420, loadings.shape[0] * 18 + 100),
    )
    return fig
