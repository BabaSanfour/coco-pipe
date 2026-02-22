"""
Plotly Visualization Utilities
==============================

Functions to generate interactive Plotly figures for dimensionality reduction analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import is_categorical, prepare_dataframe


def plot_channel_traces_interactive(
    data: np.ndarray,
    times: Optional[np.ndarray] = None,
    group_labels: Optional[np.ndarray] = None,
    channel_names: Optional[Union[List[str], np.ndarray]] = None,
    selected_channels: Optional[Union[List[int], List[str]]] = None,
    group_name_map: Optional[Dict[Any, str]] = None,
    color_map: Optional[Dict[Any, str]] = None,
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
    Plot grouped time series as channel-wise subplot rows.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_groups, n_channels, n_times).
    times : np.ndarray, optional
        Time axis of length n_times. If None, uses np.arange(n_times).
    group_labels : np.ndarray, optional
        Labels for groups of length n_groups. If None, uses np.arange(n_groups).
    channel_names : list or np.ndarray, optional
        Channel names of length n_channels.
    selected_channels : list of int or str, optional
        Channel subset to display. If None, displays all channels.
        String items require `channel_names`.
    group_name_map : dict, optional
        Mapping group label -> display name.
    color_map : dict, optional
        Mapping group label -> line color.
    title : str
        Figure title.
    xaxis_title : str
    yaxis_title : str
    template : str
    shared_xaxes : bool
    vertical_spacing : float
    line_width : float
    opacity : float
    base_height : int
    row_height : int
    showlegend : bool

    Returns
    -------
    go.Figure
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

    subplot_titles = []
    for idx in ch_indices:
        if ch_names is not None:
            subplot_titles.append(f"Channel: {ch_names[idx]}")
        else:
            subplot_titles.append(f"Channel: {idx}")

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


def plot_embedding_interactive(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Embedding",
    dimensions: int = 2,
    cmap: str = "Viridis",
    palette: Optional[Union[str, List[str]]] = None,
) -> go.Figure:
    """
    Create an interactive 2D or 3D scatter plot of the embedding using Plotly.
    Supports a dropdown menu to switch coloring if `meta` contains valid options.

    Parameters
    ----------
    embedding : np.ndarray
        (N_samples, n_components) array.
    labels : np.ndarray, optional
        Labels for default coloring points.
    meta : dict, optional
        Additional metadata for tooltips and optional coloring options.
    title : str
        Plot title.
    dimensions : int
        Number of dimensions to plot (2 or 3).
    cmap : str, default="Viridis"
        Colormap for continuous variables.
    palette : str or list, optional
        Qualitative colormap/palette for categorical variables.
        If None, uses Plotly default qualitative palette.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    # Use shared prepare_dataframe for logic centralization
    df = prepare_dataframe(embedding, labels=labels, meta=meta, dimensions=dimensions)

    # Hover data includes all columns except x, y, z
    hover_cols = [c for c in df.columns if c not in ["x", "y", "z"]]
    hover_data = hover_cols

    # Identify color options
    color_options = {}

    # Re-identify color options from DataFrame columns
    # (since prepare_dataframe simplifies this)
    # Default Label is usually the first choice
    if "Default Label" in df.columns:
        color_options["Default Label"] = df["Default Label"]

    # Add meta columns
    if meta:
        for k in meta.keys():
            if k in df.columns:
                color_options[k] = df[k]

    # Decision: Use WebGL for performance if large
    render_mode = "svg"
    if df.shape[0] > 15000:
        render_mode = "webgl"

    # Initial Color
    default_color_col = "Default Label" if "Default Label" in df.columns else None

    # If no labels but we have meta options, pick the first one
    if default_color_col is None and color_options:
        default_color_col = list(color_options.keys())[0]

    # Create Base Figure
    fig = go.Figure()

    # Determine if default is categorical
    is_cat = False
    if default_color_col:
        is_cat = is_categorical(df[default_color_col])

    # Default Palette
    import plotly.express as px

    if palette is None:
        palette = px.colors.qualitative.Plotly

    # Base Trace
    color_vals = None
    colorscale_title = ""
    marker_color = None
    plotly_scale = None
    colorbar_dict = None
    cmin, cmax = None, None

    if default_color_col:
        raw_vals = df[default_color_col]
        if is_cat:
            # Map categories to colors manually for go.Scatter performance
            if hasattr(df[default_color_col], "cat"):
                unique_cats = df[default_color_col].cat.categories.tolist()
            else:
                unique_cats = sorted(df[default_color_col].unique())

            cat_map = {cat: i for i, cat in enumerate(unique_cats)}
            color_vals = [cat_map[v] for v in raw_vals]
            colorscale_title = default_color_col

            # Create a discrete-looking scale
            n_colors = len(unique_cats)
            # Fetch qualitative colors
            pal = (
                palette
                if isinstance(palette, list)
                else getattr(
                    px.colors.qualitative, palette, px.colors.qualitative.Plotly
                )
            )

            # Repeat palette if too many categories
            actual_colors = [pal[i % len(pal)] for i in range(n_colors)]

            # Plotly colorscale format: [[0, c1], [1/N, c1], [1/N, c2], [2/N, c2]...]
            # to create discrete steps on a continuous bar (best for restyle compat)
            discrete_scale = []
            for i, c in enumerate(actual_colors):
                discrete_scale.append([i / n_colors, c])
                discrete_scale.append([(i + 1) / n_colors, c])

            # Colorbar with ticktext for categories
            colorbar_dict = dict(
                title=colorscale_title,
                tickmode="array",
                tickvals=list(range(n_colors)),
                ticktext=[str(c) for c in unique_cats],
            )

            # Convert actual_colors to Plotly scale
            step = 1.0 / n_colors
            plotly_scale = []
            for i, col in enumerate(actual_colors):
                plotly_scale.append([i * step, col])
                plotly_scale.append([(i + 1) * step, col])

            marker_color = color_vals
            cmin, cmax = 0, max(1, n_colors - 1)

        else:
            color_vals = raw_vals
            marker_color = color_vals
            colorscale_title = default_color_col
            plotly_scale = cmap
            colorbar_dict = dict(title=colorscale_title)
            cmin, cmax = None, None

    marker_dict = dict(
        size=3 if dimensions == 3 else 5,
        opacity=0.7,
        color=marker_color if default_color_col else None,
        showscale=True if default_color_col else False,
        colorscale=plotly_scale if default_color_col else None,
        colorbar=colorbar_dict if default_color_col else None,
        cmin=cmin,
        cmax=cmax,
    )

    # Hover text construction
    custom_data = df[hover_data].values
    hovertemplate = "<br>".join(
        [f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_data)]
    )

    if dimensions == 3 and "z" in df.columns:
        trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=marker_dict,
            customdata=custom_data,
            hovertemplate=hovertemplate,
            name="Embedding",
        )
    else:
        ScatterClass = go.Scattergl if render_mode == "webgl" else go.Scatter
        trace = ScatterClass(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=marker_dict,
            customdata=custom_data,
            hovertemplate=hovertemplate,
            name="Embedding",
        )

    fig.add_trace(trace)

    # Add Dropdowns (Update Menus)
    if len(color_options) > 1:

        def _get_marker_update(col_name):
            raw_v = df[col_name]
            is_cat_col = is_categorical(raw_v)

            if is_cat_col:
                if hasattr(raw_v, "cat"):
                    unique_cats = raw_v.cat.categories.tolist()
                else:
                    unique_cats = sorted(raw_v.unique())
                cat_map = {cat: i for i, cat in enumerate(unique_cats)}
                n_cols = len(unique_cats)

                # Fetch qualitative colors
                pal = (
                    palette
                    if isinstance(palette, list)
                    else getattr(
                        px.colors.qualitative,
                        str(palette),
                        px.colors.qualitative.Plotly,
                    )
                )
                actual_cols = [pal[i % len(pal)] for i in range(n_cols)]

                # Convert actual_colors to Plotly scale
                step_size = 1.0 / n_cols
                p_scale = []
                for i, col in enumerate(actual_cols):
                    p_scale.append([i * step_size, col])
                    p_scale.append([(i + 1) * step_size, col])

                return {
                    "marker.color": [[cat_map[v] for v in raw_v]],
                    "marker.colorscale": [p_scale],
                    "marker.colorbar.title": col_name,
                    "marker.colorbar.tickmode": "array",
                    "marker.colorbar.tickvals": [list(range(n_cols))],
                    "marker.colorbar.ticktext": [[str(c) for c in unique_cats]],
                    "marker.cmin": 0,
                    "marker.cmax": max(1, n_cols - 1),
                }
            else:
                return {
                    "marker.color": [raw_v],
                    "marker.colorscale": [cmap],
                    "marker.colorbar.title": col_name,
                    "marker.colorbar.tickmode": "auto",
                    "marker.colorbar.tickvals": None,
                    "marker.colorbar.ticktext": None,
                    "marker.cmin": None,
                    "marker.cmax": None,
                }

        buttons = []
        for col_name in color_options.keys():
            update_dict = _get_marker_update(col_name)
            buttons.append(dict(label=col_name, method="restyle", args=[update_dict]))

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.15,
                    yanchor="top",
                ),
            ]
        )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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
    )
    return fig


def plot_metric_details(
    metrics_df: pd.DataFrame, title: str = "Metric Details"
) -> go.Figure:
    """
    Create a grouped bar chart comparing methods across detailed quality metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Index = Method Names, Columns = Metrics (Trustworthiness, Continuity, etc.)

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Filter numeric
    df = metrics_df.select_dtypes(include=[np.number])

    methods = df.index.tolist()
    metrics = df.columns.tolist()

    for method in methods:
        fig.add_trace(
            go.Bar(
                name=method,
                x=metrics,
                y=df.loc[method],
                text=df.loc[method].apply(lambda x: f"{x:.2f}"),
                textposition="auto",
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Score",
        margin=dict(l=40, r=40, b=40, t=40),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scree_interactive(explained_variance_ratio: np.ndarray) -> go.Figure:
    """Plot Scree plot of explained variance."""
    components = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)

    fig = go.Figure()

    # Bar for individual variance
    fig.add_trace(
        go.Bar(x=components, y=explained_variance_ratio, name="Individual", opacity=0.7)
    )

    # Line for cumulative
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
    )
    return fig


def plot_radar_comparison(
    metrics_df: pd.DataFrame, normalize: bool = True, title: str = "Method Comparison"
) -> go.Figure:
    """
    Create a radar chart comparing multiple methods across metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame where Index = Method Names, Columns = Metrics.
        Values should be numeric.
    normalize : bool
        If True, normalizes each metric (column) to [0, 1] range individually.
    title : str
        Plot title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Work on a copy
    df = metrics_df.copy()

    cols = df.select_dtypes(include=[np.number]).columns
    # Radar charts ideally need 3+ variables, but 2 works (line).

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
        # Radar charts in Plotly require closing the loop
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
    )

    return fig


def plot_raw_preview(
    data: np.ndarray,
    names: Optional[list] = None,
    title: str = "Raw Data Preview",
    max_points: int = 50000,
) -> go.Figure:
    """
    Create a scrollable preview of raw data traces.

    Parameters
    ----------
    data : np.ndarray
        2D array (n_samples, n_channels).
    names : list, optional
        Names for the channels (columns).
    title : str
        Plot title.
    max_points : int
        Maximum total points to plot.
        If n_samples * n_channels > max_points, data is decimated.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    n_samples, n_channels = data.shape

    # Decimation factor
    total_points = n_samples * n_channels
    step = 1
    if total_points > max_points:
        step = int(np.ceil(total_points / max_points))
        # Ensure we don't decimate below reasonable resolution
        if n_samples // step < 100:
            step = 1

    x_axis = np.arange(0, n_samples, step)

    # Limit channels
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
    )

    return fig


def plot_shepard_interactive(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    sample_size: int = 1000,
    title: str = "Shepard Diagram",
    random_state: Optional[int] = None,
    clip_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
    scatter_max_points: int = 4000,
    scatter_opacity: float = 0.14,
) -> go.Figure:
    """
    Create an interactive Shepard Diagram using Plotly.

    By default, axes are clipped to central quantiles to avoid outlier-driven
    empty space, and a faint scatter overlay is added on top of density contours.
    """
    from ..dim_reduction.evaluation.metrics import shepard_diagram_data

    dist_high, dist_low = shepard_diagram_data(
        X_orig, X_emb, sample_size=sample_size, random_state=random_state
    )
    valid = np.isfinite(dist_high) & np.isfinite(dist_low)
    dist_high = dist_high[valid]
    dist_low = dist_low[valid]
    if dist_high.size == 0:
        raise ValueError("No valid pairwise distances to plot in Shepard diagram.")

    if clip_quantiles is not None:
        if len(clip_quantiles) != 2:
            raise ValueError("`clip_quantiles` must be a tuple (q_low, q_high).")
        q_low, q_high = clip_quantiles
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("`clip_quantiles` must satisfy 0 <= q_low < q_high <= 1.")
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
            xaxis="x",
            yaxis="y",
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

    # Ideal Line (y=x)
    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal",
        )
    )

    if len(dist_high) > 1:
        corr = np.corrcoef(dist_high, dist_low)[0, 1]
    else:
        corr = np.nan

    fig.update_layout(
        title=f"{title}<br>Pearson Corr: {corr:.3f}",
        xaxis=dict(title="Original Distances", range=[axis_min, axis_max]),
        yaxis=dict(title="Embedded Distances", range=[axis_min, axis_max]),
        margin=dict(l=40, r=40, b=40, t=40),
        height=400,
        showlegend=True,
    )
    return fig


def plot_comparison_interactive(
    comparison_manager: Any,
    metric: str = "trustworthiness",
    title: Optional[str] = None,
) -> go.Figure:
    """Plot metric comparison curves using Plotly."""
    if not comparison_manager.results_:
        raise ValueError("No results found. Run comparison.run() first.")

    if title is None:
        title = f"Comparison: {metric.replace('_', ' ').title()}"

    fig = go.Figure()

    for name, df in comparison_manager.results_.items():
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["k"],
                    y=df[metric],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Neighborhood Size (k)",
        yaxis_title=metric.replace("_", " ").title(),
        margin=dict(l=40, r=40, b=40, t=40),
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


def plot_feature_importance_interactive(
    scores: Dict[str, float], title: str = "Feature Importance", top_n: int = 20
) -> go.Figure:
    """Plot feature importance bar chart using Plotly."""
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=values[::-1], y=names[::-1], orientation="h", marker_color="#348ABD")
    )

    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        margin=dict(l=40, r=40, b=40, t=40),
        height=max(400, top_n * 20),
    )
    return fig


def plot_streamlines_interactive(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
    title: str = "Velocity Streamlines",
    random_state: Optional[int] = None,
) -> go.Figure:
    """
    Plot velocity vector field using Plotly (line segments approximation).
    """
    if X_emb.shape[1] != 2:
        raise ValueError("Streamlines currently only supported for 2D.")

    # Subsample data points for clarity
    if X_emb.shape[0] > 1000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_emb.shape[0], 1000, replace=False)
        X_sub = X_emb[idx]
        V_sub = V_emb[idx]
    else:
        X_sub = X_emb
        V_sub = V_emb

    # Calculate magnitude for possible coloring (not used in simple lines yet)
    # mag = np.sqrt(V_sub[:, 0] ** 2 + V_sub[:, 1] ** 2)

    fig = go.Figure()

    # Background points
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

    # Quiver (Arrows) - Manual segments construction
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
    )
    return fig


def plot_trajectory_interactive(
    X: Any,
    times: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
    smooth_window: Optional[int] = None,
    mode: str = "lines+markers",
    line_width: int = 6,
    group_cmaps: Optional[Dict[Any, str]] = None,
    show_group_colorbars: bool = True,
    colorbar_y: float = 0.5,
    colorbar_len: float = 0.6,
) -> go.Figure:
    """
    Plot trajectories of samples over time with group-specific shading.

    Parameters
    ----------
    X : np.ndarray or DataContainer-like
        Accepted forms:
        - 2D array: (n_samples, n_components)
        - 3D array: (n_trajectories, n_times, n_components)
        - container-like object with attributes `.X` and optional `.y/.coords/.dims`
          using either 2D or 3D underlying data.
    times : np.ndarray, optional
        Time points or indices for coloring.
    groups : np.ndarray, optional
        Group labels (e.g. trial IDs) to separate trajectories.
    values : np.ndarray, optional
        Values to color the trajectory by (e.g. speed). Overrides times.
    dimensions : int
        2 or 3.
    smooth_window : int, optional
        Window size for moving average smoothing.
    mode : str, default="lines+markers"
        Plot mode: 'lines', 'markers', or 'lines+markers'.
    line_width : int, default=6
    group_cmaps : dict, optional
        Mapping from group label to Plotly colorscale name.
    show_group_colorbars : bool, default=True
        If True, each group gets its own colorbar.
    colorbar_y : float, default=0.5
        Vertical position of colorbars in paper coordinates.
    colorbar_len : float, default=0.6
        Colorbar length in paper coordinates.

    Returns
    -------
    go.Figure
    """
    if dimensions not in [2, 3]:
        raise ValueError("Dimensions must be 2 or 3.")

    container = None
    if hasattr(X, "X"):
        container = X
        X_arr = np.asarray(container.X)
    else:
        X_arr = np.asarray(X)

    # Dataset-agnostic coercion:
    # (N, D)                    -> direct
    # (G, T, D)                 -> flatten to (G*T, D), infer groups/times if missing
    # container with 3D data    -> reorder to (obs, time, feature) then flatten
    # container with 2D data    -> direct (+ infer groups from y/obs if missing)
    if X_arr.ndim == 3:
        if container is not None and hasattr(container, "dims"):
            dims = list(container.dims)
            if len(dims) != 3:
                raise ValueError(
                    f"3D container trajectory plotting expects 3 dims, got {dims}."
                )
            obs_axis = dims.index("obs") if "obs" in dims else 0
            time_axis = dims.index("time") if "time" in dims else 1
            feat_axis_candidates = [
                i for i in range(3) if i not in (obs_axis, time_axis)
            ]
            feat_axis = feat_axis_candidates[0]
            X_arr = np.transpose(X_arr, (obs_axis, time_axis, feat_axis))
        n_traj, n_time, n_dim = X_arr.shape
        X_arr = X_arr.reshape(n_traj * n_time, n_dim)

        if groups is None:
            if container is not None and getattr(container, "y", None) is not None:
                y = np.asarray(container.y)
                if len(y) == n_traj:
                    groups = np.repeat(y, n_time)
                else:
                    groups = np.repeat(np.arange(n_traj), n_time)
            elif (
                container is not None
                and hasattr(container, "coords")
                and isinstance(container.coords, dict)
                and "obs" in container.coords
                and len(container.coords["obs"]) == n_traj
            ):
                groups = np.repeat(np.asarray(container.coords["obs"]), n_time)
            else:
                groups = np.repeat(np.arange(n_traj), n_time)

        if times is None:
            if (
                container is not None
                and hasattr(container, "coords")
                and isinstance(container.coords, dict)
                and "time" in container.coords
                and len(container.coords["time"]) == n_time
            ):
                times = np.tile(np.asarray(container.coords["time"]), n_traj)
            else:
                times = np.tile(np.arange(n_time), n_traj)

        if values is not None:
            values_arr = np.asarray(values)
            if values_arr.ndim == 2 and values_arr.shape == (n_traj, n_time):
                values = values_arr.reshape(-1)

    elif X_arr.ndim == 2:
        if container is not None and groups is None:
            n_samples_local = X_arr.shape[0]
            if (
                getattr(container, "y", None) is not None
                and len(container.y) == n_samples_local
            ):
                groups = np.asarray(container.y)
            elif (
                hasattr(container, "coords")
                and isinstance(container.coords, dict)
                and "obs" in container.coords
                and len(container.coords["obs"]) == n_samples_local
            ):
                groups = np.asarray(container.coords["obs"])
    else:
        raise ValueError(
            "`X` must be 2D or 3D array-like (or container with such data),"
            f" got shape {X_arr.shape}."
        )

    n_samples = X_arr.shape[0]
    if X_arr.shape[1] < dimensions:
        raise ValueError(
            f"X has {X_arr.shape[1]} components, cannot plot dimensions={dimensions}."
        )

    def _coerce_1d(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        out = np.asarray(arr).reshape(-1)
        if len(out) != n_samples:
            raise ValueError(
                f"`{name}` length ({len(out)}) must match number of"
                f" samples ({n_samples})."
            )
        return out

    times = _coerce_1d("times", times)
    groups = _coerce_1d("groups", groups)
    values = _coerce_1d("values", values)
    if groups is None:
        groups = np.zeros(n_samples, dtype=int)  # Single group

    # Drop rows with invalid coordinates or coloring variable values
    valid = np.all(np.isfinite(X_arr[:, :dimensions]), axis=1)
    if times is not None and np.issubdtype(np.asarray(times).dtype, np.number):
        valid &= np.isfinite(times)
    if values is not None and np.issubdtype(np.asarray(values).dtype, np.number):
        valid &= np.isfinite(values)
    if not np.all(valid):
        X_arr = X_arr[valid]
        groups = groups[valid]
        if times is not None:
            times = times[valid]
        if values is not None:
            values = values[valid]

    n_samples = X_arr.shape[0]
    if n_samples == 0:
        raise ValueError("No valid samples remain after filtering invalid values.")

    unique_groups = np.unique(groups)
    fig = go.Figure()

    # Determine coloring variable
    color_var = None
    color_label = ""
    default_colorscale = "Viridis"

    if values is not None:
        color_var = values
        color_label = "Value"
        default_colorscale = "Plasma"
    elif times is not None:
        color_var = times
        color_label = "Time"
        default_colorscale = "Viridis"

    # Global min/max for color scale consistency
    cmin, cmax = None, None
    if color_var is not None:
        cmin, cmax = np.nanmin(color_var), np.nanmax(color_var)

    import plotly.colors as pc
    import plotly.express as px

    palette = px.colors.qualitative.Plotly

    sequential_scales = [
        "Blues",
        "Reds",
        "Greens",
        "Purples",
        "Oranges",
        "YlOrBr",
        "YlGnBu",
        "RdPu",
        "GnBu",
    ]

    for g_idx, grp in enumerate(unique_groups):
        mask = groups == grp
        X_g = X_arr[mask]
        c_g = color_var[mask] if color_var is not None else None

        # Determine colorscale for this group
        if group_cmaps and grp in group_cmaps:
            g_colorscale = group_cmaps[grp]
        elif c_g is not None and len(unique_groups) > 1:
            g_colorscale = sequential_scales[g_idx % len(sequential_scales)]
        else:
            g_colorscale = default_colorscale

        # Sample color at 75% for legend representation
        if c_g is not None:
            try:
                grp_color = pc.sample_colorscale(g_colorscale, [0.75])[0]
            except Exception:
                grp_color = palette[g_idx % len(palette)]
        else:
            grp_color = palette[g_idx % len(palette)]

        # Smoothing
        if smooth_window is not None and smooth_window > 1:
            from coco_pipe.dim_reduction.evaluation.geometry import moving_average

            X_g_list = []
            for d in range(X_g.shape[1]):
                X_g_list.append(moving_average(X_g[:, d], smooth_window))
            X_g = np.stack(X_g_list, axis=1)
            if c_g is not None:
                c_g = moving_average(c_g, smooth_window)

        # Plotting Logic
        ScatterClass = go.Scatter3d if dimensions == 3 else go.Scatter
        coords_names = ["x", "y", "z"] if dimensions == 3 else ["x", "y"]

        if c_g is not None and "lines" in mode:
            # Segmented coloring for line
            try:
                scale_colors = getattr(pc.sequential, g_colorscale)
            except AttributeError:
                scale_colors = pc.get_colorscale(g_colorscale)
                scale_colors = [c[1] for c in scale_colors]

            for i in range(len(X_g) - 1):
                seg_coords = {
                    coords_names[d]: [X_g[i, d], X_g[i + 1, d]]
                    for d in range(dimensions)
                }
                seg_val = (c_g[i] + c_g[i + 1]) / 2.0
                norm_val = 0.0
                if cmax is not None and cmin is not None and cmax > cmin:
                    norm_val = (seg_val - cmin) / (cmax - cmin)

                hex_color = pc.sample_colorscale(scale_colors, [norm_val])[0]

                fig.add_trace(
                    ScatterClass(
                        **seg_coords,
                        mode="lines",
                        line=dict(color=hex_color, width=line_width),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=str(grp),
                    )
                )

            # Dummy trace for legend and individual colorbar
            dummy_coords = {coords_names[d]: [None] for d in range(dimensions)}

            marker_dict = None
            if show_group_colorbars:
                # Use wider spacing for multiple colorbars to avoid text overlap
                cb_spacing = 0.07 if len(unique_groups) > 4 else 0.08
                cb_x = 1.02 + (g_idx * cb_spacing)

                marker_dict = dict(
                    color=[cmin, cmax],
                    colorscale=g_colorscale,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Time (s)", side="right")
                        if g_idx == len(unique_groups) - 1
                        else None,
                        x=cb_x,
                        y=colorbar_y,
                        len=colorbar_len,
                        thickness=10,
                        xanchor="left",
                        outlinecolor="white",
                        outlinewidth=0,
                        showticklabels=(g_idx == len(unique_groups) - 1),
                    ),
                )

            fig.add_trace(
                ScatterClass(
                    **dummy_coords,
                    mode="lines+markers" if marker_dict else "lines",
                    marker=marker_dict,
                    line=dict(color=grp_color, width=line_width),
                    name=str(grp),
                    showlegend=True,
                    legendgroup=str(grp),
                )
            )

        elif "markers" in mode:
            marker_dict = dict(size=line_width * 0.8)
            if c_g is not None:
                marker_dict.update(
                    color=c_g,
                    colorscale=g_colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=True
                    if show_group_colorbars and len(unique_groups) == 1
                    else False,
                )
            else:
                marker_dict.update(color=grp_color)

            fig.add_trace(
                ScatterClass(
                    **{coords_names[d]: X_g[:, d] for d in range(dimensions)},
                    mode="markers",
                    marker=marker_dict,
                    name=str(grp),
                    text=[f"{color_label}: {v:.2f}" for v in c_g]
                    if c_g is not None
                    else None,
                )
            )

        if c_g is None and "lines" in mode:
            fig.add_trace(
                ScatterClass(
                    **{coords_names[d]: X_g[:, d] for d in range(dimensions)},
                    mode="lines",
                    line=dict(color=grp_color, width=line_width),
                    name=str(grp),
                )
            )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=100 + (len(unique_groups) * 40), b=40, t=40),
        height=600 if dimensions == 3 else 500,
        template="plotly_white",
    )

    if dimensions == 3:
        fig.update_layout(
            scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3")
        )

    return fig
