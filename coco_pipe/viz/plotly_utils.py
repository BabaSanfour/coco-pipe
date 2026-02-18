"""
Plotly Visualization Utilities
==============================

Functions to generate interactive Plotly figures for dimensionality reduction analysis.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import is_categorical, prepare_dataframe


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
    n_points = embedding.shape[0]

    # Use shared prepare_dataframe for logic centralization
    df = prepare_dataframe(embedding, labels=labels, meta=meta, dimensions=dimensions)
    
    # Hover data includes all columns except x, y, z
    hover_cols = [c for c in df.columns if c not in ["x", "y", "z"]]
    hover_data = hover_cols
    
    # Identify color options
    color_options = {}
    
    # Re-identify color options from DataFrame columns (since prepare_dataframe simplifies this)
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
            unique_cats = sorted(df[default_color_col].unique())
            cat_map = {cat: i for i, cat in enumerate(unique_cats)}
            color_vals = [cat_map[v] for v in raw_vals]
            colorscale_title = default_color_col
            
            # Create a discrete-looking scale
            n_colors = len(unique_cats)
            # Fetch qualitative colors
            pal = palette if isinstance(palette, list) else getattr(px.colors.qualitative, palette, px.colors.qualitative.Plotly)
            
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
                tickmode='array',
                tickvals=list(range(n_colors)),
                ticktext=[str(c) for c in unique_cats]
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
                unique_cats = sorted(raw_v.unique())
                cat_map = {cat: i for i, cat in enumerate(unique_cats)}
                n_cols = len(unique_cats)
                
                # Fetch qualitative colors
                pal = palette if isinstance(palette, list) else getattr(px.colors.qualitative, str(palette), px.colors.qualitative.Plotly)
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
                    "marker.cmax": max(1, n_cols - 1)
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
                    "marker.cmax": None
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
) -> go.Figure:
    """
    Create an interactive Shepard Diagram using Plotly (scatter/hexbin approximation).
    """
    from ..dim_reduction.evaluation.metrics import shepard_diagram_data

    dist_high, dist_low = shepard_diagram_data(X_orig, X_emb, sample_size=sample_size)

    # Use a 2D Histogram contour for density visualization (like hexbin)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram2dContour(
            x=dist_high,
            y=dist_low,
            colorscale="Blues",
            reversescale=False,
            xaxis="x",
            yaxis="y",
            name="Density",
        )
    )

    # Ideal Line (y=x)
    min_val = min(dist_high.min(), dist_low.min())
    max_val = max(dist_high.max(), dist_low.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
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
        xaxis_title="Original Distances",
        yaxis_title="Embedded Distances",
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
    X: np.ndarray,
    times: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
) -> go.Figure:
    """
    Plot trajectories of samples over time.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_components). Coordinates.
    times : np.ndarray, optional
        Time points or indices for coloring.
    groups : np.ndarray, optional
        Group labels (e.g. trial IDs) to separate trajectories.
        If None, all points are treated as one single trajectory.
    title : str
    dimensions : int
        2 or 3.

    Returns
    -------
    go.Figure
    """
    if dimensions not in [2, 3]:
        raise ValueError("Dimensions must be 2 or 3.")

    n_samples = X.shape[0]
    if groups is None:
        groups = np.zeros(n_samples, dtype=int)  # Single group

    unique_groups = np.unique(groups)
    fig = go.Figure()

    # Determine colors
    # If times provided, color by time (continuous)
    # If no times, lines are solid color by group
    use_time_color = times is not None

    for grp in unique_groups:
        mask = groups == grp
        X_g = X[mask]

        if use_time_color:
            t_g = times[mask]
            # Plotly Scatter lines don't support gradients easily solely via
            # `line.color`
            # But `markers` do. We can use markers+lines and color markers by time.
            # Lines will be constant color or we separate segments.
            # Simpler approach: Grey line + colored markers.

            # Line (path)
            if dimensions == 2:
                fig.add_trace(
                    go.Scatter(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        mode="lines",
                        line=dict(color="grey", width=1),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                # Markers (time)
                fig.add_trace(
                    go.Scatter(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=t_g,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Time")
                            if grp == unique_groups[0]
                            else None,
                        ),
                        text=[f"Time: {t}" for t in t_g],
                        name=f"Group {grp}",
                    )
                )
            else:
                # 3D
                fig.add_trace(
                    go.Scatter3d(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        z=X_g[:, 2],
                        mode="lines",
                        line=dict(color="grey", width=2),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        z=X_g[:, 2],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=t_g,
                            colorscale="Viridis",
                            colorbar=dict(title="Time")
                            if grp == unique_groups[0]
                            else None,
                        ),
                        name=f"Group {grp}",
                    )
                )

        else:
            # Color by group
            if dimensions == 2:
                fig.add_trace(
                    go.Scatter(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        mode="lines+markers",
                        marker=dict(size=6),
                        name=str(grp),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=X_g[:, 0],
                        y=X_g[:, 1],
                        z=X_g[:, 2],
                        mode="lines+markers",
                        marker=dict(size=4),
                        name=str(grp),
                    )
                )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, b=40, t=40),
        height=600 if dimensions == 3 else 500,
    )

    return fig
