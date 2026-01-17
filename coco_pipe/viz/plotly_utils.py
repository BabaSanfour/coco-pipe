"""
Plotly Visualization Utilities
==============================

Functions to generate interactive Plotly figures for dimensionality reduction analysis.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_embedding_interactive(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
    title: str = "Embedding",
    dimensions: int = 2,
) -> go.Figure:
    """
    Create an interactive 2D or 3D scatter plot of the embedding.

    Parameters
    ----------
    embedding : np.ndarray
        (N_samples, n_components) array.
    labels : np.ndarray, optional
        Labels for coloring points.
    meta : dict, optional
        Additional metadata for tooltips.
    title : str
        Plot title.
    dimensions : int
        Number of dimensions to plot (2 or 3).

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    n_points = embedding.shape[0]

    # Prepare DataFrame for Plotly Express (easier for hover/color)
    df_dict = {"x": embedding[:, 0], "y": embedding[:, 1]}

    if dimensions == 3 and embedding.shape[1] > 2:
        df_dict["z"] = embedding[:, 2]

    if labels is not None:
        df_dict["Label"] = labels
        # Convert to string if we want discrete colors for few classes
        if len(np.unique(labels)) < 20 and not np.issubdtype(labels.dtype, np.number):
            df_dict["Label"] = labels.astype(str)

    # Add metadata
    if meta:
        for k, v in meta.items():
            if len(v) == n_points:
                df_dict[k] = v

    df = pd.DataFrame(df_dict)

    hover_data = list(df_dict.keys())

    color_col = "Label" if labels is not None else None

    # Use WebGL traces (scatter_gl) for performance on 2D
    if dimensions == 3 and "z" in df.columns:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color=color_col,
            title=title,
            hover_data=hover_data,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=3))
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_col,
            title=title,
            hover_data=hover_data,
            render_mode="webgl",  # Explicitly use WebGL
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=5))

    fig.update_layout(
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

    # Select numeric columns only
    cols = df.select_dtypes(include=[np.number]).columns
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
