"""
Dimensionality Reduction Visualization
======================================

This module provides specialized plotting functions for analyzing dimensionality
reduction results, ensuring publication-ready aesthetics via Seaborn.

Functions
---------
plot_embedding
    Enhanced scatter plot for 2D/3D embeddings with automatic legend/colorbar.
plot_metrics
    Bar chart visualizing quality metrics.
plot_loss_history
    Line plot of training loss.
plot_eigenvalues
    Scree plot of eigenvalues.
plot_shepard_diagram
    Scatter plot of original vs. embedded distances.
plot_streamlines
    Streamline plot for vector fields (e.g., velocity) on 2D embeddings.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-07
"""

from typing import Optional, Union, Tuple, Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.colors as mcolors

from ..dim_reduction.benchmark.metrics import shepard_diagram_data

# --- Style Constants ---
STYLE_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

def _set_style(context: str = "paper", style: str = "ticks"):
    """
    Set plotting style for publication-ready aesthetics using Seaborn.
    
    Parameters
    ----------
    context : str, optional
        Seaborn context (e.g. 'paper', 'notebook', 'talk', 'poster'), by default "paper".
    style : str, optional
        Seaborn style (e.g. 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'), by default "ticks".
    """
    # Apply matplotlib rcParams base overrides for consistency
    plt.rcParams.update(STYLE_CONFIG)
    
    # Seaborn overrides
    sns.set_context(context, font_scale=1.2)
    sns.set_style(style, rc=STYLE_CONFIG)
    sns.despine(trim=True, offset=10)


def _is_categorical(labels: np.ndarray) -> bool:
    """
    Heuristic to check if labels are categorical or continuous.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of labels.

    Returns
    -------
    bool
        True if categorical (string, bool, or few unique numeric values), False otherwise.
    """
    if labels.dtype.kind in ('U', 'S', 'O', 'b'): # String/Object/Bool
        return True
    
    # If numeric, check unique count
    n_unique = len(np.unique(labels))
    if n_unique < 20: 
        return True
    return False

def plot_embedding(X_emb: np.ndarray, 
                   labels: Optional[np.ndarray] = None, 
                   dims: Union[Tuple[int, int], Tuple[int, int, int]] = (0, 1),
                   title: str = "Embedding", 
                   figsize: Tuple[int, int] = (10, 8),
                   cmap: str = 'viridis',
                   palette: str = 'deep', 
                   s: int = 40,
                   alpha: float = 0.8,
                   metrics: Optional[Dict[str, Any]] = None,
                   ax: Optional[plt.Axes] = None,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D or 3D embedding with publication-ready aesthetics.

    Parameters
    ----------
    X_emb : np.ndarray
        Coordinates of the embedding. Shape (n_samples, n_components).
    labels : np.ndarray, optional
        Labels for coloring points. Automatically detected as categorical or continuous.
    dims : tuple, optional
        Indices of dimensions to plot (e.g. (0, 1) for 2D, (0, 1, 2) for 3D), by default (0, 1).
    title : str, optional
        Plot title, by default "Embedding".
    figsize : tuple, optional
        Figure size in inches, by default (10, 8).
    cmap : str, optional
        Colormap for continuous labels (e.g. 'viridis', 'magma'), by default 'viridis'.
    palette : str, optional
        Seaborn palette for categorical labels (e.g. 'deep', 'tab10', 'Set2'), by default 'deep'.
    s : int, optional
        Marker size, by default 40.
    alpha : float, optional
        Point transparency (0 to 1), by default 0.8.
    metrics : dict, optional
        Dictionary of metrics to overlay on the plot, by default None.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    save_path : str, optional
        Path to save the figure to. If None, figure is not saved.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the plot.
    
    Raises
    ------
    ValueError
        If the number of dimensions in `dims` is not 2 or 3.
    """
    _set_style()
    
    n_plot_dim = len(dims)
    if n_plot_dim not in [2, 3]:
        raise ValueError(f"plot_embedding only supports 2D or 3D plotting, got {n_plot_dim}.")
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        if n_plot_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    coords = X_emb[:, list(dims)]
    
    # Label handling
    if labels is not None:
        is_cat = _is_categorical(labels)
        
        if is_cat:
            # Categorical: Use Seaborn for best legend handling
            if n_plot_dim == 2:
                sns.scatterplot(
                    x=coords[:, 0], y=coords[:, 1], 
                    hue=labels, style=None, 
                    palette=palette, s=s, alpha=alpha, 
                    edgecolor='w', linewidth=0.5,
                    ax=ax, legend='full'
                )
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, title="Group")
            else:
                # 3D Manual
                unique_labels = np.unique(labels)
                n_classes = len(unique_labels)
                colors = sns.color_palette(palette, n_classes)

                for i, lbl in enumerate(unique_labels):
                    idx = labels == lbl
                    c = colors[i]
                    ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], label=str(lbl),
                                color=c, s=s, alpha=alpha, edgecolors='w', linewidth=0.5)
                
                ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        
        else:
            # Continuous
            if n_plot_dim == 2:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, 
                                     cmap=cmap, s=s, alpha=alpha, edgecolors='none')
            else:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels,
                                     cmap=cmap, s=s, alpha=alpha, edgecolors='none')
            
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02 if n_plot_dim==2 else 0.1, fraction=0.046)
            cbar.set_label("Value", size=12)
            cbar.outline.set_visible(False)
            
    else:
        # No labels
        color = sns.color_palette()[0]
        if n_plot_dim == 2:
            ax.scatter(coords[:, 0], coords[:, 1], color=color, s=s, alpha=alpha, edgecolors='w', linewidth=0.5)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, s=s, alpha=alpha, edgecolors='w', linewidth=0.5)

    # Axis Formatting
    ax.set_xlabel(f"Dimension {dims[0]+1}", fontweight='bold')
    ax.set_ylabel(f"Dimension {dims[1]+1}", fontweight='bold')
    if n_plot_dim == 3:
        ax.set_zlabel(f"Dimension {dims[2]+1}", fontweight='bold')
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    ax.set_title(title, pad=15, fontweight='bold')

    # Metrics Overlay
    if metrics:
        clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int, str)) and k not in ['n_iter_', 'n_components']}
        text_str = '\n'.join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in clean_metrics.items()])
        
        if n_plot_dim == 2:
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props, zorder=100)
        else:
            plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        
    return fig


def plot_metrics(scores: Dict[str, Any],
                 title: str = "Quality Metrics",
                 figsize: Tuple[int, int] = (8, 6),
                 ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot bar chart of quality metrics with publication style.

    Parameters
    ----------
    scores : dict
        Dictionary of metric names and values. Non-scalar values are ignored.
    title : str, optional
        Plot title, by default "Quality Metrics".
    figsize : tuple, optional
        Figure size, by default (8, 6).
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If no scalar metrics are found in `scores`.
    """
    _set_style()
    metrics = {k: v for k, v in scores.items() if isinstance(v, (float, int)) and not isinstance(v, bool)}
    exclude_keys = {'n_iter_', 'n_components'}
    metrics = {k: v for k, v in metrics.items() if k not in exclude_keys}

    if not metrics:
        raise ValueError("No scalar metrics found to plot.")

    names = list(metrics.keys())
    values = list(metrics.values())
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    sns.barplot(x=names, y=values, ax=ax, palette="viridis", edgecolor='black', linewidth=0.8)

    # Value Labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim([min(0, min(values) - 0.1), max(1.05, max(values)*1.15)])
    
    ax.set_ylabel("Score", fontweight='bold')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def plot_loss_history(loss_history: list,
                      title: str = "Training Loss",
                      figsize: Tuple[int, int] = (8, 5),
                      ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot training loss over epochs.

    Parameters
    ----------
    loss_history : list
        List of loss values.
    title : str, optional
        Plot title, by default "Training Loss".
    figsize : tuple, optional
        Figure size, by default (8, 5).
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        
    # Nice thick line with shadow/marker
    ax.plot(loss_history, linewidth=2.5, color='#E24A33', label='Loss', marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    
    # Minimalist grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig


def plot_eigenvalues(values: np.ndarray,
                     title: str = "Scree Plot",
                     ylabel: str = "Explained Variance",
                     figsize: Tuple[int, int] = (8, 5),
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot Scree plot of eigenvalues or explained variance.

    Parameters
    ----------
    values : np.ndarray
        Array of eigenvalues or variance ratios.
    title : str, optional
        Plot title, by default "Scree Plot".
    ylabel : str, optional
        Label for y-axis, by default "Explained Variance".
    figsize : tuple, optional
        Figure size, by default (8, 5).
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    n_plot = min(len(values), 50)
    components = range(1, n_plot + 1)
    
    # Styling
    color_bar = '#348ABD'
    
    ax.plot(components, values[:n_plot], 'o-', linewidth=2, color=color_bar, 
            markersize=6, markerfacecolor='white', markeredgewidth=2, label=ylabel)
    
    # Cumulative variance
    if np.all(values <= 1.0) and np.sum(values) <= 1.05:
        cumulative = np.cumsum(values[:n_plot])
        ax2 = ax.twinx()
        ax2.plot(components, cumulative, '--', color='gray', alpha=0.7, linewidth=1.5, label='Cumulative')
        ax2.set_ylabel("Cumulative Variance", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color('gray')
        ax2.grid(False)
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Component", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    return fig

def plot_shepard_diagram(X_orig: np.ndarray, 
                         X_emb: np.ndarray, 
                         sample_size: int = 1000,
                         title: str = "Shepard Diagram",
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot Shepard Diagram (Original vs Embedded Distances).

    Parameters
    ----------
    X_orig : np.ndarray
        Original high-dimensional data.
    X_emb : np.ndarray
        Embedded low-dimensional data.
    sample_size : int, optional
        Number of points to sample for distance calculation (to speed up), by default 1000.
    title : str, optional
        Plot title, by default "Shepard Diagram".
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    _set_style()
    dist_high, dist_low = shepard_diagram_data(X_orig, X_emb, sample_size=sample_size)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # Calculate correlation
    if len(dist_high) > 1:
        corr = np.corrcoef(dist_high, dist_low)[0, 1]
    else:
        corr = np.nan
    
    # Heatmap style scatter (hexbin) usually looks science-y
    hb = ax.hexbin(dist_high, dist_low, gridsize=40, cmap='Blues', mincnt=1, edgecolors='none')
    cb = plt.colorbar(hb, ax=ax, label='Density (log scale)')
    
    # Add diagonal line (ideal)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, lw=2.5, label='Ideal', linestyle='--')
    
    ax.set_xlabel("Original Distances", fontweight='bold')
    ax.set_ylabel("Embedded Distances", fontweight='bold')
    ax.set_title(f"{title}\nSpearman Rho: {corr:.3f}", fontsize=14, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    return fig


def plot_streamlines(X_emb: np.ndarray, 
                     V_emb: np.ndarray, 
                     grid_density: int = 25,
                     title: str = "Velocity Streamlines",
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot streamlines of a vector field on the embedding.

    Parameters
    ----------
    X_emb : np.ndarray
        Coordinates of the embedding (2D only).
    V_emb : np.ndarray
        Velocity vectors in the embedding space.
    grid_density : int, optional
        Density of the grid for interpolation, by default 25.
    title : str, optional
        Plot title, by default "Velocity Streamlines".
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    plt.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If X_emb is not 2D.
    """
    _set_style()
    
    if X_emb.shape[1] != 2:
        raise ValueError("Streamlines currently only supported for 2D.")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Create grid
    x_min, x_max = X_emb[:, 0].min(), X_emb[:, 0].max()
    y_min, y_max = X_emb[:, 1].min(), X_emb[:, 1].max()
    
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    x_min -= pad_x; x_max += pad_x
    y_min -= pad_y; y_max += pad_y

    xi = np.linspace(x_min, x_max, grid_density)
    yi = np.linspace(y_min, y_max, grid_density)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate velocities
    from scipy.interpolate import griddata
    
    Ui = griddata(X_emb, V_emb[:, 0], (Xi, Yi), method='linear')
    Vi = griddata(X_emb, V_emb[:, 1], (Xi, Yi), method='linear')
    
    # Speed
    Speed = np.sqrt(Ui**2 + Vi**2)
    
    # Plot background points in muted gray
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c='#DDDDDD', s=15, alpha=0.6, zorder=1)
    
    # Streamlines
    st = ax.streamplot(Xi, Yi, Ui, Vi, color=Speed, cmap='inferno', density=1.5, linewidth=1.2, zorder=2)
    cb = plt.colorbar(st.lines, ax=ax, label='Velocity Magnitude')
    cb.outline.set_visible(False)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Dimension 1", fontweight='bold')
    ax.set_ylabel("Dimension 2", fontweight='bold')
    
    return fig


def plot_comparison(comparison_manager: Any,
                    metric: str = 'trustworthiness',
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot metric comparison curves across different reducers.

    Parameters
    ----------
    comparison_manager : MethodSelector
        The manager containing results_.
    metric : str
        Metric column to plot (e.g. 'trustworthiness', 'lcmc', 'mrre_total').
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    ax : plt.Axes, optional
        Existing axes.

    Returns
    -------
    plt.Figure
    """
    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if not comparison_manager.results_:
        raise ValueError("No results found. Run comparison.run() first.")

    # Plot lines
    for name, df in comparison_manager.results_.items():
        if metric not in df.columns:
            continue
            
        sns.lineplot(data=df, x='k', y=metric, marker='o', label=name, ax=ax, linewidth=2.5)

    ax.set_xlabel("Neighborhood Size (k)", fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
    
    if title is None:
        title = f"Comparison: {metric.replace('_', ' ').title()}"
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(scores: Dict[str, float],
                            title: str = "Feature Importance",
                            top_n: int = 20,
                            figsize: Tuple[int, int] = (8, 6),
                            ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Parameters
    ----------
    scores : dict
        Mapping feature_name -> score.
    title : str
        Plot title.
    top_n : int
        Number of top features to show.
    figsize : tuple
        Figure size.
    ax : plt.Axes
        Existing axes.

    Returns
    -------
    plt.Figure
    """
    _set_style()
    
    # Sort scores
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.barplot(x=values, y=names, ax=ax, palette="magma", orient='h')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Importance Score", fontweight='bold')
    
    return fig


def plot_local_metrics(X_emb: np.ndarray,
                       local_scores: np.ndarray,
                       title: str = "Local Quality Map",
                       cmap: str = 'RdYlGn',
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot the embedding colored by local quality (e.g. point-wise trustworthiness).

    Parameters
    ----------
    X_emb : np.ndarray
        Embedding coordinates (2D).
    local_scores : np.ndarray
        Score per sample.
    title : str
        Plot title.
    cmap : str
        Colormap (Green=Good, Red=Bad).

    Returns
    -------
    plt.Figure
    """
    return plot_embedding(X_emb, labels=local_scores, title=title, cmap=cmap, ax=ax)
