"""
Dim-Reduction Matplotlib Visualization
======================================

Matplotlib plotting helpers for explicit embeddings, tidy evaluation records,
trajectory diagnostics, and interpretation payloads.

The functions in this module accept arrays, mappings, or tidy tables directly.
They do not depend on manager-owned embedding or context state.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from ..dim_reduction.evaluation.geometry import moving_average
from ..dim_reduction.evaluation.metrics import shepard_diagram_data
from . import plotly_utils
from .utils import (
    extract_interpretation_matrix,
    filter_metric_frame,
    filter_metrics,
    infer_metric_plot_type,
    is_categorical,
    prepare_embedding_frame,
    prepare_feature_scores,
    prepare_interpretation_frame,
    prepare_metrics_frame,
)

# --- Style Constants ---
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 18,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


__all__ = [
    "plot_embedding",
    "plot_metrics",
    "plot_loss_history",
    "plot_eigenvalues",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_feature_importance",
    "plot_feature_correlation_heatmap",
    "plot_interpretation",
    "plot_trajectory",
    "plot_trajectory_metric_series",
    "plot_local_metrics",
]


def _set_style(context: str = "paper", style: str = "ticks"):
    """
    Set plotting style for publication-ready aesthetics using Seaborn.

    Parameters
    ----------
    context : str, optional
        Seaborn context (e.g. 'paper', 'notebook', 'talk', 'poster'),
        by default "paper".
    style : str, optional
        Seaborn style (e.g. 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'),
        by default "ticks".
    """
    # Apply matplotlib rcParams base overrides for consistency
    plt.rcParams.update(STYLE_CONFIG)

    # Seaborn overrides
    sns.set_context(context, font_scale=1.2)
    sns.set_style(style, rc=STYLE_CONFIG)
    sns.despine(trim=True, offset=10)


def plot_embedding(
    X_emb: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dims: Union[Tuple[int, int], Tuple[int, int, int]] = (0, 1),
    title: str = "Embedding",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    palette: str = "deep",
    s: int = 40,
    alpha: float = 0.8,
    metrics: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    interactive: bool = False,
    random_state: Optional[int] = None,
) -> Union[plt.Figure, Any]:
    """
    Plot an explicit embedding with optional labels and metadata.

    Parameters
    ----------
    X_emb : np.ndarray
        Embedding array with shape ``(n_samples, n_dimensions)``.
    labels : np.ndarray, optional
        Optional values aligned with the sample axis. Categorical values are
        shown with a legend and continuous values with a colorbar.
    metadata : dict, optional
        Optional column-oriented metadata aligned with the sample axis.
    dims : tuple of int, default=(0, 1)
        Embedding dimensions to visualize. Length must be 2 or 3.
    title : str, default="Embedding"
        Figure title.
    figsize : tuple of int, default=(10, 8)
        Matplotlib figure size for static plots.
    cmap : str, default="viridis"
        Colormap for continuous labels or value overlays.
    palette : str, default="deep"
        Seaborn categorical palette name.
    s : int, default=40
        Marker size.
    alpha : float, default=0.8
        Marker opacity.
    metrics : dict, optional
        Optional scalar metrics to annotate on the figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str, optional
        Optional file path for saving the static figure.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    random_state : int, optional
        Random seed used by the interactive path when sampling is needed.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    Raises
    ------
    ValueError
        If the embedding is not 2D or the requested dimensions are invalid.

    See Also
    --------
    plotly_utils.plot_embedding_interactive
    prepare_embedding_frame
    plot_metrics
    """
    dims = tuple(dims)
    if len(dims) not in {2, 3}:
        raise ValueError("`dims` must contain 2 or 3 embedding dimensions.")

    embedding = np.asarray(X_emb)
    if embedding.ndim != 2:
        raise ValueError("`X_emb` must be a 2D embedding array.")
    if embedding.shape[1] <= max(dims):
        raise ValueError("`dims` must reference valid embedding dimensions.")
    coords = embedding[:, list(dims)]

    if interactive:
        return plotly_utils.plot_embedding_interactive(
            embedding=coords,
            labels=labels,
            metadata=metadata,
            title=title,
            dimensions=len(dims),
            cmap=cmap,
            palette=palette,
            random_state=random_state,
        )

    _set_style()
    frame = prepare_embedding_frame(
        coords,
        labels=labels,
        metadata=metadata,
        dimensions=len(dims),
    )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if len(dims) == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    label_values = frame["Label"] if "Label" in frame.columns else None
    if label_values is not None:
        if is_categorical(label_values):
            if len(dims) == 2:
                sns.scatterplot(
                    data=frame,
                    x="x",
                    y="y",
                    hue="Label",
                    palette=palette,
                    s=s,
                    alpha=alpha,
                    edgecolor="w",
                    linewidth=0.5,
                    ax=ax,
                    legend="full",
                )
                ax.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                    frameon=False,
                    title="Label",
                )
            else:
                unique_labels = frame["Label"].cat.categories.tolist()
                colors = sns.color_palette(palette, len(unique_labels))
                for color, label in zip(colors, unique_labels):
                    mask = frame["Label"] == label
                    ax.scatter(
                        frame.loc[mask, "x"],
                        frame.loc[mask, "y"],
                        frame.loc[mask, "z"],
                        label=str(label),
                        color=color,
                        s=s,
                        alpha=alpha,
                        edgecolors="w",
                        linewidth=0.5,
                    )
                ax.legend(
                    title="Label",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    frameon=False,
                )
        else:
            scatter = (
                ax.scatter(
                    frame["x"],
                    frame["y"],
                    c=label_values,
                    cmap=cmap,
                    s=s,
                    alpha=alpha,
                    edgecolors="none",
                )
                if len(dims) == 2
                else ax.scatter(
                    frame["x"],
                    frame["y"],
                    frame["z"],
                    c=label_values,
                    cmap=cmap,
                    s=s,
                    alpha=alpha,
                    edgecolors="none",
                )
            )
            cbar = plt.colorbar(
                scatter,
                ax=ax,
                pad=0.02 if len(dims) == 2 else 0.1,
                fraction=0.046,
            )
            cbar.set_label("Value", size=12)
            cbar.outline.set_visible(False)
    else:
        color = sns.color_palette()[0]
        if len(dims) == 2:
            ax.scatter(
                frame["x"],
                frame["y"],
                color=color,
                s=s,
                alpha=alpha,
                edgecolors="w",
                linewidth=0.5,
            )
        else:
            ax.scatter(
                frame["x"],
                frame["y"],
                frame["z"],
                color=color,
                s=s,
                alpha=alpha,
                edgecolors="w",
                linewidth=0.5,
            )

    ax.set_xlabel(f"Dimension {dims[0] + 1}", fontweight="bold")
    ax.set_ylabel(f"Dimension {dims[1] + 1}", fontweight="bold")
    if len(dims) == 3:
        ax.set_zlabel(f"Dimension {dims[2] + 1}", fontweight="bold")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_title(title, pad=15, fontweight="bold")

    if metrics:
        clean_metrics = filter_metrics(metrics)
        text_str = "\n".join(
            f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in clean_metrics.items()
        )
        props = dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="#CCCCCC",
        )
        if len(dims) == 2:
            ax.text(
                0.02,
                0.98,
                text_str,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=props,
                zorder=100,
            )
        else:
            plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=props)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
    return fig


def _metric_grouping_roles(metrics_df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Choose x/hue roles for scalar or repeated metric observations."""
    n_methods = metrics_df["method"].nunique()
    n_metrics = metrics_df["metric"].nunique()

    if n_metrics == 1 and n_methods > 1:
        return "method", None
    if n_methods == 1:
        return "metric", None
    return "metric", "method"


def _plot_metric_bars(
    metrics_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    palette: str = "viridis",
    annotate: bool = True,
) -> None:
    """Render scalar metric comparisons as grouped bars."""
    x_col, hue_col = _metric_grouping_roles(metrics_df)

    sns.barplot(
        data=metrics_df,
        x=x_col,
        y="value",
        hue=hue_col,
        estimator=np.mean,
        errorbar=None,
        palette=palette if hue_col is not None else None,
        color=None if hue_col is not None else sns.color_palette(palette, 1)[0],
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    if annotate:
        for patch in ax.patches:
            height = patch.get_height()
            if np.isfinite(height):
                ax.text(
                    patch.get_x() + patch.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#333333",
                )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlabel(x_col.replace("_", " ").title(), fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    if hue_col is not None and ax.legend_ is not None:
        ax.legend(title="Method", frameon=False)


def _plot_metric_distribution(
    metrics_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    plot_type: str,
) -> None:
    """Render repeated observations with distribution-aware plots."""
    x_col, hue_col = _metric_grouping_roles(metrics_df)

    if plot_type == "box":
        sns.boxplot(data=metrics_df, x=x_col, y="value", hue=hue_col, ax=ax)
    elif plot_type == "boxen":
        sns.boxenplot(data=metrics_df, x=x_col, y="value", hue=hue_col, ax=ax)
    else:
        sns.violinplot(
            data=metrics_df,
            x=x_col,
            y="value",
            hue=hue_col,
            inner=None if plot_type == "raincloud" else "box",
            cut=0,
            linewidth=0.8,
            ax=ax,
        )
        if plot_type == "raincloud":
            sns.boxplot(
                data=metrics_df,
                x=x_col,
                y="value",
                hue=hue_col,
                width=0.25,
                showcaps=True,
                boxprops={"facecolor": "white", "zorder": 3},
                showfliers=False,
                whiskerprops={"linewidth": 1},
                ax=ax,
            )
            sns.stripplot(
                data=metrics_df,
                x=x_col,
                y="value",
                hue=hue_col,
                dodge=hue_col is not None,
                jitter=0.18,
                alpha=0.45,
                size=3,
                color="black",
                ax=ax,
            )
        elif plot_type in {"strip", "swarm"}:
            plot_fn = sns.swarmplot if plot_type == "swarm" else sns.stripplot
            plot_fn(
                data=metrics_df,
                x=x_col,
                y="value",
                hue=hue_col,
                dodge=hue_col is not None,
                alpha=0.65,
                size=3,
                color="black",
                ax=ax,
            )

    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        if hue_col is not None and handles:
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), title="Method", frameon=False)
        else:
            ax.legend_.remove()

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlabel(x_col.replace("_", " ").title(), fontweight="bold")
    ax.tick_params(axis="x", rotation=35)


def _plot_metric_heatmap(metrics_df: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    """Render metric comparisons as a heatmap."""
    scope_values = metrics_df["scope_value"].astype(str).nunique()
    if scope_values > 1 and metrics_df["metric"].nunique() == 1:
        heatmap_df = metrics_df.pivot_table(
            index="method", columns="scope_value", values="value", aggfunc="mean"
        )
        x_label = metrics_df["scope"].iloc[0].replace("_", " ").title()
    else:
        heatmap_df = metrics_df.pivot_table(
            index="method", columns="metric", values="value", aggfunc="mean"
        )
        x_label = "Metric"

    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Method", fontweight="bold")


def _plot_metric_lines(metrics_df: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    """Render scope-varying metrics as trajectories with optional variance bands."""
    group_cols = ["method"]
    if metrics_df["metric"].nunique() > 1:
        group_cols.append("metric")

    summary = (
        metrics_df.groupby(group_cols + ["scope", "scope_value"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for keys, sub_df in summary.groupby(group_cols, dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        label = " / ".join(str(k) for k in keys)
        sub_df = sub_df.copy()
        sub_df["scope_numeric"] = pd.to_numeric(sub_df["scope_value"], errors="coerce")
        use_numeric = sub_df["scope_numeric"].notna().all()
        sort_col = "scope_numeric" if use_numeric else "scope_value"
        sub_df = sub_df.sort_values(sort_col)
        x_vals = sub_df["scope_numeric"] if use_numeric else sub_df["scope_value"]

        ax.plot(x_vals, sub_df["mean"], marker="o", linewidth=2.0, label=label)
        if use_numeric and sub_df["count"].max() > 1:
            y_low = sub_df["mean"] - sub_df["std"].fillna(0)
            y_high = sub_df["mean"] + sub_df["std"].fillna(0)
            ax.fill_between(x_vals, y_low, y_high, alpha=0.15)

    scope_label = metrics_df["scope"].iloc[0] if not metrics_df.empty else "scope"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(scope_label.replace("_", " ").title(), fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Series", frameon=False)


def _plot_metric_dumbbell(metrics_df: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    """Render pairwise method deltas per metric."""
    method_means = metrics_df.pivot_table(
        index="metric", columns="method", values="value", aggfunc="mean"
    )
    if method_means.shape[1] != 2:
        raise ValueError("Dumbbell plots require exactly two methods.")

    left_method, right_method = method_means.columns.tolist()
    y_positions = np.arange(len(method_means.index))

    ax.hlines(
        y=y_positions,
        xmin=method_means[left_method],
        xmax=method_means[right_method],
        color="#BBBBBB",
        linewidth=2,
    )
    ax.scatter(method_means[left_method], y_positions, color="#1f77b4", s=60)
    ax.scatter(method_means[right_method], y_positions, color="#ff7f0e", s=60)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(method_means.index)
    ax.set_xlabel("Score", fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend([left_method, right_method], frameon=False)


def plot_metrics(
    scores: Any,
    title: str = "Quality Metrics",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    plot_type: str = "auto",
    metric: Optional[str] = None,
    scope: Optional[str] = None,
    method: Optional[Union[str, Sequence[str]]] = None,
) -> Union[plt.Figure, Any]:
    """
    Plot tidy metric observations using one shared entrypoint.

    Parameters
    ----------
    scores : Any
        Metric mapping, tidy metric frame, list of records, or object exposing
        ``to_frame()``.
    title : str, default="Quality Metrics"
        Figure title.
    figsize : tuple of int, default=(8, 6)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    plot_type : str, default="auto"
        Plot style to use. ``"auto"`` infers an appropriate view from the
        filtered metric records.
    metric : str, optional
        Restrict plotting to one metric.
    scope : str, optional
        Restrict plotting to one scope.
    method : str or sequence of str, optional
        Restrict plotting to one or more methods.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    Raises
    ------
    ValueError
        If no metrics remain after filtering.

    See Also
    --------
    plotly_utils.plot_metric_details
    prepare_metrics_frame
    infer_metric_plot_type
    """
    if interactive:
        return plotly_utils.plot_metric_details(
            scores,
            title=title,
            plot_type=plot_type,
            metric=metric,
            scope=scope,
            method=method,
        )

    metrics_df = filter_metric_frame(
        prepare_metrics_frame(scores),
        metric=metric,
        scope=scope,
        method=method,
    )
    if metrics_df.empty:
        raise ValueError("No scalar metrics found to plot.")

    resolved_plot_type = infer_metric_plot_type(metrics_df, requested=plot_type)
    _set_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if resolved_plot_type in {"bar", "grouped_bar", "lollipop"}:
        _plot_metric_bars(metrics_df, title=title, ax=ax)
    elif resolved_plot_type in {
        "box",
        "boxen",
        "violin",
        "raincloud",
        "strip",
        "swarm",
    }:
        _plot_metric_distribution(
            metrics_df,
            title=title,
            ax=ax,
            plot_type=resolved_plot_type,
        )
    elif resolved_plot_type == "heatmap":
        _plot_metric_heatmap(metrics_df, title=title, ax=ax)
    elif resolved_plot_type == "line":
        _plot_metric_lines(metrics_df, title=title, ax=ax)
    elif resolved_plot_type in {"dumbbell", "slopegraph"}:
        _plot_metric_dumbbell(metrics_df, title=title, ax=ax)
    else:
        raise ValueError(f"Unsupported plot_type: {resolved_plot_type}")

    return fig


def plot_loss_history(
    loss_history: list,
    title: str = "Training Loss",
    figsize: Tuple[int, int] = (8, 5),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
) -> Union[plt.Figure, Any]:
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
    interactive : bool, optional
        If True, returns a Plotly figure.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plotly_utils.plot_loss_history_interactive
    """
    if interactive:
        return plotly_utils.plot_loss_history_interactive(loss_history, title=title)

    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Nice thick line with shadow/marker
    ax.plot(
        loss_history,
        linewidth=2.5,
        color="#E24A33",
        label="Loss",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")

    # Minimalist grid
    ax.grid(True, linestyle="--", alpha=0.3)

    return fig


def plot_eigenvalues(
    values: np.ndarray,
    title: str = "Scree Plot",
    ylabel: str = "Explained Variance",
    figsize: Tuple[int, int] = (8, 5),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
) -> Union[plt.Figure, Any]:
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
    interactive : bool, optional
        If True, returns a Plotly figure.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plotly_utils.plot_scree_interactive
    """
    if interactive:
        # Note: plotly util expects simple 1D array, same as here
        return plotly_utils.plot_scree_interactive(values)

    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_plot = min(len(values), 50)
    components = range(1, n_plot + 1)

    # Styling
    color_bar = "#348ABD"

    ax.plot(
        components,
        values[:n_plot],
        "o-",
        linewidth=2,
        color=color_bar,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label=ylabel,
    )

    # Cumulative variance
    if np.all(values <= 1.0) and np.sum(values) <= 1.05:
        cumulative = np.cumsum(values[:n_plot])
        ax2 = ax.twinx()
        ax2.plot(
            components,
            cumulative,
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Cumulative",
        )
        ax2.set_ylabel("Cumulative Variance", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("gray")
        ax2.grid(False)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Component", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    return fig


def plot_shepard_diagram(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    sample_size: int = 1000,
    title: str = "Shepard Diagram",
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    random_state: Optional[int] = None,
    distances: Optional[Dict[str, np.ndarray]] = None,
) -> Union[plt.Figure, Any]:
    """
    Plot Shepard Diagram (Original vs Embedded Distances).

    Parameters
    ----------
    X_orig : np.ndarray
        Original high-dimensional data.
    X_emb : np.ndarray
        Embedded low-dimensional data.
    sample_size : int, optional
        Number of points to sample for distance calculation (to speed up),
        by default 1000.
    title : str, optional
        Plot title, by default "Shepard Diagram".
    ax : plt.Axes, optional
        Existing axes to plot on.
    interactive : bool, optional
        If True, returns a Plotly figure.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plotly_utils.plot_shepard_interactive
    """
    if interactive:
        return plotly_utils.plot_shepard_interactive(
            X_orig,
            X_emb,
            sample_size=sample_size,
            title=title,
            random_state=random_state,
            distances=distances,
        )

    _set_style()
    if isinstance(distances, dict) and {"original", "embedded"} <= set(distances):
        dist_high = np.asarray(distances["original"])
        dist_low = np.asarray(distances["embedded"])
    else:
        dist_high, dist_low = shepard_diagram_data(
            X_orig, X_emb, sample_size=sample_size, random_state=random_state
        )

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
    hb = ax.hexbin(
        dist_high, dist_low, gridsize=40, cmap="Blues", mincnt=1, edgecolors="none"
    )
    plt.colorbar(hb, ax=ax, label="Density (log scale)")

    # Add diagonal line (ideal)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.8, lw=2.5, label="Ideal")

    ax.set_xlabel("Original Distances", fontweight="bold")
    ax.set_ylabel("Embedded Distances", fontweight="bold")
    ax.set_title(f"{title}\nPearson Corr: {corr:.3f}", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    return fig


def plot_streamlines(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
    title: str = "Velocity Streamlines",
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    random_state: Optional[int] = None,
) -> Union[plt.Figure, Any]:
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
    interactive : bool, optional
        If True, returns a Plotly figure.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    Raises
    ------
    ValueError
        If X_emb is not 2D.

    See Also
    --------
    plotly_utils.plot_streamlines_interactive
    """
    if interactive:
        return plotly_utils.plot_streamlines_interactive(
            X_emb,
            V_emb,
            grid_density=grid_density,
            title=title,
            random_state=random_state,
        )

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
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    xi = np.linspace(x_min, x_max, grid_density)
    yi = np.linspace(y_min, y_max, grid_density)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate velocities
    from scipy.interpolate import griddata

    Ui = griddata(X_emb, V_emb[:, 0], (Xi, Yi), method="linear")
    Vi = griddata(X_emb, V_emb[:, 1], (Xi, Yi), method="linear")

    # Speed
    Speed = np.sqrt(Ui**2 + Vi**2)

    # Plot background points in muted gray
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c="#DDDDDD", s=15, alpha=0.6, zorder=1)

    # Streamlines
    st = ax.streamplot(
        Xi,
        Yi,
        Ui,
        Vi,
        color=Speed,
        cmap="inferno",
        density=1.5,
        linewidth=1.2,
        zorder=2,
    )
    cb = plt.colorbar(st.lines, ax=ax, label="Velocity Magnitude")
    cb.outline.set_visible(False)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontweight="bold")
    ax.set_ylabel("Dimension 2", fontweight="bold")

    return fig


def plot_feature_importance(
    scores: Any,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    analysis: Optional[str] = None,
    method: Optional[str] = None,
    dimension: Optional[str] = None,
) -> Union[plt.Figure, Any]:
    """
    Plot feature-importance scores as a horizontal bar chart.

    Parameters
    ----------
    scores : Any
        Raw ``feature -> score`` mapping, interpretation payload, or
        interpretation record table.
    title : str, default="Feature Importance"
        Figure title.
    top_n : int, default=20
        Maximum number of features to show.
    figsize : tuple of int, default=(8, 6)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    analysis : str, optional
        Interpretation analysis to select when multiple analyses are present.
    method : str, optional
        Method name to select when multiple methods are present.
    dimension : str, optional
        Dimension label to select when the interpretation contains multiple
        dimensions.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    prepare_feature_scores
    plot_interpretation
    plotly_utils.plot_feature_importance_interactive
    """
    if interactive:
        return plotly_utils.plot_feature_importance_interactive(
            scores,
            title=title,
            top_n=top_n,
            analysis=analysis,
            method=method,
            dimension=dimension,
        )
    _set_style()

    feature_scores = prepare_feature_scores(
        scores,
        analysis=analysis,
        method=method,
        dimension=dimension,
    ).head(top_n)
    names = feature_scores.index.astype(str).tolist()
    values = feature_scores.values.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.barplot(
        x=values,
        y=names,
        ax=ax,
        palette="magma",
        orient="h",
        hue=names,
        legend=False,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    return fig


def plot_feature_correlation_heatmap(
    correlations: Any,
    title: str = "Feature Correlation",
    top_n: Optional[int] = 25,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    method: Optional[str] = None,
) -> Union[plt.Figure, Any]:
    """
    Plot feature-to-dimension correlations as a heatmap.

    Parameters
    ----------
    correlations : Any
        Correlation interpretation payload or records.
    title : str, default="Feature Correlation"
        Figure title.
    top_n : int, optional
        Maximum number of features to show. Features are ranked by the maximum
        absolute correlation across dimensions.
    figsize : tuple of int, default=(10, 8)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    method : str, optional
        Method name to select when multiple methods are present.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plot_interpretation
    plotly_utils.plot_feature_correlation_heatmap_interactive
    """
    if interactive:
        return plotly_utils.plot_feature_correlation_heatmap_interactive(
            correlations,
            title=title,
            top_n=top_n,
            method=method,
        )

    frame = prepare_interpretation_frame(correlations)
    frame = frame[frame["analysis"] == "correlation"]
    if method is not None:
        frame = frame[frame["method"] == method]
    elif frame["method"].dropna().nunique() > 1:
        raise ValueError("Specify `method` when multiple methods are present.")
    if frame.empty:
        raise ValueError("No correlation records available to plot.")

    heatmap = frame.pivot_table(
        index="feature",
        columns="dimension",
        values="value",
        aggfunc="mean",
    ).fillna(0.0)
    if top_n is not None and len(heatmap.index) > top_n:
        ranking = heatmap.abs().max(axis=1).sort_values(ascending=False)
        heatmap = heatmap.loc[ranking.head(top_n).index]

    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        heatmap,
        cmap="coolwarm",
        center=0.0,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Dimension", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    return fig


def plot_interpretation(
    interpretation: Any,
    *,
    analysis: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    method: Optional[str] = None,
    dimension: Optional[str] = None,
    top_n: int = 20,
) -> Union[plt.Figure, Any]:
    """
    Plot one interpretation analysis using an appropriate visualization.

    Parameters
    ----------
    interpretation : Any
        Interpretation payload or interpretation records.
    analysis : str
        Interpretation analysis to plot.
    title : str, optional
        Figure title. Defaults to a title derived from ``analysis``.
    figsize : tuple of int, default=(10, 8)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    method : str, optional
        Method name to select when multiple methods are present.
    dimension : str, optional
        Dimension label to select when the interpretation contains multiple
        dimensions.
    top_n : int, default=20
        Maximum number of features to show in bar or heatmap views.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plot_feature_importance
    plot_feature_correlation_heatmap
    plotly_utils.plot_interpretation_interactive
    """
    if interactive:
        return plotly_utils.plot_interpretation_interactive(
            interpretation,
            analysis=analysis,
            title=title,
            method=method,
            dimension=dimension,
            top_n=top_n,
        )

    if analysis == "correlation":
        return plot_feature_correlation_heatmap(
            interpretation,
            title=title or "Feature Correlation",
            top_n=top_n,
            figsize=figsize,
            ax=ax,
            method=method,
        )

    matrix = extract_interpretation_matrix(interpretation, analysis=analysis)
    if matrix is not None:
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            scores = {
                f"Feature {i + 1}": float(value) for i, value in enumerate(matrix)
            }
            return plot_feature_importance(
                scores,
                title=title or analysis.replace("_", " ").title(),
                top_n=top_n,
                figsize=figsize,
                ax=ax,
            )

        _set_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        sns.heatmap(matrix, cmap="magma", ax=ax)
        ax.set_title(
            title or analysis.replace("_", " ").title(),
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Feature Index", fontweight="bold")
        ax.set_ylabel("Feature Axis", fontweight="bold")
        return fig

    return plot_feature_importance(
        interpretation,
        title=title or analysis.replace("_", " ").title(),
        top_n=top_n,
        figsize=figsize,
        ax=ax,
        analysis=analysis,
        method=method,
        dimension=dimension,
    )


def plot_trajectory_metric_series(
    series: Any,
    *,
    times: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "Trajectory Metric",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
) -> Union[plt.Figure, Any]:
    """
    Plot evaluated trajectory metric time series.

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
    figsize : tuple of int, default=(10, 6)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    See Also
    --------
    plot_trajectory
    plotly_utils.plot_trajectory_metric_series_interactive
    """
    if interactive:
        return plotly_utils.plot_trajectory_metric_series_interactive(
            series,
            times=times,
            labels=labels,
            title=title,
            ylabel=ylabel,
        )

    _set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if isinstance(series, dict):
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
            y_vals = np.asarray(values).reshape(-1)
            ax.plot(x_vals, y_vals, label=str(name), linewidth=2.5)
        ax.legend(title="Series", frameon=False)
    else:
        arr = np.asarray(series)
        if arr.ndim == 1:
            x_vals = np.arange(arr.shape[0]) if times is None else np.asarray(times)
            if len(x_vals) != arr.shape[0]:
                raise ValueError("`times` must align with the trajectory time axis.")
            ax.plot(x_vals, arr, linewidth=2.5)
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
                    mean = subset.mean(axis=0)
                    std = subset.std(axis=0) if subset.shape[0] > 1 else None
                    ax.plot(x_vals, mean, linewidth=2.5, label=str(label))
                    if std is not None:
                        ax.fill_between(x_vals, mean - std, mean + std, alpha=0.15)
                ax.legend(title="Label", frameon=False)
            else:
                mean = arr.mean(axis=0)
                std = arr.std(axis=0) if arr.shape[0] > 1 else None
                ax.plot(x_vals, mean, linewidth=2.5)
                if std is not None:
                    ax.fill_between(x_vals, mean - std, mean + std, alpha=0.15)
        else:
            raise ValueError("Trajectory metric series must be 1D, 2D, or a dict.")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_trajectory(
    X: np.ndarray,
    times: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    smooth_window: int = 1,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    interactive: bool = False,
    cmap: str = "viridis",
) -> Union[plt.Figure, Any]:
    """
    Plot already-prepared native trajectory tensors.

    Parameters
    ----------
    X : np.ndarray
        Trajectory tensor with shape ``(n_trajectories, n_times, n_dimensions)``.
    times : np.ndarray, optional
        Explicit time axis aligned with the time dimension.
    values : np.ndarray, optional
        Optional scalar overlay with shape ``(n_trajectories, n_times)``.
    labels : np.ndarray, optional
        Optional label per trajectory.
    smooth_window : int, default=1
        Moving-average window applied independently to each already-valid
        trajectory when greater than 1.
    title : str, default="Trajectory Plot"
        Figure title.
    dimensions : int, default=2
        Number of embedding dimensions to display. Must be 2 or 3.
    figsize : tuple of int, default=(10, 8)
        Matplotlib figure size for static plots.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    interactive : bool, default=False
        If ``True``, return the Plotly equivalent.
    cmap : str, default="viridis"
        Colormap used for scalar overlays.

    Returns
    -------
    matplotlib.figure.Figure or Any
        Matplotlib figure for static plots or the Plotly figure returned by the
        interactive backend.

    Raises
    ------
    ValueError
        If the input is not a native 3D trajectory tensor or if aligned arrays
        do not match the trajectory/time axes.

    See Also
    --------
    plot_trajectory_metric_series
    plotly_utils.plot_trajectory_interactive
    """
    if interactive:
        return plotly_utils.plot_trajectory_interactive(
            X,
            times=times,
            labels=labels,
            values=values,
            title=title,
            dimensions=dimensions,
            smooth_window=smooth_window,
        )

    _set_style()
    trajectories = np.asarray(X)
    if trajectories.ndim != 3:
        raise ValueError(
            "`X` must be a 3D trajectory tensor with shape "
            "(n_trajectories, n_times, n_dimensions)."
        )
    if dimensions not in [2, 3]:
        raise ValueError("Dimensions must be 2 or 3.")

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

    if smooth_window > 1 and n_times >= smooth_window:
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
            values = np.asarray(
                [moving_average(traj_values, smooth_window) for traj_values in values]
            )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if dimensions == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("deep", n_trajectories)
    label_map = None
    if labels is not None:
        unique_labels = list(dict.fromkeys(labels.tolist()))
        colors = sns.color_palette("deep", len(unique_labels))
        label_map = {label: color for label, color in zip(unique_labels, colors)}

    color_values = values if values is not None else None
    if color_values is not None:
        vmin = float(np.nanmin(color_values))
        vmax = float(np.nanmax(color_values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    colorbar_added = False
    for idx, traj in enumerate(trajectories[:, :, :dimensions]):
        line_label = str(labels[idx]) if labels is not None else None
        line_color = (
            label_map[labels[idx]]
            if label_map is not None
            else palette[idx % len(palette)]
        )

        if color_values is not None:
            c_vals = color_values[idx]
            if dimensions == 2:
                from matplotlib.collections import LineCollection

                points = traj[:, :2].reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(c_vals[:-1])
                lc.set_linewidth(2.5)
                lc.set_alpha(0.85)
                ax.add_collection(lc)
                scatter = ax.scatter(
                    traj[:, 0],
                    traj[:, 1],
                    c=c_vals,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    zorder=10,
                )
                if not colorbar_added:
                    plt.colorbar(scatter, ax=ax, label="Value")
                    colorbar_added = True
            else:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#999999", alpha=0.45)
                scatter = ax.scatter(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    c=c_vals,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                )
                if not colorbar_added:
                    plt.colorbar(scatter, ax=ax, label="Value", pad=0.1)
                    colorbar_added = True
        else:
            if dimensions == 2:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    marker="o",
                    linewidth=2.5,
                    alpha=0.85,
                    color=line_color,
                    label=line_label,
                )
            else:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    marker="o",
                    linewidth=2.5,
                    alpha=0.85,
                    color=line_color,
                    label=line_label,
                )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontweight="bold")
    ax.set_ylabel("Dimension 2", fontweight="bold")
    if dimensions == 3:
        ax.set_zlabel("Dimension 3", fontweight="bold")

    if labels is not None and color_values is None and len(np.unique(labels)) > 1:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            dedup = dict(zip(legend_labels, handles))
            ax.legend(dedup.values(), dedup.keys(), title="Label", frameon=False)

    if color_values is None and labels is None:
        ax.legend().remove() if ax.get_legend() else None

    if color_values is not None and dimensions == 2:
        ax.autoscale_view()
    return fig


def plot_local_metrics(
    X_emb: np.ndarray,
    local_scores: np.ndarray,
    title: str = "Local Quality Map",
    cmap: str = "RdYlGn",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
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
    matplotlib.figure.Figure
        Static figure colored by the provided local scores.

    See Also
    --------
    plot_embedding
    """
    return plot_embedding(X_emb, labels=local_scores, title=title, cmap=cmap, ax=ax)
