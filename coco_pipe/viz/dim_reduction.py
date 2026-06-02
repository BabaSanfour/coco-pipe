"""Matplotlib visualization helpers for dimensionality reduction outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import (
    _scalar_metrics,
    coerce_reduction_frame,
    finalize_axes,
    get_figure,
    prepare_component_loadings_frame,
    prepare_eigenvalue_curves,
    prepare_embedding_frame,
    prepare_feature_scores,
    prepare_interpretation_frame,
    prepare_loss_history,
    prepare_metrics_frame,
    prepare_shepard_distances,
    prepare_streamline_grid,
    prepare_streamline_inputs,
    prepare_trajectory_data,
    prepare_trajectory_metric_series,
    prepare_trajectory_separation_series,
    require_non_empty,
    select_dimensions,
    select_reduction_rows,
)
from .base import (
    _colored_line_collection,
    _plot_alpha_encoded_line,
    plot_bar,
    plot_heatmap,
    plot_hexbin,
    plot_line,
    plot_scatter2d,
    plot_scatter3d,
    plot_streamfield,
)
from .theme import DIVERGING, SEQUENTIAL, ColorKind, coco_theme


def _embedding_metric_text(metrics: dict[str, Any] | None, metric_name: str) -> str:
    """Return compact metric annotation text for an embedding panel."""
    if not metrics:
        return ""
    frame = prepare_metrics_frame(metrics)
    frame = frame[frame["Metric"].astype(str) == str(metric_name)]
    if frame.empty:
        raise ValueError(f"Metric {metric_name!r} was not found in `metrics`.")
    return "\n".join(f"{row.Metric}: {row.Value:.3f}" for row in frame.itertuples())


def plot_embedding(
    X_emb: np.ndarray,
    labels: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    dims: tuple[int, int] | tuple[int, int, int] = (0, 1),
    title: str = "Embedding",
    figsize: tuple[float, float] | None = (10, 8),
    cmap: str | None = None,
    palette: str = "deep",
    s: int = 40,
    alpha: float = 0.8,
    label_kind: ColorKind = "categorical",
    metrics: dict[str, Any] | None = None,
    metric_name: str | None = None,
    ax: plt.Axes | None = None,
    random_state: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an explicit 2D or 3D embedding.

    Parameters
    ----------
    X_emb
        Embedding array with shape ``(n_samples, n_dimensions)``.
    labels
        Optional label array aligned with samples used for color encoding.
    metadata
        Optional column-oriented metadata aligned with samples.
    dims
        Column indices to use as plot axes. Two indices produce a 2D plot;
        three produce a 3D plot.
    title
        Axes title.
    figsize
        Figure size used when creating new axes.
    cmap
        Colormap used when ``label_kind="continuous"``.
    palette
        Seaborn palette name used when ``label_kind="categorical"``.
    s
        Scatter marker size.
    alpha
        Scatter point opacity.
    label_kind
        ``"categorical"`` to color by class, ``"continuous"`` to apply a
        colormap to numeric labels.
    metrics
        Optional metrics mapping used to annotate the plot when
        ``metric_name`` is provided.
    metric_name
        Name of the metric from ``metrics`` to display as an annotation.
    ax
        Existing Matplotlib axes to draw into.
    random_state
        Accepted for API compatibility; not used internally.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_embedding : Interactive Plotly version.
    plot_metrics : Quality metric overview for the embedding run.
    plot_shepard_diagram : Distance-preservation diagnostic.
    plot_eigenvalues : Explained variance for linear reducers.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> X_emb = rng.normal(size=(50, 2))
    >>> labels = np.arange(50) % 5
    >>> fig, ax = viz.plot_embedding(X_emb, labels=labels)
    """
    dims = tuple(dims)
    coords = select_dimensions(X_emb, dims, context="X_emb")
    cmap = cmap or SEQUENTIAL
    _ = random_state

    with coco_theme():
        frame = prepare_embedding_frame(
            coords,
            labels=labels,
            metadata=metadata,
            dimensions=len(dims),
            label_kind=label_kind,
        )
        require_non_empty(frame, "embedding")
        fig, ax = get_figure(
            ax, figsize, (10, 8), projection="3d" if len(dims) == 3 else None
        )
        label_values = frame["Label"] if "Label" in frame.columns else None
        scatter_kwargs: dict[str, Any] = {"s": s, "alpha": alpha, "ax": ax}
        if label_values is not None and label_kind == "categorical":
            categories = frame["Label"].cat.categories.tolist()
            scatter_kwargs.update(
                labels=frame["Label"],
                palette=sns.color_palette(palette, len(categories)),
                legend_title="Label",
            )
        elif label_values is not None:
            scatter_kwargs.update(
                c=label_values,
                cmap=cmap,
                colorbar=True,
                colorbar_label="Value",
            )
        else:
            scatter_kwargs["color"] = sns.color_palette(palette, 1)[0]

        if len(dims) == 2:
            fig, ax = plot_scatter2d(frame["x"], frame["y"], **scatter_kwargs)
        else:
            fig, ax = plot_scatter3d(
                frame["x"], frame["y"], frame["z"], **scatter_kwargs
            )
        finalize_axes(
            ax,
            title=title,
            xlabel=f"Dimension {dims[0] + 1}",
            ylabel=f"Dimension {dims[1] + 1}",
            zlabel=f"Dimension {dims[2] + 1}" if len(dims) == 3 else None,
            tick_nbins=5,
        )
        metric_text = (
            _embedding_metric_text(metrics, metric_name)
            if metric_name is not None
            else ""
        )
        if metric_text and len(dims) == 2:
            ax.text(0.02, 0.98, metric_text, transform=ax.transAxes, va="top")
        return fig, ax


def _metric_x_hue_columns(metrics_df: pd.DataFrame) -> tuple[str, str | None]:
    n_methods = metrics_df["Method"].nunique()
    n_metrics = metrics_df["Metric"].nunique()
    if n_metrics == 1 and n_methods > 1:
        return "Method", None
    if n_methods == 1:
        return "Metric", None
    return "Metric", "Method"


def _plot_metric_bars(
    metrics_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    palette: str = SEQUENTIAL,
    annotate: bool = True,
    axes_kws: dict | None = None,
) -> None:
    x_col, hue_col = _metric_x_hue_columns(metrics_df)
    sns.barplot(
        data=metrics_df,
        x=x_col,
        y="Value",
        hue=hue_col,
        estimator=np.mean,
        errorbar=None,
        palette=palette if hue_col is not None else None,
        color=None if hue_col is not None else sns.color_palette(palette, 1)[0],
        ax=ax,
    )
    if annotate:
        for patch in ax.patches:
            height = patch.get_height()
            if np.isfinite(height):
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    finalize_axes(
        ax,
        title=title,
        xlabel=x_col.replace("_", " ").title(),
        ylabel="Score",
        xtick_rotation=35,
        **(axes_kws or {}),
    )
    if hue_col is not None and ax.legend_ is not None:
        ax.legend(title="Method", frameon=False)


def _plot_metric_distribution(
    metrics_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    plot_type: str,
    axes_kws: dict | None = None,
) -> None:
    x_col, hue_col = _metric_x_hue_columns(metrics_df)
    if plot_type == "box":
        sns.boxplot(data=metrics_df, x=x_col, y="Value", hue=hue_col, ax=ax)
    elif plot_type == "boxen":
        sns.boxenplot(data=metrics_df, x=x_col, y="Value", hue=hue_col, ax=ax)
    else:
        sns.violinplot(data=metrics_df, x=x_col, y="Value", hue=hue_col, cut=0, ax=ax)
        if plot_type in {"strip", "swarm", "raincloud"}:
            plot_fn = sns.swarmplot if plot_type == "swarm" else sns.stripplot
            plot_fn(
                data=metrics_df,
                x=x_col,
                y="Value",
                hue=hue_col,
                dodge=hue_col is not None,
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
    finalize_axes(
        ax,
        title=title,
        xlabel=x_col.replace("_", " ").title(),
        ylabel="Score",
        xtick_rotation=35,
        **(axes_kws or {}),
    )


def _plot_metric_heatmap(
    metrics_df: pd.DataFrame, title: str, ax: plt.Axes, axes_kws: dict | None = None
) -> None:
    has_scope_axis = metrics_df["ScopeValue"].astype(str).nunique() > 1
    has_single_metric = metrics_df["Metric"].nunique() == 1
    if has_scope_axis and has_single_metric:
        heatmap_df = metrics_df.pivot_table(
            index="Method", columns="ScopeValue", values="Value", aggfunc="mean"
        )
        x_label = str(metrics_df["Scope"].iloc[0]).replace("_", " ").title()
    else:
        heatmap_df = metrics_df.pivot_table(
            index="Method", columns="Metric", values="Value", aggfunc="mean"
        )
        x_label = "Metric"
    plot_heatmap(
        heatmap_df,
        cmap=SEQUENTIAL,
        annotate=True,
        annotation_format=".3f",
        colorbar_label="Score",
        title=title,
        xlabel=x_label,
        ylabel="Method",
        ax=ax,
        **(axes_kws or {}),
    )


def _plot_metric_lines(
    metrics_df: pd.DataFrame, title: str, ax: plt.Axes, axes_kws: dict | None = None
) -> None:
    group_cols = ["Method"] + (["Metric"] if metrics_df["Metric"].nunique() > 1 else [])
    summary = (
        metrics_df.groupby(group_cols + ["Scope", "ScopeValue"], dropna=False)["Value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for keys, sub_df in summary.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = " / ".join(str(k) for k in keys)
        sub_df = sub_df.copy()
        sub_df["scope_numeric"] = pd.to_numeric(sub_df["ScopeValue"], errors="coerce")
        use_numeric = sub_df["scope_numeric"].notna().all()
        sub_df = sub_df.sort_values("scope_numeric" if use_numeric else "ScopeValue")
        x_vals = sub_df["scope_numeric"] if use_numeric else sub_df["ScopeValue"]
        yerr = (
            sub_df["std"].fillna(0)
            if use_numeric and sub_df["count"].max() > 1
            else None
        )
        plot_line(
            x_vals,
            sub_df["mean"],
            yerr=yerr,
            marker="o",
            label=label,
            ax=ax,
        )
    finalize_axes(
        ax,
        title=title,
        xlabel=str(metrics_df["Scope"].iloc[0]).replace("_", " ").title(),
        ylabel="Score",
        legend=True,
        legend_title="Series",
        **(axes_kws or {}),
    )


def _plot_metric_dumbbell(
    metrics_df: pd.DataFrame, title: str, ax: plt.Axes, axes_kws: dict | None = None
) -> None:
    method_means = metrics_df.pivot_table(
        index="Metric", columns="Method", values="Value", aggfunc="mean"
    )
    if method_means.shape[1] != 2:
        raise ValueError("Dumbbell plots require exactly two methods.")
    left, right = method_means.columns.tolist()
    y_pos = np.arange(len(method_means.index))
    ax.hlines(y_pos, method_means[left], method_means[right], color="0.7", linewidth=2)
    left_points = ax.scatter(method_means[left], y_pos, color="#1f77b4", s=60)
    right_points = ax.scatter(method_means[right], y_pos, color="#ff7f0e", s=60)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_means.index)
    finalize_axes(ax, title=title, xlabel="Score", **(axes_kws or {}))
    ax.legend([left_points, right_points], [str(left), str(right)], frameon=False)


def plot_metrics(
    scores: Any,
    title: str = "Quality Metrics",
    figsize: tuple[float, float] | None = (8, 6),
    ax: plt.Axes | None = None,
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
    metric: str | None = None,
    scope: str | None = None,
    method: str | Sequence[str] | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot tidy metric observations using one shared entrypoint.

    Parameters
    ----------
    scores
        Metric source: a tidy DataFrame, ``{metric: value}`` mapping, list of
        records, or any object exposing ``to_frame()`` or ``metrics_``.
    title
        Axes title.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.
    plot_type
        Visualization style. ``"bar"`` / ``"grouped_bar"`` / ``"lollipop"``
        aggregate to global scalars; ``"box"`` / ``"boxen"`` / ``"violin"`` /
        ``"raincloud"`` / ``"strip"`` / ``"swarm"`` show per-observation
        distributions; ``"heatmap"`` produces a method × metric grid;
        ``"line"`` plots metrics across a numeric scope axis; ``"dumbbell"``
        / ``"slopegraph"`` require exactly two methods.
    metric
        Optional metric name used to filter rows before plotting.
    scope
        Optional scope name used to filter rows before plotting.
    method
        Optional method name or names used to filter rows before plotting.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_metrics : Interactive Plotly version.
    plot_embedding : 2D or 3D scatter of the embedding points.
    plot_eigenvalues : Per-component explained variance.
    plot_coranking_matrix : Neighbourhood-preservation co-ranking heatmap.

    Examples
    --------
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> fig, ax = viz.plot_metrics({"trustworthiness": 0.92, "continuity": 0.88})
    """
    with coco_theme():
        raw_metric_input = isinstance(
            scores, (pd.DataFrame, Mapping, list)
        ) or callable(getattr(scores, "to_frame", None))
        frame = coerce_reduction_frame(
            scores,
            accessor=None if raw_metric_input else "metrics_",
            prepare_fn=prepare_metrics_frame,
        )
        require_non_empty(frame, "metrics")
        metrics_df = select_reduction_rows(
            frame, method=method, metric=metric, scope=scope
        )
        require_non_empty(metrics_df, "metrics after filtering")

        if plot_type in {"bar", "grouped_bar", "dumbbell"}:
            metrics_df = _scalar_metrics(metrics_df, "scalar metrics plot")

        fig, ax = get_figure(ax, figsize, (8, 6))
        if plot_type in {"bar", "grouped_bar", "lollipop"}:
            _plot_metric_bars(metrics_df, title, ax, axes_kws=axes_kws)
        elif plot_type in {"box", "boxen", "violin", "raincloud", "strip", "swarm"}:
            _plot_metric_distribution(
                metrics_df, title, ax, plot_type, axes_kws=axes_kws
            )
        elif plot_type == "heatmap":
            _plot_metric_heatmap(metrics_df, title, ax, axes_kws=axes_kws)
        elif plot_type == "line":
            _plot_metric_lines(metrics_df, title, ax, axes_kws=axes_kws)
        elif plot_type in {"dumbbell", "slopegraph"}:
            _plot_metric_dumbbell(metrics_df, title, ax, axes_kws=axes_kws)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")
        return fig, ax


def plot_loss_history(
    loss_history: Sequence[float] | np.ndarray,
    title: str = "Training Loss",
    figsize: tuple[float, float] | None = (8, 5),
    ax: plt.Axes | None = None,
    scope: str | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot reducer loss history. Linear reducers usually do not expose this.

    Parameters
    ----------
    loss_history
        Sequence of per-epoch loss values from an iterative reducer.
    title
        Axes title.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.
    scope
        Optional scope tag, accepted as ``"train"`` or ``"val"``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    Raises
    ------
    ValueError
        If ``loss_history`` is empty or ``scope`` is not ``"train"`` or
        ``"val"``.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_loss_history :
        Interactive Plotly version.
    plot_eigenvalues : Explained variance for linear reducers.
    plot_metrics : Scalar quality metric overview.

    Examples
    --------
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> fig, ax = viz.plot_loss_history([1.0, 0.7, 0.4, 0.25, 0.18])
    """
    losses = prepare_loss_history(loss_history, scope=scope)
    with coco_theme():
        x_vals = np.arange(losses.size)
        fig, ax = plot_line(
            x_vals,
            losses,
            linewidth=2.5,
            color="#E24A33",
            marker="o",
            label="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            title=title,
            ax=ax,
            figsize=figsize or (8, 5),
            **(axes_kws or {}),
        )
        return fig, ax


def plot_eigenvalues(
    values: dict[str, np.ndarray],
    title: str = "Scree Plot",
    ylabel: str = "Explained Variance",
    figsize: tuple[float, float] | None = (8, 5),
    ax: plt.Axes | None = None,
    max_components: int | None = None,
    condition_colors: dict[str, str] | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot explained variance curves from linear reducers (PCA, TruncatedSVD).

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Mapping of label → array. Array shapes:

        - 1-D ``(n_pcs,)`` — pre-averaged curve, no SEM band.
        - 2-D ``(n_subjects, n_pcs)`` — per-subject data; mean ± SEM band drawn.
    max_components : int, optional
        Cap the number of components shown.
    condition_colors : dict[str, str], optional
        Per-label hex colour overrides.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_eigenvalues :
        Interactive Plotly version.
    plot_metrics : Scalar quality metric overview.
    plot_loss_history : Iterative training loss curve.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> evals = {"PCA": np.array([0.50, 0.30, 0.12, 0.05, 0.03])}
    >>> fig, ax = viz.plot_eigenvalues(evals)
    """
    curves = prepare_eigenvalue_curves(values, max_components=max_components)
    condition_colors = condition_colors or {}
    conditions = [curve["label"] for curve in curves]
    palette = sns.color_palette("deep", len(conditions))
    color_cycle = {
        c: condition_colors.get(c, palette[i % len(palette)])
        for i, c in enumerate(conditions)
    }
    with coco_theme():
        fig, cur_ax = get_figure(ax, figsize, (8, 5))
        for curve in curves:
            condition = curve["label"]
            plot_line(
                curve["components"],
                curve["mean"],
                yerr=curve["sem"],
                error_style="band",
                marker="o",
                linewidth=1.8,
                label=condition,
                color=color_cycle[condition],
                ax=cur_ax,
            )
        finalize_axes(
            cur_ax,
            title=title,
            xlabel="Component",
            ylabel=ylabel,
            legend=True,
            **(axes_kws or {}),
        )
        return fig, cur_ax


def plot_shepard_diagram(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    sample_size: int = 1000,
    title: str = "Shepard Diagram",
    ax: plt.Axes | None = None,
    random_state: int | None = None,
    distances: dict[str, np.ndarray] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot original vs embedded pairwise distances.

    Parameters
    ----------
    X_orig
        Original high-dimensional data with shape ``(n_samples, n_features)``.
        Ignored when ``distances`` is provided.
    X_emb
        Low-dimensional embedding with shape ``(n_samples, n_dims)``.
        Ignored when ``distances`` is provided.
    sample_size
        Number of point pairs to sample for distance computation.
    title
        Axes title.
    ax
        Existing Matplotlib axes to draw into.
    random_state
        Random seed for reproducible distance sampling.
    distances
        Pre-computed distance dict with ``"original"`` and ``"embedded"``
        keys. When both keys are present, ``X_orig`` and ``X_emb`` are not
        used.
    figsize
        Figure size used when creating new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_shepard_diagram :
        Interactive Plotly version.
    plot_embedding : Scatter of the low-dimensional embedding.
    plot_coranking_matrix : Neighbourhood-rank preservation heatmap.
    plot_metrics : Scalar quality metrics such as trustworthiness.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> X_orig = rng.normal(size=(60, 10))
    >>> X_emb = rng.normal(size=(60, 2))
    >>> fig, ax = viz.plot_shepard_diagram(X_orig, X_emb, sample_size=200)
    """
    dist_high, dist_low, corr = prepare_shepard_distances(
        X_orig,
        X_emb,
        sample_size=sample_size,
        random_state=random_state,
        distances=distances,
    )
    with coco_theme():
        fig, ax = plot_hexbin(
            dist_high,
            dist_low,
            gridsize=40,
            cmap=SEQUENTIAL,
            mincnt=1,
            colorbar_label="Density",
            reference_identity=True,
            title=f"{title}\nPearson Corr: {corr:.3f}",
            xlabel="Original Distances",
            ylabel="Embedded Distances",
            legend=True,
            ax=ax,
            figsize=figsize or (8, 6),
        )
        return fig, ax


def plot_streamlines(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
    title: str = "Velocity Streamlines",
    ax: plt.Axes | None = None,
    random_state: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a velocity field on a 2D embedding.

    Parameters
    ----------
    X_emb
        2D embedding coordinates with shape ``(n_samples, 2)``.
    V_emb
        Velocity vectors with the same shape as ``X_emb``, as returned by
        ``coco_pipe.dim_reduction.evaluation.velocity.compute_velocity_fields``.
    grid_density
        Number of grid points per axis used for velocity interpolation.
    title
        Axes title.
    ax
        Existing Matplotlib axes to draw into.
    random_state
        Accepted for API compatibility; not used internally.
    figsize
        Figure size used when creating new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_streamlines :
        Interactive Plotly version.
    plot_embedding : Scatter of the underlying embedding points.
    plot_trajectory : Plotted trajectory paths over the embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> X_emb = rng.normal(size=(80, 2))
    >>> V_emb = rng.normal(size=(80, 2)) * 0.2
    >>> fig, ax = viz.plot_streamlines(X_emb, V_emb)
    """
    X_emb, V_emb = prepare_streamline_inputs(X_emb, V_emb)
    Xi, Yi, Ui, Vi = prepare_streamline_grid(X_emb, V_emb, grid_density=grid_density)
    _ = random_state
    with coco_theme():
        fig, ax = plot_streamfield(
            Xi,
            Yi,
            Ui,
            Vi,
            points=X_emb,
            cmap=SEQUENTIAL,
            colorbar_label="Velocity Magnitude",
            title=title,
            xlabel="Dimension 1",
            ylabel="Dimension 2",
            ax=ax,
            figsize=figsize or (10, 8),
        )
        return fig, ax


def plot_feature_importance(
    scores: Any,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: tuple[float, float] | None = (8, 6),
    ax: plt.Axes | None = None,
    analysis: str | None = None,
    method: str | None = None,
    dimension: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot feature-importance scores as horizontal bars.

    Parameters
    ----------
    scores
        Feature-score source: a raw ``{feature: score}`` mapping,
        interpretation payload, tidy interpretation DataFrame, or any object
        exposing ``interpretation_``.
    title
        Axes title.
    top_n
        Maximum number of features to display, ranked by score magnitude.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.
    analysis
        Interpretation analysis to select when multiple analyses are present.
    method
        Method name to select when multiple methods are present.
    dimension
        Dimension label to select when multiple dimensions are present.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_feature_importance :
        Interactive Plotly version.
    plot_feature_correlation_heatmap : Feature-to-dimension correlation heatmap.
    plot_component_loadings : Linear-reducer component loading matrix.

    Examples
    --------
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> scores = {"alpha": 0.72, "beta": 0.55, "gamma": 0.31}
    >>> fig, ax = viz.plot_feature_importance(scores)
    """
    with coco_theme():
        frame = coerce_reduction_frame(
            scores, accessor="interpretation_", prepare_fn=prepare_interpretation_frame
        )
        require_non_empty(frame, "feature importance")
        frame = select_reduction_rows(frame, analysis=analysis, method=method)
        require_non_empty(frame, "feature importance after filtering")

        feature_scores = prepare_feature_scores(
            frame, analysis=analysis, method=method, dimension=dimension
        ).head(top_n)
        if feature_scores.empty:
            raise ValueError("No feature scores available to plot.")

        plot_scores = feature_scores.sort_values()
        fig, ax = plot_bar(
            plot_scores,
            cmap=SEQUENTIAL,
            orientation="horizontal",
            ax=ax,
            figsize=figsize or (8, max(4, len(plot_scores) * 0.28)),
            xlabel="Importance Score",
            title=title,
        )
        return fig, ax


def plot_feature_correlation_heatmap(
    correlations: Any,
    title: str = "Feature Correlation",
    top_n: int | None = 25,
    figsize: tuple[float, float] | None = (10, 8),
    ax: plt.Axes | None = None,
    method: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot feature-to-dimension correlations as a heatmap.

    Parameters
    ----------
    correlations
        Correlation source: a raw correlation payload, tidy interpretation
        DataFrame, or any object exposing ``interpretation_``.
    title
        Axes title.
    top_n
        Maximum number of features to show, ranked by peak absolute
        correlation across dimensions.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.
    method
        Method name to select when multiple methods are present.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_feature_correlation_heatmap :
        Interactive Plotly version.
    plot_feature_importance : Ranked feature-importance bar chart.
    plot_component_loadings : Linear-reducer component loading matrix.

    Examples
    --------
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> payload = {"correlation": {"D1": {"F1": 0.6,
    ...                           "F2": -0.3},
    ...                           "D2": {"F1": 0.1,
    ...                           "F2": 0.8}}}
    >>> fig, ax = viz.plot_feature_correlation_heatmap(payload)
    """
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be a positive integer or None.")

    with coco_theme():
        frame = coerce_reduction_frame(
            correlations,
            accessor="interpretation_",
            prepare_fn=prepare_interpretation_frame,
        )
        require_non_empty(frame, "feature correlation")
        frame = select_reduction_rows(frame, method=method, analysis="correlation")
        require_non_empty(frame, "feature correlation after filtering")

        if frame["Method"].dropna().nunique() > 1:
            raise ValueError("Specify `method` when multiple methods are present.")

        heatmap = frame.pivot_table(
            index="Feature", columns="Dimension", values="Value", aggfunc="mean"
        ).fillna(0)
        if top_n is not None and len(heatmap.index) > top_n:
            ranking = heatmap.abs().max(axis=1).sort_values(ascending=False)
            heatmap = heatmap.loc[ranking.head(top_n).index]

        fig, ax = plot_heatmap(
            heatmap,
            cmap=DIVERGING,
            center=0.0,
            ax=ax,
            figsize=figsize or (10, 8),
            colorbar_label="Correlation",
            title=title,
            xlabel="Dimension",
            ylabel="Feature",
        )
        return fig, ax


def plot_trajectory_metric_series(
    series: Any,
    times: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    smooth_window: int = 1,
    title: str = "Trajectory Metric",
    ylabel: str = "Value",
    figsize: tuple[float, float] | None = (10, 6),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot evaluated trajectory metric time series.

    Parameters
    ----------
    series
        Metric values: a 1D array, 2D ``(trajectory, time)`` array, or a
        ``{name: timecourse}`` mapping. 2D arrays are averaged across
        trajectories per unique label.
    times
        Explicit time axis aligned with the time dimension.
    labels
        Trajectory labels aligned with the first axis of 2D inputs.
    title
        Axes title.
    ylabel
        Y-axis label.
    figsize
        Figure size used when creating new axes.
    ax
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_trajectory_metric_series :
        Interactive Plotly version.
    plot_trajectory : Raw trajectory paths in embedding space.
    plot_trajectory_separation : Pairwise label-separation timecourses.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> series = rng.normal(size=(3, 20))
    >>> fig, ax = viz.plot_trajectory_metric_series(series)
    """
    frame = prepare_trajectory_metric_series(series, times=times, labels=labels)
    with coco_theme():
        fig, ax = get_figure(ax, figsize, (10, 6))
        groups = list(frame.groupby("Series", sort=False))
        for name, group in groups:
            errors = group["Error"].to_numpy(dtype=float)
            yerr = None if np.isnan(errors).all() else errors
            y_vals = group["Value"].to_numpy(dtype=float)

            if smooth_window > 1:
                import pandas as pd

                y_vals = (
                    pd.Series(y_vals)
                    .rolling(window=smooth_window, min_periods=1, center=True)
                    .mean()
                    .values
                )
                if yerr is not None:
                    yerr = (
                        pd.Series(yerr)
                        .rolling(window=smooth_window, min_periods=1, center=True)
                        .mean()
                        .values
                    )

            color = color_map.get(name) if color_map else None
            raw_style = linestyle_map.get(name, "-") if linestyle_map else "-"
            _style_mapper = {"dash": "--", "solid": "-", "dot": ":", "dashdot": "-."}
            line_style = _style_mapper.get(raw_style, raw_style)

            fig, ax = plot_line(
                group["Time"].to_numpy(),
                y_vals,
                yerr=yerr,
                label=str(name) if len(groups) > 1 else None,
                linewidth=2,
                color=color,
                linestyle=line_style,
                ax=ax,
            )
        legend_title = "Series" if isinstance(series, Mapping) else "Label"
        finalize_axes(
            ax,
            title=title,
            xlabel="Time",
            ylabel=ylabel,
            legend=len(groups) > 1,
            legend_title=legend_title,
        )
        return fig, ax


def plot_trajectory(
    X: np.ndarray,
    times: np.ndarray | None = None,
    values: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    smooth_window: int = 1,
    downsample: int = 1,
    speed_mode: Literal["linecollection", "alpha"] = "linecollection",
    add_start_end_markers: bool = False,
    show_markers: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    linewidth: float = 2.5,
    title: str = "Trajectory Plot",
    dimensions: int = 2,
    figsize: tuple[float, float] | None = (10, 8),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    axis_labels: list[str] | None = None,
    axes_kws: dict | None = None,
    showlegend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot prepared trajectory tensors of shape ``trajectory x time x dim``.

    Parameters
    ----------
    X : np.ndarray of shape (n_trajectories, n_times, n_dims)
        Each trajectory along the first axis — can be individual subjects,
        pre-averaged conditions, or any other grouping.
    times : np.ndarray, optional
        Time stamps for the time axis.  Defaults to integer indices.
    values : np.ndarray of shape (n_trajectories, n_times), optional
        Per-point scalar values (e.g. speed) used for colour encoding.
    labels : array-like of length n_trajectories, optional
        Label per trajectory used for colouring and legend.
    color_map : dict[str, str], optional
        Optional mapping of label to hex color string.
    linestyle_map : dict[str, str], optional
        Optional mapping of label to Matplotlib linestyle string.
    smooth_window : int, default=1
        Moving-average window applied before plotting.
    downsample : int, default=1
        Keep every ``downsample``-th time point.
    speed_mode : {"linecollection", "alpha"}, default="linecollection"
        Colour-encoding style when ``values`` is provided (2D only).
        ``"linecollection"`` colours each segment by value;
        ``"alpha"`` modulates transparency and lightness of the base colour.
    add_start_end_markers : bool, default=False
        Draw a circle (●) at the start and a cross (✕) at the end of each
        trajectory instead of a marker on every point.
    show_markers : bool, default=True
        If True, draws markers at each sampled time point unless
        ``add_start_end_markers`` is True.
    xlim, ylim : tuple[float, float], optional
        Fixed axis limits.  Auto-scaled when ``None``.
    linewidth : float, default=2.5
    title : str, default="Trajectory Plot"
    dimensions : int, default=2
        Number of spatial dimensions to render (2 or 3).
    figsize : tuple[float, float], optional
    ax : matplotlib.axes.Axes, optional
    cmap : str, optional
        Colormap name for value encoding.  Defaults to the theme sequential map.
    axis_labels : list[str], optional
        Custom axis labels (e.g. ``["PC1", "PC2"]``).
    axes_kws : dict, optional
        Additional kwargs passed to ``ax.set()``.
    showlegend : bool, default=True
        Whether to draw the legend.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_trajectory :
        Interactive Plotly version.
    plot_trajectory_separation : Pairwise label-separation timecourses.
    plot_trajectory_metric_series : Metric time series for trajectories.
    plot_streamlines : Velocity field overlay on the embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(3, 20, 2))
    >>> labels = np.array(["A", "B", "C"])
    >>> fig, ax = viz.plot_trajectory(X, labels=labels)
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
    n_trajectories = trajectories.shape[0]
    cmap = cmap or SEQUENTIAL

    with coco_theme():
        fig, ax = get_figure(
            ax, figsize, (10, 8), projection="3d" if dimensions == 3 else None
        )
        palette = sns.color_palette("deep", n_trajectories)
        label_colors = None
        if labels is not None:
            if color_map is not None:
                label_colors = color_map
            else:
                unique = list(dict.fromkeys(np.asarray(labels).tolist()))
                colors = sns.color_palette("deep", len(unique))
                label_colors = dict(zip(unique, colors))

        norm = None
        colorbar_added = False
        if values is not None:
            values = np.asarray(values)
            norm = plt.Normalize(
                vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values))
            )

        for idx, traj in enumerate(trajectories[:, :, :dimensions]):
            line_label = str(labels[idx]) if labels is not None else None
            line_color = (
                label_colors.get(labels[idx], palette[idx % len(palette)])
                if label_colors is not None
                else palette[idx % len(palette)]
            )
            raw_style = (
                linestyle_map.get(labels[idx], "-")
                if linestyle_map is not None and labels is not None
                else "-"
            )

            # Map Plotly string styles to Matplotlib line styles seamlessly
            _style_mapper = {"dash": "--", "solid": "-", "dot": ":", "dashdot": "-."}
            line_style = _style_mapper.get(raw_style, raw_style)

            if values is not None:
                c_vals = values[idx]
                if dimensions == 2:
                    if speed_mode == "alpha":
                        _plot_alpha_encoded_line(
                            ax,
                            traj[:, 0],
                            traj[:, 1],
                            c_vals,
                            base_color=line_color,
                            label=line_label,
                            linewidth=linewidth,
                            linestyle=line_style,
                        )
                    else:
                        lc = _colored_line_collection(
                            traj[:, 0],
                            traj[:, 1],
                            c_vals,
                            cmap,
                            linewidth,
                            norm=norm,
                            linestyle=line_style,
                        )
                        ax.add_collection(lc)
                        ax.plot(
                            [],
                            [],
                            color=line_color,
                            linewidth=linewidth,
                            linestyle=line_style,
                            label=line_label,
                        )
                        if not colorbar_added:
                            fig.colorbar(lc, ax=ax, label="Value", pad=0.02)
                            colorbar_added = True
                else:
                    ax.plot(
                        traj[:, 0],
                        traj[:, 1],
                        traj[:, 2],
                        color="0.6",
                        alpha=0.45,
                        linestyle=line_style,
                    )
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
                        fig.colorbar(scatter, ax=ax, label="Value", pad=0.1)
                        colorbar_added = True
            elif dimensions == 2:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    marker="o" if show_markers and not add_start_end_markers else None,
                    linewidth=linewidth,
                    color=line_color,
                    linestyle=line_style,
                    label=line_label,
                )
            else:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    traj[:, 2],
                    marker="o" if show_markers and not add_start_end_markers else None,
                    linewidth=linewidth,
                    color=line_color,
                    linestyle=line_style,
                    label=line_label,
                )

            if add_start_end_markers and dimensions == 2:
                ax.scatter(
                    traj[0, 0], traj[0, 1], color=line_color, s=35, marker="o", zorder=5
                )
                ax.scatter(
                    traj[-1, 0],
                    traj[-1, 1],
                    color=line_color,
                    s=55,
                    marker="X",
                    zorder=5,
                )

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if values is not None and dimensions == 2 and xlim is None:
            ax.autoscale_view()

        ax_labels = (
            axis_labels
            if axis_labels
            else [f"Dimension {i+1}" for i in range(dimensions)]
        )

        finalize_axes(
            ax,
            title=title,
            xlabel=ax_labels[0],
            ylabel=ax_labels[1],
            zlabel=ax_labels[2] if dimensions == 3 else None,
        )
        if axes_kws:
            kws = axes_kws.copy()
            tick_params = kws.pop("tick_params", None)
            locator_params = kws.pop("locator_params", None)
            labelsize = kws.pop("labelsize", None)

            ax.set(**kws)

            if tick_params:
                ax.tick_params(**tick_params)
            if locator_params:
                ax.locator_params(**locator_params)
            if labelsize:
                ax.xaxis.label.set_size(labelsize)
                ax.yaxis.label.set_size(labelsize)
                if dimensions == 3 and hasattr(ax, "zaxis"):
                    ax.zaxis.label.set_size(labelsize)
        if showlegend and labels is not None:
            handles, legend_labels = ax.get_legend_handles_labels()
            if handles:
                dedup = dict(zip(legend_labels, handles))
                ax.legend(dedup.values(), dedup.keys(), title="Label", frameon=False)
        return fig, ax


def plot_coranking_matrix(
    coranking_matrix: np.ndarray,
    title: str = "Co-Ranking Matrix",
    max_k: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of a co-ranking matrix produced by ``DimReduction.score()``.

    Parameters
    ----------
    coranking_matrix
        Square co-ranking matrix with shape ``(n_samples-1, n_samples-1)``.
    title
        Axes title.
    max_k
        Crop the matrix to the top-left ``max_k × max_k`` corner. Defaults
        to ``min(n, 50)``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_coranking_matrix :
        Interactive Plotly version.
    plot_shepard_diagram : Continuous distance-preservation scatter.
    plot_metrics : Scalar quality metric overview.
    plot_embedding : Low-dimensional scatter being diagnosed.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> Q = np.random.default_rng(42).integers(0, 10, size=(15, 15)).astype(float)
    >>> fig, ax = viz.plot_coranking_matrix(Q)
    """
    if coranking_matrix is None:
        raise ValueError("coranking_matrix is required.")
    matrix = np.asarray(coranking_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coranking_matrix must be a 2D square array.")
    k = min(matrix.shape[0], 50 if max_k is None else max_k)
    matrix = matrix[:k, :k]
    with coco_theme():
        fig, ax = plot_heatmap(
            matrix,
            cmap=SEQUENTIAL,
            aspect="auto",
            origin="lower",
            colorbar_label="Count",
            xlabel="Embedding Rank",
            ylabel="Original Rank",
            title=title,
            ax=ax,
            figsize=figsize or (6, 5),
        )
        return fig, ax


def plot_trajectory_separation(
    separation: dict,
    times: np.ndarray | None = None,
    top_n: int | None = None,
    color_map: dict[tuple, str] | None = None,
    linestyle_map: dict[tuple, str] | None = None,
    smooth_window: int = 1,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Pairwise label-separation timecourses for trajectory embeddings.

    Parameters
    ----------
    separation
        Mapping of ``(label_a, label_b)`` tuples (or any hashable key) to
        1D separation timecourses, as returned by
        ``DimReduction.evaluate_trajectory``.
    times
        Explicit time axis aligned with the separation arrays.
    top_n
        Keep only the ``top_n`` pairs ranked by peak separation.
    color_map
        Optional mapping of pair key to color.
    linestyle_map
        Optional mapping of pair key to linestyle.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_trajectory_separation :
        Interactive Plotly version.
    plot_trajectory : Raw trajectory paths in embedding space.
    plot_trajectory_metric_series : Metric time series for trajectories.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> sep = {("A",
    ...                           "B"): np.linspace(0.1, 0.9, 20), ("A",
    ...                           "C"): np.linspace(0.3, 0.6, 20)}
    >>> fig, ax = viz.plot_trajectory_separation(sep)
    """
    items = prepare_trajectory_separation_series(
        separation,
        times=times,
        top_n=top_n,
    )

    if smooth_window > 1:
        import pandas as pd

        for item in items:
            item["y"] = (
                pd.Series(item["y"])
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean()
                .values
            )

    with coco_theme():
        fig, ax = get_figure(ax, figsize, (10, 5))

        plot_kws = {}
        if color_map:
            plot_kws["palette"] = color_map
        if linestyle_map:
            plot_kws["style"] = "label"
            plot_kws["dashes"] = linestyle_map

        _plot_metric_lines(
            items,
            title="Trajectory Separation",
            ax=ax,
            axes_kws={"ylabel": "Separation", "plot_kws": plot_kws},
        )
        return fig, ax


def plot_component_loadings(
    components: np.ndarray,
    feature_names: list[str] | None = None,
    n_components: int | None = None,
    center: float = 0.0,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of component loadings from linear reducers.

    Parameters
    ----------
    components
        Loading matrix with shape ``(n_features, n_components)``.
    feature_names
        Optional feature names for row labels. Defaults to
        ``["Feature 0", "Feature 1", ...]``.
    n_components
        Crop to this many components (columns).
    center
        Colormap center value for the diverging palette.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_component_loadings :
        Interactive Plotly version.
    plot_feature_importance : Ranked feature-importance bar chart.
    plot_feature_correlation_heatmap : Feature-to-dimension correlation heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> components = rng.normal(size=(12, 3))
    >>> fig, ax = viz.plot_component_loadings(components)
    """
    loadings = prepare_component_loadings_frame(
        components,
        feature_names=feature_names,
        n_components=n_components,
    )
    with coco_theme():
        fig, ax = plot_heatmap(
            loadings,
            cmap=DIVERGING,
            center=center,
            colorbar_label="Loading",
            title="Component Loadings",
            xlabel="Component",
            ylabel="Feature",
            ax=ax,
            figsize=figsize
            or (
                max(6, loadings.shape[1] * 0.7),
                max(5, loadings.shape[0] * 0.25),
            ),
        )
        return fig, ax


def plot_phase_portrait(
    X: np.ndarray,
    times: np.ndarray,
    labels: Sequence,
    component_idx: int = 0,
    title: str = "Phase Portrait",
    figsize: tuple[float, float] | None = (8, 10),
    ax: plt.Axes | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a phase portrait (amplitude vs velocity) for condition-mean trajectories.

    Parameters
    ----------
    X : np.ndarray
        Trajectory array with shape ``(n_conditions, n_times, n_components)``.
    times : np.ndarray
        One-dimensional time axis aligned with the time dimension of ``X``.
    labels : sequence
        Condition labels, one per trajectory (first axis of ``X``).
    component_idx : int, default=0
        Index of the component to extract for the portrait.
    title : str, default="Phase Portrait"
        Axes title.
    figsize : tuple[float, float], optional
        Figure size used when creating new axes.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw into.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_phase_portrait :
        Interactive Plotly version.
    plot_trajectory : Full trajectory geometry in 2D or 3D space.
    plot_trajectory_metric_series : Scalar metric timecourses per trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.viz import dim_reduction as viz
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(3, 20, 5))
    >>> times = np.linspace(0, 1, 20)
    >>> fig, ax = viz.plot_phase_portrait(X, times, labels=["A", "B", "C"])
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            f"`X` must be 3D with shape (n_conditions, n_times, n_components). "
            f"Got {X.shape}."
        )
    times = np.asarray(times, dtype=float)
    if len(times) != X.shape[1]:
        raise ValueError(
            f"`times` length ({len(times)}) must match n_times ({X.shape[1]})."
        )
    if component_idx < 0 or component_idx >= X.shape[2]:
        raise ValueError(
            f"`component_idx` {component_idx} out of bounds "
            f"for n_components={X.shape[2]}."
        )

    dt = np.diff(times).mean() if len(times) > 1 else 1.0
    amplitude = X[:, :, component_idx]
    velocity = np.gradient(amplitude, axis=1) / dt

    with coco_theme():
        fig, cur_ax = get_figure(ax, figsize, (8, 6))
        palette = sns.color_palette("deep", len(labels))
        for idx, label in enumerate(labels):
            cur_ax.plot(
                amplitude[idx],
                velocity[idx],
                marker="o",
                markersize=4,
                linewidth=2,
                color=palette[idx % len(palette)],
                label=str(label),
            )
        finalize_axes(
            cur_ax,
            title=title,
            xlabel=f"PC{component_idx + 1} Amplitude",
            ylabel=f"PC{component_idx + 1} Velocity",
            legend=True,
            legend_title="Condition",
        )
        return fig, cur_ax


def plot_scree(
    evr: np.ndarray,
    title: str | None = "Scree Plot",
    figsize: tuple[float, float] | None = (8, 10),
    ax: plt.Axes | None = None,
    axes_kws: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a scree plot.
    Plots individual explained variance as bars and cumulative variance as a line.
    """
    from matplotlib.ticker import MultipleLocator

    # Modern monochrome theme
    bar_color = "#e0e0e0"  # Pale Silver/Light Gray
    bar_edge_color = "#9e9e9e"  # Medium Gray border
    line_color = "#000000"  # Pitch Black

    with coco_theme():
        fig, ax1 = get_figure(ax, figsize, (8, 10))

        components = np.arange(1, len(evr) + 1)
        cumulative = np.cumsum(evr)

        ax1.bar(
            components,
            evr,
            width=0.8,
            alpha=0.8,
            color=bar_color,
            edgecolor=bar_edge_color,
            linewidth=1.5,
            label="Individual",
        )

        ax2 = ax1.twinx()
        plot_line(
            components,
            cumulative,
            marker="o",
            color=line_color,
            linewidth=4,
            markersize=8,
            label="Cumulative",
            ax=ax2,
        )

        finalize_axes(
            ax1,
            xlabel="Principal Component",
            ylabel="Explained Variance Ratio",
            **(axes_kws or {}),
        )
        finalize_axes(ax2, ylabel="Cumulative Explained Variance", **(axes_kws or {}))

        # Override styling
        ax1.set_xlabel("Principal Component", fontsize=22, labelpad=10)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=22, labelpad=10)
        ax2.set_ylabel("Cumulative Explained Variance", fontsize=22, labelpad=10)
        ax1.tick_params(axis="both", labelsize=20, length=0)
        ax2.tick_params(axis="y", labelsize=16, length=0)

        ax1.set_xticks(components)

        # Spines
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(True)
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(True)

        # Tick locators
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))

        if title:
            ax1.set_title(title, fontsize=20, fontweight="bold", pad=20)

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        return fig, ax1
