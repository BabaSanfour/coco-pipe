"""
Decoding Visualization
======================

Focused plotting helpers for decoding result tables.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _temporal_summary_frame(result_or_scores: Any) -> pd.DataFrame:
    if hasattr(result_or_scores, "get_temporal_score_summary"):
        return result_or_scores.get_temporal_score_summary()
    return pd.DataFrame(result_or_scores)


def _filter_temporal_summary(
    summary: pd.DataFrame,
    metric: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    frame = summary.copy()
    if metric is not None:
        frame = frame[frame["Metric"] == metric]
    if model is not None:
        frame = frame[frame["Model"] == model]
    return frame


def _result_frame(result_or_frame: Any, accessor: str) -> pd.DataFrame:
    if hasattr(result_or_frame, accessor):
        return getattr(result_or_frame, accessor)()
    return pd.DataFrame(result_or_frame)


def _filter_frame(
    frame: pd.DataFrame,
    model: Optional[str] = None,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    data = frame.copy()
    if model is not None and "Model" in data:
        data = data[data["Model"] == model]
    if fold is not None and "Fold" in data:
        data = data[data["Fold"] == fold]
    return data


def plot_confusion_matrix(
    result_or_matrix: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot an aggregated confusion matrix from decoding diagnostics."""
    frame = _filter_frame(
        _result_frame(result_or_matrix, "get_confusion_matrices"),
        model=model,
        fold=fold,
    )
    if frame.empty:
        raise ValueError("No confusion-matrix rows available to plot.")

    matrix = frame.pivot_table(
        index="TrueLabel",
        columns="PredictedLabel",
        values="Value",
        aggfunc="sum",
        fill_value=0,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    im = ax.imshow(np.asarray(matrix, dtype=float), cmap="Blues")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(value) for value in matrix.columns], rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(value) for value in matrix.index])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix.iloc[row_idx, col_idx]:.3g}",
                ha="center",
                va="center",
            )
    fig.colorbar(im, ax=ax, label="Count")
    return fig


def plot_roc_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    mean_only: bool = False,
):
    """Plot ROC curves from decoding curve diagnostics."""
    frame = _filter_frame(
        _result_frame(result_or_curve, "get_roc_curve"),
        model=model,
        fold=fold,
    )
    if frame.empty:
        raise ValueError("No ROC curve rows available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    group_cols = ["Model"] + (["Class"] if "Class" in frame else [])
    for keys, group in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        if mean_only:
            # Pivot to align FPR and interpolate TPR
            from scipy import interpolate

            all_fpr = np.unique(
                np.concatenate([g["FPR"].to_numpy() for _, g in group.groupby("Fold")])
            )
            all_fpr = np.sort(all_fpr)
            tprs = []
            for _, fold_group in group.groupby("Fold"):
                interp = interpolate.interp1d(
                    fold_group["FPR"],
                    fold_group["TPR"],
                    bounds_error=False,
                    fill_value=(0, 1),
                )
                tprs.append(interp(all_fpr))
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            label = f"{keys[0]}"
            if len(keys) > 1:
                label = f"{label} class {keys[1]}"
            ax.plot(all_fpr, mean_tpr, label=f"{label} (mean)", linewidth=2)
            ax.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)
        else:
            for f_idx, fold_group in group.groupby("Fold"):
                label = f"{keys[0]} fold {f_idx}"
                if len(keys) > 1:
                    label = f"{label} class {keys[1]}"
                ax.plot(fold_group["FPR"], fold_group["TPR"], label=label, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curve")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_pr_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    mean_only: bool = False,
):
    """Plot precision-recall curves from decoding diagnostics."""
    frame = _filter_frame(
        _result_frame(result_or_curve, "get_pr_curve"),
        model=model,
        fold=fold,
    )
    if frame.empty:
        raise ValueError("No precision-recall curve rows available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    group_cols = ["Model"] + (["Class"] if "Class" in frame else [])
    for keys, group in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        if mean_only:
            from scipy import interpolate

            all_recall = np.unique(
                np.concatenate(
                    [g["Recall"].to_numpy() for _, g in group.groupby("Fold")]
                )
            )
            all_recall = np.sort(all_recall)
            precs = []
            for _, fold_group in group.groupby("Fold"):
                # PR curves are not necessarily monotonic, but interpolation is
                # standard for mean PR
                interp = interpolate.interp1d(
                    fold_group["Recall"],
                    fold_group["Precision"],
                    bounds_error=False,
                    fill_value=(1, 0),
                )
                precs.append(interp(all_recall))
            mean_prec = np.mean(precs, axis=0)
            std_prec = np.std(precs, axis=0)
            label = f"{keys[0]}"
            if len(keys) > 1:
                label = f"{label} class {keys[1]}"
            ax.plot(all_recall, mean_prec, label=f"{label} (mean)", linewidth=2)
            ax.fill_between(
                all_recall, mean_prec - std_prec, mean_prec + std_prec, alpha=0.15
            )
        else:
            for f_idx, fold_group in group.groupby("Fold"):
                label = f"{keys[0]} fold {f_idx}"
                if len(keys) > 1:
                    label = f"{label} class {keys[1]}"
                ax.plot(
                    fold_group["Recall"],
                    fold_group["Precision"],
                    label=label,
                    alpha=0.5,
                )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_calibration_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot reliability curves from decoding calibration diagnostics."""
    frame = _filter_frame(
        _result_frame(result_or_curve, "get_calibration_curve"),
        model=model,
        fold=fold,
    )
    if frame.empty:
        raise ValueError("No calibration curve rows available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    group_cols = ["Model", "Fold"] + (["Class"] if "Class" in frame else [])
    for keys, group in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = f"{keys[0]} fold {keys[1]}"
        if len(keys) > 2:
            label = f"{label} class {keys[2]}"
        ax.plot(
            group["MeanPredictedProbability"],
            group["FractionPositive"],
            marker="o",
            label=label,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title(title or "Calibration Curve")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_fold_score_dispersion(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot scalar fold-score dispersion by model and metric."""
    frame = _result_frame(result_or_scores, "get_detailed_scores")
    if frame.empty:
        raise ValueError("No fold score rows available to plot.")
    if metric is not None:
        frame = frame[frame["Metric"] == metric]
    if model is not None:
        frame = frame[frame["Model"] == model]
    if "Value" not in frame:
        raise ValueError("No scalar fold score rows available to plot.")
    frame = frame[frame["Value"].notna()].copy()
    for column in ["Time", "TrainTime", "TestTime"]:
        if column in frame:
            frame = frame[frame[column].isna()]
    if frame.empty:
        raise ValueError("No scalar fold score rows available to plot.")

    labels = []
    values = []
    for (model_name, metric_name), group in frame.groupby(["Model", "Metric"]):
        labels.append(f"{model_name}\n{metric_name}")
        values.append(group["Value"].astype(float).to_numpy())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(6, len(labels) * 1.5), 5))
    else:
        fig = ax.get_figure()

    for idx, val_set in enumerate(values):
        x = np.random.normal(idx + 1, 0.04, size=len(val_set))
        ax.scatter(x, val_set, alpha=0.5, color="black", zorder=3, s=15)

    if all(len(v) >= 8 for v in values) and len(values) > 0:
        ax.violinplot(values, showmeans=True, showmedians=False)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    else:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Score")
    ax.set_title(title or "Fold Score Dispersion")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    return fig


def plot_temporal_score_curve(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """
    Plot mean temporal decoding score curves.

    Parameters
    ----------
    result_or_scores : ExperimentResult or DataFrame-like
        Result object or output from ``get_temporal_score_summary()``.
    metric, model : str, optional
        Optional filters.
    title : str, optional
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    """
    summary = _filter_temporal_summary(
        _temporal_summary_frame(result_or_scores), metric=metric, model=model
    )
    if summary.empty or "Time" not in summary:
        raise ValueError("No 1D temporal score rows available to plot.")

    curve_data = summary[summary["Time"].notna()].copy()
    if curve_data.empty:
        raise ValueError("No 1D temporal score rows available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for (model_name, metric_name), group in curve_data.groupby(["Model", "Metric"]):
        times = group["Time"].to_numpy()
        numeric_times = pd.api.types.is_numeric_dtype(group["Time"])
        x_vals = times if numeric_times else np.arange(len(times))
        label = f"{model_name} / {metric_name}"
        ax.plot(x_vals, group["Mean"], marker="o", linewidth=2, label=label)
        if "Std" in group:
            ax.fill_between(
                x_vals,
                group["Mean"] - group["Std"],
                group["Mean"] + group["Std"],
                alpha=0.15,
            )
        if not numeric_times:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(value) for value in times], rotation=45)

    ax.set_title(title or "Temporal Decoding Score")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_temporal_generalization_matrix(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """
    Plot a train-time by test-time temporal generalization heatmap.

    Parameters
    ----------
    result_or_scores : ExperimentResult or DataFrame-like
        Result object or output from ``get_temporal_score_summary()``.
    metric, model : str, optional
        Optional filters. When omitted and multiple matrices are present, the
        first model/metric pair in the table is plotted.
    title : str, optional
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    """
    summary = _filter_temporal_summary(
        _temporal_summary_frame(result_or_scores), metric=metric, model=model
    )
    required = {"TrainTime", "TestTime", "Mean"}
    if summary.empty or not required.issubset(summary.columns):
        raise ValueError("No temporal generalization matrix rows available to plot.")

    matrix_data = summary[
        summary["TrainTime"].notna() & summary["TestTime"].notna()
    ].copy()
    if matrix_data.empty:
        raise ValueError("No temporal generalization matrix rows available to plot.")

    first = matrix_data.iloc[0]
    matrix_data = matrix_data[
        (matrix_data["Model"] == first["Model"])
        & (matrix_data["Metric"] == first["Metric"])
    ]
    train_order = pd.unique(matrix_data["TrainTime"])
    test_order = pd.unique(matrix_data["TestTime"])
    matrix = matrix_data.pivot(index="TrainTime", columns="TestTime", values="Mean")
    matrix = matrix.reindex(index=train_order, columns=test_order)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(np.asarray(matrix), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(value) for value in matrix.columns], rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(value) for value in matrix.index])
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    ax.set_title(title or f"{first['Model']} / {first['Metric']}")
    fig.colorbar(im, ax=ax, label="Score")
    return fig


def plot_temporal_statistical_assessment(
    result_or_assessment: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot temporal observed scores, null bands, and significant segments."""
    frame = _result_frame(result_or_assessment, "get_statistical_assessment")
    if frame.empty:
        raise ValueError("No statistical assessment rows available to plot.")
    if metric is not None and "Metric" in frame:
        frame = frame[frame["Metric"] == metric]
    if model is not None and "Model" in frame:
        frame = frame[frame["Model"] == model]
    if "Time" not in frame:
        raise ValueError("No temporal statistical assessment rows available.")
    frame = frame[frame["Time"].notna()].copy()
    if frame.empty:
        raise ValueError("No temporal statistical assessment rows available.")

    first = frame.iloc[0]
    frame = frame[
        (frame["Model"] == first["Model"]) & (frame["Metric"] == first["Metric"])
    ]
    numeric_times = pd.api.types.is_numeric_dtype(frame["Time"])
    x_vals = frame["Time"].to_numpy() if numeric_times else np.arange(len(frame))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    ax.plot(x_vals, frame["Observed"], marker="o", linewidth=2, label="Observed")
    if {"NullLower", "NullUpper"}.issubset(frame.columns):
        if frame["NullLower"].notna().any() and frame["NullUpper"].notna().any():
            ax.fill_between(
                x_vals,
                frame["NullLower"].astype(float),
                frame["NullUpper"].astype(float),
                alpha=0.2,
                label="Permutation null band",
            )
    if "Significant" in frame and frame["Significant"].fillna(False).any():
        sig = frame["Significant"].fillna(False).to_numpy(dtype=bool)
        ax.scatter(
            x_vals[sig],
            frame["Observed"].to_numpy(dtype=float)[sig],
            marker="s",
            color="black",
            label="Corrected significant",
            zorder=3,
        )
    if not numeric_times:
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(value) for value in frame["Time"]], rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    default_title = f"{first['Model']} / {first['Metric']} statistical assessment"
    ax.set_title(title or default_title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_statistical_null_distribution(
    result_or_assessment: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot observed scores with available null interval summaries."""
    frame = _result_frame(result_or_assessment, "get_statistical_assessment")
    if metric is not None and "Metric" in frame:
        frame = frame[frame["Metric"] == metric]
    if model is not None and "Model" in frame:
        frame = frame[frame["Model"] == model]
    frame = frame[frame["Observed"].notna()].copy()
    if frame.empty:
        raise ValueError("No statistical assessment rows available to plot.")

    labels = [f"{row.Model}\n{row.Metric}" for row in frame.itertuples()]
    x_vals = np.arange(len(frame))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(6, len(frame) * 1.2), 5))
    else:
        fig = ax.get_figure()
    ax.scatter(x_vals, frame["Observed"].astype(float), label="Observed")
    if {"NullLower", "NullUpper"}.issubset(frame.columns):
        lower = frame["NullLower"].astype(float)
        upper = frame["NullUpper"].astype(float)
        center = (lower + upper) / 2
        yerr = np.vstack([center - lower, upper - center])
        ax.errorbar(x_vals, center, yerr=yerr, fmt="o", label="Null band")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title or "Statistical Assessment")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_training_history(
    result_or_artifacts: Any,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
):
    """Plot neural training and validation history from model artifacts."""
    artifacts = _result_frame(result_or_artifacts, "get_model_artifacts")
    if model is not None and "Model" in artifacts:
        artifacts = artifacts[artifacts["Model"] == model]
    rows = artifacts[
        (artifacts["Key"].isin(["training", "validation"]))
        | (artifacts["ArtifactType"] == "history")
    ]
    if rows.empty:
        raise ValueError("No training history artifacts available to plot.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
    else:
        fig = ax.get_figure()
    for row in rows.itertuples():
        history = row.Value or []
        if not history:
            continue
        frame = pd.DataFrame(history)
        if "epoch" not in frame:
            continue
        value_cols = [col for col in frame.columns if col != "epoch"]
        for col in value_cols:
            ax.plot(frame["epoch"], frame[col], marker="o", label=f"{row.Model} {col}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title or "Training History")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    return fig
