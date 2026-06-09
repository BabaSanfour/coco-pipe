import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
from coco_pipe.viz.decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_fold_score_dispersion,
    plot_pr_curve,
    plot_roc_curve,
    plot_statistical_null_distribution,
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
    plot_temporal_statistical_assessment,
    plot_training_history,
)


def _diagnostic_result():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200, kind="classical")},
        metrics=["accuracy", "roc_auc", "brier_score"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=7),
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config).run(X, y)


def test_diagnostic_plots_return_matplotlib_figures():
    result = _diagnostic_result()

    figures = [
        plot_confusion_matrix(result),
        plot_roc_curve(result),
        plot_pr_curve(result),
        plot_calibration_curve(result),
        plot_fold_score_dispersion(result),
    ]

    for fig in figures:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_viz_temporal_plots():
    # Mock data for temporal plots
    summary_data = {
        "Model": ["LR", "LR"],
        "Metric": ["accuracy", "accuracy"],
        "Time": [0.0, 0.1],
        "Mean": [0.6, 0.7],
        "Std": [0.05, 0.05],
        "Significant": [True, False],
    }
    summary_df = pd.DataFrame(summary_data)

    # plot_temporal_score_curve
    fig = plot_temporal_score_curve(summary_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Non-numeric time
    summary_non_numeric = summary_df.copy()
    summary_non_numeric["Time"] = ["T1", "T2"]
    fig = plot_temporal_score_curve(summary_non_numeric)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # plot_temporal_generalization_matrix
    tg_data = {
        "Model": ["LR"] * 4,
        "Metric": ["accuracy"] * 4,
        "TrainTime": [0, 0, 1, 1],
        "TestTime": [0, 1, 0, 1],
        "Mean": [0.5, 0.6, 0.7, 0.8],
    }
    tg_df = pd.DataFrame(tg_data)
    fig = plot_temporal_generalization_matrix(tg_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # plot_temporal_statistical_assessment
    stats_data = {
        "Model": ["LR", "LR"],
        "Metric": ["accuracy", "accuracy"],
        "Time": [0.0, 0.1],
        "Observed": [0.6, 0.7],
        "NullLower": [0.4, 0.4],
        "NullUpper": [0.6, 0.6],
        "Significant": [True, False],
    }
    stats_df = pd.DataFrame(stats_data)
    fig = plot_temporal_statistical_assessment(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # plot_statistical_null_distribution
    fig = plot_statistical_null_distribution(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_viz_curves_mean_only():
    # Mock ROC/PR curve data with multiple folds
    curve_data = {
        "Model": ["LR"] * 4,
        "Fold": [0, 0, 1, 1],
        "FPR": [0, 1, 0, 1],
        "TPR": [0, 1, 0, 1],
        "Recall": [0, 1, 0, 1],
        "Precision": [1, 0, 1, 0],
        "Class": [0, 0, 0, 0],
    }
    curve_df = pd.DataFrame(curve_data)

    fig = plot_roc_curve(curve_df, mean_only=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = plot_pr_curve(curve_df, mean_only=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_training_history_hardening():
    # Mock history artifacts
    history = [
        {"epoch": 1, "loss": 0.5, "val_loss": 0.6},
        {"epoch": 2, "loss": 0.4, "val_loss": 0.5},
    ]
    artifacts_df = pd.DataFrame(
        [{"Model": "NN", "Key": "history", "ArtifactType": "history", "Value": history}]
    )

    fig = plot_training_history(artifacts_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
