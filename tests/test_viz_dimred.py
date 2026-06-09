import matplotlib.pyplot as plt
import numpy as np
import pytest

from coco_pipe.viz import dim_reduction as viz
from coco_pipe.viz import interactive as viz_interactive
from coco_pipe.viz.theme import DIVERGING, SEQUENTIAL


def _assert_fig_ax(out, figsize=None):
    assert isinstance(out, tuple)
    assert len(out) == 2
    fig, ax = out
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    if figsize is not None:
        width, height = fig.get_size_inches()
        assert round(width, 1) == figsize[0]
        assert round(height, 1) == figsize[1]
    plt.close(fig)


def test_dimred_static_plots_return_fig_ax_and_respect_figsize():
    rng = np.random.default_rng(42)
    X_emb = rng.normal(size=(60, 2))
    X_3d = rng.normal(size=(20, 5, 3))
    X_orig = rng.normal(size=(60, 5))
    V_emb = rng.normal(size=(60, 2))
    metric_df = {
        "trustworthiness": 0.92,
        "continuity": 0.88,
    }
    corr_payload = {
        "correlation": {
            "D1": {"F1": 0.3, "F2": 0.1},
            "D2": {"F1": -0.2, "F2": 0.5},
        }
    }

    calls = [
        lambda: viz.plot_embedding(X_emb, labels=np.arange(len(X_emb)), figsize=(3, 3)),
        lambda: viz.plot_metrics(metric_df, figsize=(3, 3)),
        lambda: viz.plot_loss_history([1.0, 0.7, 0.4], figsize=(3, 3)),
        lambda: viz.plot_eigenvalues(
            {"cond": np.array([0.5, 0.3, 0.1])}, figsize=(3, 3)
        ),
        lambda: viz.plot_shepard_diagram(X_orig, X_emb, figsize=(3, 3)),
        lambda: viz.plot_streamlines(X_emb, V_emb, figsize=(3, 3)),
        lambda: viz.plot_feature_importance({"F1": 0.6, "F2": 0.4}, figsize=(3, 3)),
        lambda: viz.plot_feature_correlation_heatmap(corr_payload, figsize=(3, 3)),
        lambda: viz.plot_trajectory_metric_series(
            np.vstack([np.arange(5), np.arange(5) + 1]), figsize=(3, 3)
        ),
        lambda: viz.plot_trajectory(X_3d, dimensions=3, figsize=(3, 3)),
        lambda: viz.plot_coranking_matrix(np.eye(8), figsize=(3, 3)),
        lambda: viz.plot_trajectory_separation(
            {("A", "B"): np.arange(5)}, figsize=(3, 3)
        ),
        lambda: viz.plot_component_loadings(rng.normal(size=(4, 2)), figsize=(3, 3)),
    ]
    for call in calls:
        _assert_fig_ax(call(), figsize=(3, 3))


def test_dimred_interactive_module_delegates():
    go = pytest.importorskip("plotly.graph_objects")
    X_emb = np.random.default_rng(0).normal(size=(20, 2))
    fig = viz_interactive.plot_embedding(X_emb)
    assert isinstance(fig, go.Figure)


def test_embedding_label_kind_is_explicit():
    X_emb = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.linspace(0, 1, len(X_emb))

    _assert_fig_ax(
        viz.plot_embedding(
            X_emb,
            labels=labels,
            label_kind="continuous",
            figsize=(3, 3),
        ),
        figsize=(3, 3),
    )
    with pytest.raises(ValueError, match="label_kind"):
        viz.plot_embedding(X_emb, labels=labels, label_kind="auto")


def test_embedding_metric_annotation_requires_metric_name():
    X_emb = np.random.default_rng(0).normal(size=(20, 2))
    metrics = {"trustworthiness": 0.91, "continuity": 0.88}

    fig, ax = viz.plot_embedding(
        X_emb,
        metrics=metrics,
        metric_name="trustworthiness",
        figsize=(3, 3),
    )
    assert any("trustworthiness" in text.get_text() for text in ax.texts)
    plt.close(fig)

    fig, ax = viz.plot_embedding(X_emb, metrics=metrics, figsize=(3, 3))
    assert not ax.texts
    plt.close(fig)

    with pytest.raises(ValueError, match="not found"):
        viz.plot_embedding(X_emb, metrics=metrics, metric_name="missing")


def test_dimred_error_paths():
    with pytest.raises(ValueError, match="2D"):
        viz.plot_embedding(np.arange(10))
    with pytest.raises(ValueError, match="empty"):
        viz.plot_loss_history([])
    with pytest.raises(ValueError, match="2D"):
        viz.plot_streamlines(np.random.randn(10, 3), np.random.randn(10, 2))
    with pytest.raises(ValueError):
        viz.plot_coranking_matrix(None)
    with pytest.raises(ValueError):
        viz.plot_trajectory_separation({})
    with pytest.raises(ValueError):
        viz.plot_component_loadings(None)


def test_dimred_palette_constants_are_defaults():
    assert SEQUENTIAL == "viridis"
    assert DIVERGING == "RdBu_r"


def test_static_dimred_interface_returns_fig_ax():
    X_emb = np.random.rand(10, 2)
    labels = np.array([0, 1] * 5)
    metrics = {"trustworthiness": 0.9}
    loss_history = [0.5, 0.4, 0.3]
    evals = {"cond": np.array([0.5, 0.3, 0.2])}
    X_orig = np.random.rand(10, 5)

    _assert_fig_ax(viz.plot_embedding(X_emb, labels))
    _assert_fig_ax(viz.plot_metrics(metrics))
    _assert_fig_ax(viz.plot_loss_history(loss_history))
    _assert_fig_ax(viz.plot_eigenvalues(evals))
    _assert_fig_ax(viz.plot_shepard_diagram(X_orig, X_emb))


def test_trajectory_plots_static_and_interactive():
    X = np.random.rand(2, 20, 2)
    times = np.linspace(0, 1, 20)
    labels = np.array([0, 1])

    _assert_fig_ax(viz.plot_trajectory(X, times=times, labels=labels))

    X_3d = np.random.rand(2, 20, 3)
    _assert_fig_ax(viz.plot_trajectory(X_3d, times=times, labels=labels, dimensions=3))


def test_trajectory_plot_rejects_flat_inputs():
    with pytest.raises(ValueError, match="3D trajectory tensor"):
        viz.plot_trajectory(np.random.rand(20, 2))
