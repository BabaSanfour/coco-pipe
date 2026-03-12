import matplotlib.pyplot as plt
import numpy as np
import pytest

from coco_pipe.viz.dim_reduction import (
    plot_eigenvalues,
    plot_embedding,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_interpretation,
    plot_loss_history,
    plot_metrics,
    plot_shepard_diagram,
    plot_streamlines,
    plot_trajectory,
    plot_trajectory_metric_series,
)


@pytest.fixture
def mock_embedding():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 2))


@pytest.fixture
def mock_embedding_3d():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 3))


@pytest.fixture
def mock_labels_cat():
    return np.array(["A"] * 50 + ["B"] * 50)


@pytest.fixture
def mock_labels_cont():
    return np.linspace(0, 1, 100)


@pytest.fixture
def mock_velocities():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 2))


@pytest.fixture
def mock_trajectory_data():
    n_times = 100
    t = np.linspace(0, 10, n_times)
    traj_a = np.column_stack([np.cos(t), np.sin(t)])
    traj_b = np.column_stack([t / t.max(), 1 - (t / t.max())])
    X = np.stack([traj_a, traj_b], axis=0)
    labels = np.array(["A", "B"])
    values = np.stack([np.linspace(0, 1, n_times), np.linspace(1, 0, n_times)], axis=0)
    return X, labels, t, values


def test_plot_embedding_2d_categorical(mock_embedding, mock_labels_cat):
    fig = plot_embedding(mock_embedding, labels=mock_labels_cat, title="2D Categorical")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_2d_continuous(mock_embedding, mock_labels_cont):
    fig = plot_embedding(mock_embedding, labels=mock_labels_cont, title="2D Continuous")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_3d_categorical(mock_embedding_3d, mock_labels_cat):
    fig = plot_embedding(
        mock_embedding_3d,
        labels=mock_labels_cat,
        dims=(0, 1, 2),
        title="3D Categorical",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        plot_embedding(np.arange(10))


def test_plot_embedding_params(mock_embedding):
    metrics = {"trustworthiness": 0.95, "continuity": 0.98}
    fig = plot_embedding(
        mock_embedding,
        title="Custom Params",
        s=50,
        alpha=0.5,
        metrics=metrics,
        cmap="plasma",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metrics():
    scores = {"metric1": 0.8, "metric2": 0.9, "n_iter_": 100, "is_good": True}
    fig = plot_metrics(scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metrics_distribution_heatmap_and_line():
    repeated_df = pytest.importorskip("pandas").DataFrame(
        {
            "method": ["PCA", "PCA", "UMAP", "UMAP"],
            "metric": ["trustworthiness"] * 4,
            "value": [0.82, 0.85, 0.91, 0.93],
            "scope": ["subject"] * 4,
            "scope_value": ["s1", "s2", "s1", "s2"],
        }
    )
    fig_violin = plot_metrics(repeated_df, plot_type="violin")
    assert isinstance(fig_violin, plt.Figure)
    plt.close(fig_violin)

    heatmap_df = pytest.importorskip("pandas").DataFrame(
        {
            "method": ["PCA", "PCA", "UMAP", "UMAP"],
            "metric": ["trustworthiness", "continuity"] * 2,
            "value": [0.82, 0.79, 0.91, 0.88],
            "scope": ["global"] * 4,
            "scope_value": ["global"] * 4,
        }
    )
    fig_heatmap = plot_metrics(heatmap_df, plot_type="heatmap")
    assert isinstance(fig_heatmap, plt.Figure)
    plt.close(fig_heatmap)

    line_df = pytest.importorskip("pandas").DataFrame(
        {
            "method": ["PCA", "PCA", "UMAP", "UMAP"],
            "metric": ["trustworthiness"] * 4,
            "value": [0.82, 0.85, 0.91, 0.93],
            "scope": ["k"] * 4,
            "scope_value": [5, 10, 5, 10],
        }
    )
    fig_line = plot_metrics(line_df, plot_type="line", metric="trustworthiness")
    assert isinstance(fig_line, plt.Figure)
    plt.close(fig_line)


def test_plot_loss_history():
    history = [1.0, 0.8, 0.6, 0.4, 0.2]
    fig = plot_loss_history(history)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_eigenvalues():
    eigs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    fig = plot_eigenvalues(eigs, ylabel="Variance Ratio")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_shepard_diagram(mock_embedding):
    X_orig = np.random.rand(100, 5)
    fig = plot_shepard_diagram(X_orig, mock_embedding)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_shepard_diagram_uses_cached_distances(monkeypatch, mock_embedding):
    X_orig = np.random.rand(100, 5)
    cached = {
        "original": np.linspace(0.0, 1.0, 20),
        "embedded": np.linspace(0.0, 1.0, 20),
    }

    def _unexpected(*args, **kwargs):
        raise AssertionError("shepard_diagram_data should not be recomputed")

    monkeypatch.setattr("coco_pipe.viz.dim_reduction.shepard_diagram_data", _unexpected)

    fig = plot_shepard_diagram(X_orig, mock_embedding, distances=cached)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_streamlines(mock_embedding, mock_velocities):
    fig = plot_streamlines(mock_embedding, mock_velocities)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_streamlines_error(mock_embedding_3d, mock_velocities):
    with pytest.raises(ValueError, match="only supported for 2D"):
        plot_streamlines(mock_embedding_3d, mock_velocities)


def test_plot_embedding_interactive_categorical(mock_embedding):
    import plotly.graph_objects as go

    labels = np.array([0] * 50 + [1] * 50)
    fig = plot_embedding(
        mock_embedding, labels=labels, interactive=True, title="Interactive Cat"
    )
    assert isinstance(fig, go.Figure)
    trace = fig.data[0]
    assert trace.marker.colorbar.ticktext == ("0", "1")
    assert trace.marker.colorbar.tickvals == (0, 1)


def test_plot_embedding_interactive_metadata_dropdown(mock_embedding):
    import plotly.graph_objects as go

    labels = np.array(["A"] * 50 + ["B"] * 50)
    metadata = {"Class": ["X"] * 25 + ["Y"] * 75, "Score": np.random.rand(100)}
    fig = plot_embedding(
        mock_embedding,
        labels=labels,
        metadata=metadata,
        interactive=True,
        title="Interactive Metadata",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.updatemenus) > 0


def test_plot_embedding_interactive_colormaps(mock_embedding):
    labels_cont = np.linspace(0, 1, 100)
    fig_cont = plot_embedding(
        mock_embedding,
        labels=labels_cont,
        interactive=True,
        cmap="Magma",
        title="Interactive Cont",
    )
    assert fig_cont.data[0].marker.colorscale is not None
    assert isinstance(fig_cont.data[0].marker.colorscale, (tuple, list, str))

    labels_cat = np.array(["A", "B", "C"] * 33 + ["A"])
    fig_cat = plot_embedding(
        mock_embedding,
        labels=labels_cat,
        interactive=True,
        palette="Set1",
        title="Interactive Palette",
    )
    assert isinstance(fig_cat.data[0].marker.colorscale, (list, tuple))
    assert len(fig_cat.data[0].marker.colorscale) > 0


def test_plot_feature_importance_from_dict():
    fig = plot_feature_importance({"F1": 0.6, "F2": 0.4}, title="Importance")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_correlation_heatmap_from_payload():
    payload = {
        "correlation": {
            "Dimension 1": {"F1": 0.8, "F2": -0.2},
            "Dimension 2": {"F1": 0.1, "F2": 0.9},
        }
    }
    fig = plot_feature_correlation_heatmap(payload)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_interpretation_dispatches():
    corr_payload = {
        "analysis": {
            "correlation": {
                "Dimension 1": {"F1": 0.8, "F2": -0.2},
                "Dimension 2": {"F1": 0.1, "F2": 0.9},
            }
        }
    }
    fig_corr = plot_interpretation(corr_payload, analysis="correlation")
    assert isinstance(fig_corr, plt.Figure)
    plt.close(fig_corr)

    pert_payload = {
        "records": [
            {
                "method": "PCA",
                "analysis": "perturbation",
                "feature": "F1",
                "value": 0.7,
            },
            {
                "method": "PCA",
                "analysis": "perturbation",
                "feature": "F2",
                "value": 0.3,
            },
        ]
    }
    fig_pert = plot_interpretation(pert_payload, analysis="perturbation", method="PCA")
    assert isinstance(fig_pert, plt.Figure)
    plt.close(fig_pert)


def test_plot_trajectory_basic(mock_trajectory_data):
    X, labels, times, _ = mock_trajectory_data
    fig = plot_trajectory(X, labels=labels, times=times, title="Basic Trajectory")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_values(mock_trajectory_data):
    X, labels, times, values = mock_trajectory_data
    fig = plot_trajectory(
        X, values=values, labels=labels, times=times, title="Value Colored"
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_smoothing(mock_trajectory_data):
    X, labels, times, _ = mock_trajectory_data
    fig = plot_trajectory(
        X, labels=labels, times=times, smooth_window=5, title="Smoothed"
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_3d(mock_trajectory_data):
    X_2d, labels, times, _ = mock_trajectory_data
    X_3d = np.concatenate(
        [X_2d, np.random.randn(X_2d.shape[0], X_2d.shape[1], 1)], axis=2
    )
    fig = plot_trajectory(X_3d, labels=labels, dimensions=3, times=times)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_rejects_2d():
    with pytest.raises(ValueError, match="3D trajectory tensor"):
        plot_trajectory(np.random.rand(20, 2))


def test_plot_trajectory_interactive(mock_trajectory_data):
    import plotly.graph_objects as go

    X, labels, times, values = mock_trajectory_data
    fig = plot_trajectory(
        X,
        labels=labels,
        times=times,
        values=values,
        smooth_window=5,
        interactive=True,
        title="Interactive Trajectory",
    )
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_metric_series():
    series = np.random.rand(3, 25)
    times = np.linspace(0, 1, 25)
    labels = np.array(["A", "A", "B"])
    fig = plot_trajectory_metric_series(
        series, times=times, labels=labels, title="Trajectory Metric"
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
