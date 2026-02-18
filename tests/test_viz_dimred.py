import matplotlib.pyplot as plt
import numpy as np
import pytest

from coco_pipe.viz.dim_reduction import (
    plot_eigenvalues,
    plot_embedding,
    plot_loss_history,
    plot_metrics,
    plot_shepard_diagram,
    plot_streamlines,
)


@pytest.fixture
def mock_embedding():
    """Create a mock 2D embedding."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 2))


@pytest.fixture
def mock_embedding_3d():
    """Create a mock 3D embedding."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 3))


@pytest.fixture
def mock_labels_cat():
    """Create categorical labels."""
    return np.array(["A"] * 50 + ["B"] * 50)


@pytest.fixture
def mock_labels_cont():
    """Create continuous labels."""
    return np.linspace(0, 1, 100)


@pytest.fixture
def mock_velocities():
    """Create mock velocity vectors."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 2))


def test_plot_embedding_2d_categorical(mock_embedding, mock_labels_cat):
    """Test 2D embedding plot with categorical labels."""
    fig = plot_embedding(mock_embedding, labels=mock_labels_cat, title="2D Categorical")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_2d_continuous(mock_embedding, mock_labels_cont):
    """Test 2D embedding plot with continuous labels."""
    fig = plot_embedding(mock_embedding, labels=mock_labels_cont, title="2D Continuous")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_3d_categorical(mock_embedding_3d, mock_labels_cat):
    """Test 3D embedding plot with categorical labels."""
    fig = plot_embedding(
        mock_embedding_3d,
        labels=mock_labels_cat,
        dims=(0, 1, 2),
        title="3D Categorical",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_params(mock_embedding):
    """Test various parameters of plot_embedding."""
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
    """Test plot_metrics."""
    scores = {"metric1": 0.8, "metric2": 0.9, "n_iter_": 100, "is_good": True}
    fig = plot_metrics(scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_loss_history():
    """Test plot_loss_history."""
    history = [1.0, 0.8, 0.6, 0.4, 0.2]
    fig = plot_loss_history(history)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_eigenvalues():
    """Test plot_eigenvalues."""
    eigs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    fig = plot_eigenvalues(eigs, ylabel="Variance Ratio")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_shepard_diagram(mock_embedding):
    """Test plot_shepard_diagram."""
    X_orig = np.random.rand(100, 5)
    fig = plot_shepard_diagram(X_orig, mock_embedding)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_streamlines(mock_embedding, mock_velocities):
    """Test plot_streamlines."""
    fig = plot_streamlines(mock_embedding, mock_velocities)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_streamlines_error(mock_embedding_3d, mock_velocities):
    """Test error raising for 3D streamlines."""
    with pytest.raises(ValueError, match="only supported for 2D"):
        plot_streamlines(mock_embedding_3d, mock_velocities)


def test_viz_hybrid_support():
    """Test plotting with DataContainer input."""
    from coco_pipe.io import DataContainer

    np.random.randn(20, 10)
    Y = np.array([0] * 10 + [1] * 10)

    # Create DataContainer wrapper for embedding manually
    # (Simulating user pipeline output)
    emb_data = np.random.randn(20, 2)
    container = DataContainer(X=emb_data, dims=["obs", "comp"], y=Y)

    # Plot should accept container and extract Y automatically
    fig = plot_embedding(container, title="Test Hybrid Plot")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_embedding_interactive_categorical(mock_embedding):
    """Test interactive embedding plot with integer categorical labels."""
    # Integer labels < 20 should be detected as categorical
    labels = np.array([0] * 50 + [1] * 50)

    # We need to import Plotly go.Figure to check types
    import plotly.graph_objects as go

    fig = plot_embedding(
        mock_embedding, labels=labels, interactive=True, title="Interactive Cat"
    )
    assert isinstance(fig, go.Figure)

    # Check if colorbar text exists and contains "0" and "1"
    # Note: In our implementation, we use ticktext for categorical
    trace = fig.data[0]
    assert trace.marker.colorbar.ticktext == ("0", "1")
    assert trace.marker.colorbar.tickvals == (0, 1)


def test_plot_embedding_interactive_colormaps(mock_embedding):
    """Test interactive embedding plot with custom colormaps."""

    # Continuous
    labels_cont = np.linspace(0, 1, 100)
    fig_cont = plot_embedding(
        mock_embedding,
        labels=labels_cont,
        interactive=True,
        cmap="Magma",
        title="Interactive Cont",
    )
    # Plotly converts "Magma" to an explicit tuple scale
    assert fig_cont.data[0].marker.colorscale is not None
    assert isinstance(fig_cont.data[0].marker.colorscale, (tuple, list, str))

    # Categorical Palette
    labels_cat = np.array(["A", "B", "C"] * 33 + ["A"])
    fig_cat = plot_embedding(
        mock_embedding,
        labels=labels_cat,
        interactive=True,
        palette="Set1",
        title="Interactive Palette",
    )
    # Our implementation converts palette to a discrete scale
    # Check if a scale was created
    assert isinstance(fig_cat.data[0].marker.colorscale, (list, tuple))
    assert len(fig_cat.data[0].marker.colorscale) > 0
