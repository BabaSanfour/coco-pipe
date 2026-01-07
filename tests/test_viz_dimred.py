
import pytest
import numpy as np
import matplotlib.pyplot as plt
from coco_pipe.viz.dim_reduction import (
    plot_embedding, 
    plot_metrics, 
    plot_loss_history, 
    plot_eigenvalues,
    plot_shepard_diagram,
    plot_streamlines
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
    return np.array(['A'] * 50 + ['B'] * 50)

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
    fig = plot_embedding(mock_embedding_3d, labels=mock_labels_cat, dims=(0, 1, 2), title="3D Categorical")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_embedding_params(mock_embedding):
    """Test various parameters of plot_embedding."""
    metrics = {'trustworthiness': 0.95, 'continuity': 0.98}
    fig = plot_embedding(
        mock_embedding, 
        title="Custom Params", 
        s=50, 
        alpha=0.5, 
        metrics=metrics,
        cmap='plasma'
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_metrics():
    """Test plot_metrics."""
    scores = {'metric1': 0.8, 'metric2': 0.9, 'n_iter_': 100, 'is_good': True}
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
