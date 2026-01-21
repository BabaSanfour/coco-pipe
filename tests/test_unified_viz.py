
import numpy as np
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import MagicMock

from coco_pipe.viz import dim_reduction as viz_dr

def test_unified_plotting_interface():
    # Setup dummy data
    X_emb = np.random.rand(10, 2)
    labels = np.array([0, 1] * 5)
    metrics = {"trustworthiness": 0.9}
    loss_history = [0.5, 0.4, 0.3]
    evals = np.array([0.5, 0.3, 0.2])
    
    # 1. Plot Embedding
    # Static
    fig_static = viz_dr.plot_embedding(X_emb, labels, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)
    
    # Interactive
    fig_interactive = viz_dr.plot_embedding(X_emb, labels, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    # 2. Plot Metrics
    # Static
    fig_static = viz_dr.plot_metrics(metrics, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    # Interactive
    fig_interactive = viz_dr.plot_metrics(metrics, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    # 3. Plot Loss
    # Static
    fig_static = viz_dr.plot_loss_history(loss_history, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    # Interactive
    fig_interactive = viz_dr.plot_loss_history(loss_history, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    # 4. Plot Scree
    # Static
    fig_static = viz_dr.plot_eigenvalues(evals, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)
    
    # Interactive
    fig_interactive = viz_dr.plot_eigenvalues(evals, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    # 5. Plot Shepard
    X_orig = np.random.rand(10, 5)
    # Static
    fig_static = viz_dr.plot_shepard_diagram(X_orig, X_emb, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)
    
    # Interactive
    fig_interactive = viz_dr.plot_shepard_diagram(X_orig, X_emb, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    print("All unified interface tests passed!")

if __name__ == "__main__":
    test_unified_plotting_interface()
