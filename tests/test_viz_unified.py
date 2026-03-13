import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest

from coco_pipe.viz import dim_reduction as viz_dr


def test_unified_plotting_interface():
    X_emb = np.random.rand(10, 2)
    labels = np.array([0, 1] * 5)
    metrics = {"trustworthiness": 0.9}
    loss_history = [0.5, 0.4, 0.3]
    evals = np.array([0.5, 0.3, 0.2])

    fig_static = viz_dr.plot_embedding(X_emb, labels, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    fig_interactive = viz_dr.plot_embedding(X_emb, labels, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    fig_static = viz_dr.plot_metrics(metrics, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    fig_interactive = viz_dr.plot_metrics(metrics, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    fig_static = viz_dr.plot_loss_history(loss_history, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    fig_interactive = viz_dr.plot_loss_history(loss_history, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    fig_static = viz_dr.plot_eigenvalues(evals, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    fig_interactive = viz_dr.plot_eigenvalues(evals, interactive=True)
    assert isinstance(fig_interactive, go.Figure)

    X_orig = np.random.rand(10, 5)
    fig_static = viz_dr.plot_shepard_diagram(X_orig, X_emb, interactive=False)
    assert isinstance(fig_static, plt.Figure)
    plt.close(fig_static)

    fig_interactive = viz_dr.plot_shepard_diagram(X_orig, X_emb, interactive=True)
    assert isinstance(fig_interactive, go.Figure)


def test_trajectory_plots():
    X = np.random.rand(2, 20, 2)
    times = np.linspace(0, 1, 20)
    labels = np.array([0, 1])

    fig = viz_dr.plot_trajectory(X, times=times, labels=labels, interactive=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = viz_dr.plot_trajectory(X, times=times, labels=labels, interactive=True)
    assert isinstance(fig, go.Figure)

    X_3d = np.random.rand(2, 20, 3)
    fig = viz_dr.plot_trajectory(
        X_3d, times=times, labels=labels, dimensions=3, interactive=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = viz_dr.plot_trajectory(
        X_3d, times=times, labels=labels, dimensions=3, interactive=True
    )
    assert isinstance(fig, go.Figure)


def test_trajectory_plot_rejects_flat_inputs():
    with pytest.raises(ValueError, match="3D trajectory tensor"):
        viz_dr.plot_trajectory(np.random.rand(20, 2), interactive=True)
