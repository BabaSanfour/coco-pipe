import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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


def test_trajectory_plots():
    X = np.random.rand(20, 2)
    times = np.linspace(0, 1, 20)
    groups = np.array([0] * 10 + [1] * 10)

    # 1. Static 2D Trajectory
    fig = viz_dr.plot_trajectory(X, times=times, groups=groups, interactive=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # 2. Interactive 2D Trajectory
    fig = viz_dr.plot_trajectory(X, times=times, groups=groups, interactive=True)
    assert isinstance(fig, go.Figure)

    # 3. Static 3D Trajectory
    X_3d = np.random.rand(20, 3)
    fig = viz_dr.plot_trajectory(
        X_3d, times=times, groups=groups, dimensions=3, interactive=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # 4. Interactive 3D Trajectory
    fig = viz_dr.plot_trajectory(
        X_3d, times=times, groups=groups, dimensions=3, interactive=True
    )
    assert isinstance(fig, go.Figure)


def test_trajectory_plot_interactive_accepts_3d_tensor():
    # (n_trajectories, n_times, n_dims)
    X = np.random.rand(4, 25, 3)
    fig = viz_dr.plot_trajectory(X, interactive=True, dimensions=3)
    assert isinstance(fig, go.Figure)


def test_trajectory_plot_interactive_accepts_container():
    n_obs, n_times, n_dims = 5, 30, 3
    X = np.random.randn(n_obs, n_times, n_dims)

    class DummyContainer:
        def __init__(self, X, dims, coords, y):
            self.X = X
            self.dims = dims
            self.coords = coords
            self.y = y

    container = DummyContainer(
        X=X,
        dims=("obs", "time", "component"),
        coords={"time": np.linspace(0.0, 1.5, n_times)},
        y=np.array([0, 1, 0, 1, 0]),
    )
    fig = viz_dr.plot_trajectory(container, interactive=True, dimensions=3)
    assert isinstance(fig, go.Figure)
