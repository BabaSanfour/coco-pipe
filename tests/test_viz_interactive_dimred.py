import numpy as np
import pytest

from coco_pipe.viz import interactive as viz_interactive


def test_interactive_dimred_plots_return_figure():
    import plotly.graph_objects as go

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
        lambda: viz_interactive.plot_embedding(X_emb, labels=np.arange(len(X_emb))),
        lambda: viz_interactive.plot_metrics(metric_df),
        lambda: viz_interactive.plot_loss_history([1.0, 0.7, 0.4]),
        lambda: viz_interactive.plot_scree(np.array([0.5, 0.3, 0.1])),
        lambda: viz_interactive.plot_shepard_diagram(X_orig, X_emb),
        lambda: viz_interactive.plot_streamlines(X_emb, V_emb),
        lambda: viz_interactive.plot_feature_importance({"F1": 0.6, "F2": 0.4}),
        lambda: viz_interactive.plot_feature_correlation_heatmap(corr_payload),
        lambda: viz_interactive.plot_trajectory_metric_series(
            np.vstack([np.arange(5), np.arange(5) + 1])
        ),
        lambda: viz_interactive.plot_trajectory(X_3d, dimensions=3),
        lambda: viz_interactive.plot_coranking_matrix(np.eye(8)),
        lambda: viz_interactive.plot_trajectory_separation({("A", "B"): np.arange(5)}),
        lambda: viz_interactive.plot_component_loadings(rng.normal(size=(4, 2))),
    ]
    for call in calls:
        fig = call()
        assert isinstance(fig, go.Figure)


def test_interactive_dimred_plots_args():
    rng = np.random.default_rng(42)
    rng.normal(size=(60, 2))
    metric_df = {"trustworthiness": 0.92, "continuity": 0.88}
    viz_interactive.plot_metrics(metric_df, plot_type="box")
    viz_interactive.plot_metrics(metric_df, plot_type="violin")
    viz_interactive.plot_metrics(metric_df, plot_type="heatmap")
    viz_interactive.plot_metrics(metric_df, plot_type="line")

    data = rng.normal(size=(2, 4, 50))
    viz_interactive.plot_timecourses(data)

    import pandas as pd

    df = pd.DataFrame({"trustworthiness": [0.9, 0.85]}, index=["UMAP", "t-SNE"])
    viz_interactive.plot_radar_comparison(df)
    viz_interactive.plot_raw_preview(data.reshape(8, 50))


def test_interactive_module_returns_plotly():
    import plotly.graph_objects as go

    X_emb = np.random.rand(10, 2)
    fig_interactive = viz_interactive.plot_embedding(X_emb)
    assert isinstance(fig_interactive, go.Figure)


def test_plot_phase_portrait_returns_figure():
    import plotly.graph_objects as go

    rng = np.random.default_rng(42)
    X = rng.normal(size=(3, 20, 5))
    times = np.linspace(0.0, 1.0, 20)
    labels = ["CondA", "CondB", "CondC"]

    fig = viz_interactive.plot_phase_portrait(X, times, labels)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    for trace, label in zip(fig.data, labels):
        assert trace.name == label
        assert len(trace.x) == 20
        assert len(trace.y) == 20


def test_plot_phase_portrait_component_idx():
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 15, 4))
    times = np.arange(15, dtype=float)
    fig = viz_interactive.plot_phase_portrait(
        X, times, labels=["X", "Y"], component_idx=2
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_phase_portrait_invalid_inputs():
    rng = np.random.default_rng(1)
    times = np.linspace(0, 1, 10)

    with pytest.raises(ValueError, match="3D"):
        viz_interactive.plot_phase_portrait(
            rng.normal(size=(3, 10)), times, ["A", "B", "C"]
        )

    X = rng.normal(size=(2, 10, 3))
    with pytest.raises(ValueError, match="times"):
        viz_interactive.plot_phase_portrait(X, np.linspace(0, 1, 5), ["A", "B"])

    with pytest.raises(ValueError, match="component_idx"):
        viz_interactive.plot_phase_portrait(X, times, ["A", "B"], component_idx=10)


def test_plot_metrics_plot_types():
    # 1. Dumbbell plot (requires exactly two methods)
    import pandas as pd
    import plotly.graph_objects as go

    df_two = pd.DataFrame(
        {
            "Metric": [
                "trustworthiness",
                "continuity",
                "trustworthiness",
                "continuity",
            ],
            "Method": ["UMAP", "UMAP", "t-SNE", "t-SNE"],
            "Value": [0.9, 0.8, 0.85, 0.75],
            "Scope": ["global", "global", "global", "global"],
        }
    )
    fig = viz_interactive.plot_metrics(df_two, plot_type="dumbbell")
    assert isinstance(fig, go.Figure)

    # Dumbbell raises if not exactly two methods
    df_three = pd.DataFrame(
        {
            "Metric": ["trustworthiness", "trustworthiness", "trustworthiness"],
            "Method": ["UMAP", "t-SNE", "PCA"],
            "Value": [0.9, 0.85, 0.7],
            "Scope": ["global", "global", "global"],
        }
    )
    with pytest.raises(ValueError, match="two methods"):
        viz_interactive.plot_metrics(df_three, plot_type="dumbbell")

    # 2. Unsupported plot type
    with pytest.raises(ValueError, match="Unsupported plot_type"):
        viz_interactive.plot_metrics(df_two, plot_type="invalid_type")


def test_plot_trajectory_sem_and_values():
    import plotly.graph_objects as go

    rng = np.random.default_rng(42)

    # Shape: (n_trajectories, n_times, n_dimensions) -> (2, 20, 3)
    X = rng.normal(size=(2, 20, 3))

    # Mismatched SEM shape raises ValueError
    sem_mismatched = rng.normal(size=(2, 10, 3))
    with pytest.raises(ValueError, match="must match.*shape"):
        viz_interactive.plot_trajectory(X, sem=sem_mismatched)

    # Matching SEM shape and values, 3D
    sem_matching_3d = rng.normal(size=(2, 20, 3))
    values_matching = rng.normal(size=(2, 20))
    fig_3d = viz_interactive.plot_trajectory(
        X, sem=sem_matching_3d, values=values_matching, dimensions=3
    )
    assert isinstance(fig_3d, go.Figure)

    # Matching SEM shape and values, 2D
    X_2d = rng.normal(size=(2, 20, 2))
    sem_matching_2d = rng.normal(size=(2, 20, 2))
    fig_2d = viz_interactive.plot_trajectory(
        X_2d, sem=sem_matching_2d, values=values_matching, dimensions=2
    )
    assert isinstance(fig_2d, go.Figure)

    fig_pad = viz_interactive.plot_trajectory(
        X_2d, sem=sem_matching_2d, smooth_window=5, dimensions=2
    )
    assert isinstance(fig_pad, go.Figure)
