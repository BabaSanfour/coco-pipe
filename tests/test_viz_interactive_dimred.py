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
        lambda: viz_interactive.plot_reduction_feature_importance(
            {"F1": 0.6, "F2": 0.4}
        ),
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


def test_interactive_shepard_clip_and_scatter_downsample():
    import plotly.graph_objects as go

    rng = np.random.default_rng(3)
    X_orig = rng.normal(size=(80, 6))
    X_emb = rng.normal(size=(80, 2))
    fig = viz_interactive.plot_shepard_diagram(
        X_orig,
        X_emb,
        clip_quantiles=(0.05, 0.95),
        scatter_max_points=50,
        sample_size=80,
        random_state=0,
    )
    assert isinstance(fig, go.Figure)


def test_interactive_radar_constant_column_and_no_normalize():
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(
        {"trustworthiness": [0.9, 0.9], "continuity": [0.7, 0.85]},
        index=["UMAP", "t-SNE"],
    )
    # constant column triggers the max==min branch
    assert isinstance(viz_interactive.plot_radar_comparison(df), go.Figure)
    # normalize=False uses an unbounded radial axis
    assert isinstance(
        viz_interactive.plot_radar_comparison(df, normalize=False), go.Figure
    )


def test_interactive_feature_correlation_branches():
    import pandas as pd
    import plotly.graph_objects as go

    records = []
    for method in ("UMAP", "t-SNE"):
        for feature in [f"F{i}" for i in range(6)]:
            for dim in ("D1", "D2"):
                records.append(
                    {
                        "Method": method,
                        "Feature": feature,
                        "Dimension": dim,
                        "Analysis": "correlation",
                        "Value": 0.5,
                    }
                )
    frame = pd.DataFrame(records)

    with pytest.raises(ValueError, match="top_n"):
        viz_interactive.plot_feature_correlation_heatmap(frame, top_n=0)
    with pytest.raises(ValueError, match="Specify `method`"):
        viz_interactive.plot_feature_correlation_heatmap(frame)
    with pytest.raises(ValueError, match="No correlation records"):
        viz_interactive.plot_feature_correlation_heatmap(
            frame[frame["Method"] == "missing"]
        )
    fig = viz_interactive.plot_feature_correlation_heatmap(
        frame, method="UMAP", top_n=3
    )
    assert isinstance(fig, go.Figure)


def test_interactive_trajectory_metric_series_styling_and_smoothing():
    import plotly.graph_objects as go

    rng = np.random.default_rng(7)
    # 2D (trajectory, time) input aggregates to one series with a SEM band.
    series = rng.normal(size=(5, 12))
    fig = viz_interactive.plot_trajectory_metric_series(
        series,
        smooth_window=3,
        color_map={"Series": "#1f77b4"},
        linestyle_map={"Series": "--"},
    )
    assert isinstance(fig, go.Figure)


def test_interactive_trajectory_2d_and_separation_topn():
    import plotly.graph_objects as go

    rng = np.random.default_rng(11)
    X2d = rng.normal(size=(3, 18, 2))
    values = rng.normal(size=(3, 18))
    sem = np.abs(rng.normal(size=(3, 18, 2))) * 0.1
    fig = viz_interactive.plot_trajectory(
        X2d,
        labels=["A", "B", "C"],
        values=values,
        sem=sem,
        dimensions=2,
    )
    assert isinstance(fig, go.Figure)

    separation = {
        ("A", "B"): np.arange(8),
        ("A", "C"): np.arange(8) * 0.5,
        ("B", "C"): np.arange(8) * 0.2,
    }
    fig_sep = viz_interactive.plot_trajectory_separation(
        separation, times=np.linspace(0, 1, 8), top_n=2
    )
    assert isinstance(fig_sep, go.Figure)


def test_plot_embedding_3d_scatter():
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    fig = viz_interactive.plot_embedding(X, dimensions=3)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Scatter3d) for t in fig.data)


def test_plot_embedding_with_metadata_dropdown():
    import plotly.graph_objects as go

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    labels = np.tile([0, 1], 10)
    metadata = {"Group": list("ABCDEFGHIJKLMNOPQRST")}
    fig = viz_interactive.plot_embedding(X, labels=labels, metadata=metadata)
    assert isinstance(fig, go.Figure)
    # Dropdown menus are added when len(color_columns) > 1
    assert fig.layout.updatemenus is not None and len(fig.layout.updatemenus) > 0


def test_plot_metrics_empty_raises():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Metric": ["trustworthiness"],
            "Method": ["UMAP"],
            "Value": [0.9],
            "Scope": ["global"],
        }
    )
    with pytest.raises(ValueError, match="No metrics available"):
        viz_interactive.plot_metrics(df, metric="nonexistent_metric")


def test_plot_raw_preview_subsampling_guard():
    import plotly.graph_objects as go

    rng = np.random.default_rng(5)
    data = rng.normal(size=(50, 10))  # 500 total < 50000, no subsampling
    fig = viz_interactive.plot_raw_preview(data, max_points=100)
    assert isinstance(fig, go.Figure)
    # Also verify channel names are used when supplied
    names = [f"EEG{i}" for i in range(10)]
    fig2 = viz_interactive.plot_raw_preview(data, names=names, max_points=100)
    assert isinstance(fig2, go.Figure)


def test_plot_shepard_no_clip_quantiles():
    import plotly.graph_objects as go

    rng = np.random.default_rng(9)
    X_orig = rng.normal(size=(40, 4))
    X_emb = rng.normal(size=(40, 2))
    fig = viz_interactive.plot_shepard_diagram(
        X_orig, X_emb, clip_quantiles=None, sample_size=40
    )
    assert isinstance(fig, go.Figure)


def test_plot_shepard_fallback_few_points():
    import plotly.graph_objects as go

    rng = np.random.default_rng(22)
    # Very small dataset – most points will be outside the window, triggering fallback
    X_orig = rng.normal(size=(10, 3))
    X_emb = rng.normal(size=(10, 2))
    fig = viz_interactive.plot_shepard_diagram(
        X_orig, X_emb, sample_size=10, clip_quantiles=(0.01, 0.99)
    )
    assert isinstance(fig, go.Figure)


def test_plot_streamlines_large_input_subsampling():
    import plotly.graph_objects as go

    rng = np.random.default_rng(13)
    X = rng.normal(size=(1200, 2))
    V = rng.normal(size=(1200, 2))
    fig = viz_interactive.plot_streamlines(X, V, random_state=0)
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_metric_series_color_dash_fillcolor_and_layout_kwargs():
    import plotly.graph_objects as go

    rng = np.random.default_rng(14)
    series = rng.normal(size=(3, 15))
    # Named dict input produces one series per key; color triggers mcolors branch
    named = {"Alpha": rng.normal(size=15), "Beta": rng.normal(size=15)}
    fig = viz_interactive.plot_trajectory_metric_series(
        named,
        smooth_window=3,
        color_map={"Alpha": "#e41a1c", "Beta": "#377eb8"},
        linestyle_map={"Alpha": "--", "Beta": ":"},
        height=350,
    )
    assert isinstance(fig, go.Figure)

    fig2 = viz_interactive.plot_trajectory_metric_series(series, smooth_window=2)
    assert isinstance(fig2, go.Figure)


def test_sem_envelope_traces_empty_and_edge_cases():
    from coco_pipe.viz.interactive.dim_reduction import _sem_envelope_traces

    # n_times == 0 returns []
    traj_empty = np.zeros((0, 2))
    sem_empty = np.zeros((0, 2))
    result = _sem_envelope_traces(
        traj_empty,
        sem_empty,
        color="#ff0000",
        name="T",
        dimensions=2,
        sem_alpha=0.2,
        sem_n_steps=4,
    )
    assert result == []

    rng = np.random.default_rng(99)
    traj = np.abs(rng.normal(size=(5, 2))) + 0.1
    sem = np.abs(rng.normal(size=(5, 2))) * 0.3 + 0.05
    result_nonhex = _sem_envelope_traces(
        traj, sem, color="red", name="T", dimensions=2, sem_alpha=0.2, sem_n_steps=4
    )
    # Should produce traces (color is passed through)
    assert isinstance(result_nonhex, list)

    traj3d = np.abs(rng.normal(size=(5, 3))) + 0.1
    sem3d = np.abs(rng.normal(size=(5, 3))) * 0.3 + 0.05
    result_3d = _sem_envelope_traces(
        traj3d,
        sem3d,
        color="#00ff00",
        name="T3",
        dimensions=3,
        sem_alpha=0.15,
        sem_n_steps=3,
    )
    import plotly.graph_objects as go

    assert all(isinstance(t, go.Scatter3d) for t in result_3d)


def test_plot_trajectory_sem_padding_and_color_map():
    import plotly.graph_objects as go

    rng = np.random.default_rng(55)
    # downsample=2: sem_arr after downsampling may be shorter than trajectory
    X = rng.normal(size=(2, 12, 2))
    sem = np.abs(rng.normal(size=(2, 12, 2))) * 0.2
    fig = viz_interactive.plot_trajectory(
        X,
        sem=sem,
        downsample=2,
        dimensions=2,
        color_map={"Trajectory 1": "#e41a1c", "Trajectory 2": "#377eb8"},
    )
    assert isinstance(fig, go.Figure)

    X2 = rng.normal(size=(2, 10, 2))
    sem2 = np.abs(rng.normal(size=(2, 10, 2))) * 0.1
    fig2 = viz_interactive.plot_trajectory(
        X2,
        labels=["CondA", "CondB"],
        sem=sem2,
        color_map={"CondA": "#e41a1c", "CondB": "#377eb8"},
        dimensions=2,
    )
    assert isinstance(fig2, go.Figure)

    # linestyle_map exercises the dash lookup path alongside color_map.
    fig3 = viz_interactive.plot_trajectory(
        X2,
        labels=["CondA", "CondB"],
        linestyle_map={"CondA": "dash", "CondB": "dot"},
        dimensions=2,
    )
    assert isinstance(fig3, go.Figure)


def test_plot_trajectory_add_start_end_markers_and_layout_kws():
    import plotly.graph_objects as go

    rng = np.random.default_rng(77)
    X = rng.normal(size=(2, 8, 3))
    fig = viz_interactive.plot_trajectory(
        X,
        dimensions=3,
        add_start_end_markers=True,
        width=800,
        height=600,
        layout_kws={"paper_bgcolor": "white"},
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_plot_trajectory_separation_smooth_color_linestyle_layout_kwargs():
    import plotly.graph_objects as go

    sep = {
        ("A", "B"): np.linspace(0, 1, 10),
        ("A", "C"): np.linspace(0.5, 0, 10),
    }
    fig = viz_interactive.plot_trajectory_separation(
        sep,
        times=np.linspace(0, 1, 10),
        smooth_window=3,
        color_map={("A", "B"): "#e41a1c", ("A", "C"): "#377eb8"},
        linestyle_map={("A", "B"): "--", ("A", "C"): "-."},
        height=300,
    )
    assert isinstance(fig, go.Figure)


def test_plot_feature_importance_basic():
    import plotly.graph_objects as go

    scores = {"Feature_A": 0.8, "Feature_B": 0.6, "Feature_C": 0.3}
    fig = viz_interactive.plot_reduction_feature_importance(scores)
    assert isinstance(fig, go.Figure)
    # Verify top_n limits the number of bars shown
    fig2 = viz_interactive.plot_reduction_feature_importance(scores, top_n=2)
    assert isinstance(fig2, go.Figure)


def test_plot_coranking_matrix_max_k():
    """plot_coranking_matrix with explicit max_k crops the matrix."""
    import plotly.graph_objects as go

    matrix = np.eye(20)
    fig = viz_interactive.plot_coranking_matrix(matrix, max_k=5)
    assert isinstance(fig, go.Figure)


def test_plot_component_loadings_with_feature_names_and_n_components():
    """plot_component_loadings with feature_names and n_components args."""
    import plotly.graph_objects as go

    rng = np.random.default_rng(3)
    components = rng.normal(size=(8, 4))
    feature_names = [f"feat_{i}" for i in range(8)]
    fig = viz_interactive.plot_component_loadings(
        components, feature_names=feature_names, n_components=2
    )
    assert isinstance(fig, go.Figure)
