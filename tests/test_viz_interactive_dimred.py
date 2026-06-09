import numpy as np
import pytest

from coco_pipe.viz import interactive as viz_interactive


def test_interactive_dimred_plots_return_figure():
    go = pytest.importorskip("plotly.graph_objects")
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
        lambda: viz_interactive.plot_eigenvalues({"cond": np.array([0.5, 0.3, 0.1])}),
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
    viz_interactive.plot_channel_traces(data)

    import pandas as pd

    df = pd.DataFrame({"trustworthiness": [0.9, 0.85]}, index=["UMAP", "t-SNE"])
    viz_interactive.plot_radar_comparison(df)
    viz_interactive.plot_raw_preview(data.reshape(8, 50))


def test_interactive_module_returns_plotly():
    go = pytest.importorskip("plotly.graph_objects")
    X_emb = np.random.rand(10, 2)
    fig_interactive = viz_interactive.plot_embedding(X_emb)
    assert isinstance(fig_interactive, go.Figure)
