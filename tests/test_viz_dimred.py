import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    import plotly.graph_objects as go

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


# plot_metrics fixtures shared across parametrized render cases.
_METRICS_DICT = {"trustworthiness": 0.92, "continuity": 0.88}


def _multi_method_df():
    return pd.DataFrame(
        {
            "Method": ["A", "A", "B", "B"],
            "Metric": ["m1", "m2", "m1", "m2"],
            "Value": [0.9, 0.8, 0.7, 0.6],
        }
    )


def _scope_df(scope, scope_values):
    return pd.DataFrame(
        {
            "Method": ["A"] * 3,
            "Metric": ["m1"] * 3,
            "Value": [0.9, 0.85, 0.8],
            "Scope": [scope] * 3,
            "ScopeValue": scope_values,
        }
    )


@pytest.mark.parametrize(
    "call",
    [
        # dict input: every supported plot_type branch.
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="bar"), id="dict-bar"
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="grouped_bar"),
            id="dict-grouped_bar",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="lollipop"),
            id="dict-lollipop",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="box"), id="dict-box"
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="boxen"), id="dict-boxen"
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="violin"),
            id="dict-violin",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="strip"), id="dict-strip"
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="swarm"), id="dict-swarm"
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="raincloud"),
            id="dict-raincloud",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_METRICS_DICT, plot_type="heatmap"),
            id="dict-heatmap",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_multi_method_df(), plot_type="bar"),
            id="multidf-bar",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_multi_method_df(), plot_type="heatmap"),
            id="multidf-heatmap",
        ),
        pytest.param(
            lambda: viz.plot_metrics(_multi_method_df(), plot_type="dumbbell"),
            id="multidf-dumbbell",
        ),
        # scoped frames feed the line plot_type.
        pytest.param(
            lambda: viz.plot_metrics(
                _scope_df("fold", ["1", "2", "3"]), plot_type="line"
            ),
            id="line-str_scope",
        ),
        pytest.param(
            lambda: viz.plot_metrics(
                _scope_df("n_components", [2, 5, 10]), plot_type="line"
            ),
            id="line-numeric_scope",
        ),
        # method/metric filtering with the default plot_type.
        pytest.param(
            lambda: viz.plot_metrics(_multi_method_df(), method="A", metric="m1"),
            id="filtering",
        ),
    ],
)
def test_plot_metrics_renders(call):
    _assert_fig_ax(call())


def test_plot_metrics_invalid_type():
    with pytest.raises(ValueError, match="Unsupported"):
        viz.plot_metrics(_METRICS_DICT, plot_type="invalid")


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: viz.plot_embedding(
                np.random.randn(20, 3), dims=(0, 1, 2), figsize=(5, 5)
            ),
            id="3d",
        ),
        pytest.param(
            lambda: viz.plot_embedding(
                np.random.randn(20, 2),
                labels=np.arange(20, dtype=float),
                label_kind="continuous",
            ),
            id="continuous",
        ),
        pytest.param(
            lambda: viz.plot_embedding(np.random.randn(20, 2)), id="no_labels"
        ),
    ],
)
def test_plot_embedding_renders(call):
    _assert_fig_ax(call())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(3, 20, 3),
                values=np.random.randn(3, 20),
                labels=["A", "B", "C"],
            ),
            id="with_values",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 2),
                values=np.random.randn(2, 10),
                speed_mode="alpha",
            ),
            id="values_alpha_mode",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 3), values=np.random.randn(2, 10), dimensions=3
            ),
            id="3d_with_values",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 2), labels=["A", "B"], add_start_end_markers=True
            ),
            id="start_end_markers",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 2),
                labels=["A", "B"],
                xlim=(-2, 2),
                ylim=(-2, 2),
                axis_labels=["PC1", "PC2"],
                axes_kws={"xlim": (-3, 3)},
            ),
            id="axes_kws",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 2),
                axes_kws={
                    "tick_params": {"labelsize": 8},
                    "locator_params": {"axis": "x", "nbins": 5},
                    "labelsize": 10,
                },
            ),
            id="axes_kws_with_params",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 3), dimensions=3, axes_kws={"labelsize": 12}
            ),
            id="3d_axes_kws",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 10, 2),
                labels=["A", "B"],
                color_map={"A": "red", "B": "blue"},
                linestyle_map={"A": "solid", "B": "dash"},
            ),
            id="color_map_and_linestyle",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(np.random.randn(2, 10, 2), showlegend=False),
            id="no_legend",
        ),
        pytest.param(
            lambda: viz.plot_trajectory(
                np.random.randn(2, 20, 2), smooth_window=3, downsample=2
            ),
            id="smooth_and_downsample",
        ),
    ],
)
def test_plot_trajectory_variations(call):
    _assert_fig_ax(call())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: viz.plot_trajectory_metric_series(
                {"A": np.random.randn(20), "B": np.random.randn(20)}, smooth_window=3
            ),
            id="metric_series_smooth",
        ),
        pytest.param(
            lambda: viz.plot_trajectory_metric_series(
                {"A": np.random.randn(10), "B": np.random.randn(10)},
                color_map={"A": "red", "B": "blue"},
                linestyle_map={"A": "dash", "B": "dot"},
            ),
            id="metric_series_color_and_linestyle",
        ),
        pytest.param(
            lambda: viz.plot_trajectory_separation(
                {("A", "B"): np.random.randn(20)}, smooth_window=3
            ),
            id="separation_smooth",
        ),
        pytest.param(
            lambda: viz.plot_trajectory_separation(
                {("A", "B"): np.arange(10), ("C", "D"): np.arange(10)},
                color_map={("A", "B"): "red"},
                linestyle_map={("A", "B"): "dash"},
            ),
            id="separation_color_and_linestyle",
        ),
    ],
)
def test_plot_trajectory_series_and_separation(call):
    _assert_fig_ax(call())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: viz.plot_phase_portrait(
                np.random.default_rng(42).normal(size=(3, 20, 5)),
                np.linspace(0.0, 1.0, 20),
                labels=["A", "B", "C"],
            ),
            id="basic",
        ),
        pytest.param(
            lambda: viz.plot_phase_portrait(
                np.random.default_rng(42).normal(size=(2, 10, 4)),
                np.linspace(0, 1, 10),
                labels=["A", "B"],
                component_idx=2,
            ),
            id="component_idx",
        ),
    ],
)
def test_plot_phase_portrait_renders(call):
    _assert_fig_ax(call())


def test_plot_phase_portrait_errors():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="3D"):
        viz.plot_phase_portrait(
            rng.normal(size=(3, 10)), np.linspace(0, 1, 10), ["A", "B", "C"]
        )
    X = rng.normal(size=(2, 10, 3))
    with pytest.raises(ValueError, match="times"):
        viz.plot_phase_portrait(X, np.linspace(0, 1, 5), ["A", "B"])
    with pytest.raises(ValueError, match="component_idx"):
        viz.plot_phase_portrait(X, np.linspace(0, 1, 10), ["A", "B"], component_idx=10)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: viz.plot_scree(np.array([0.4, 0.2, 0.1, 0.05, 0.03])), id="default"
        ),
        pytest.param(
            lambda: viz.plot_scree(np.array([0.4, 0.2, 0.1]), title=None), id="no_title"
        ),
    ],
)
def test_plot_scree_renders(call):
    _assert_fig_ax(call())


def test_plot_eigenvalues_with_conditions():
    vals = {"cond_A": np.random.rand(5), "cond_B": np.random.rand(5)}
    fig, ax = viz.plot_eigenvalues(vals, condition_colors={"cond_A": "red"})
    plt.close(fig)


def test_plot_feature_correlation_heatmap_top_n():
    payload = {
        "correlation": {
            "D1": {f"F{i}": float(i) / 30 for i in range(30)},
            "D2": {f"F{i}": float(30 - i) / 30 for i in range(30)},
        }
    }
    fig, ax = viz.plot_feature_correlation_heatmap(payload, top_n=10)
    plt.close(fig)


def test_plot_feature_correlation_heatmap_invalid_top_n():
    with pytest.raises(ValueError, match="positive"):
        viz.plot_feature_correlation_heatmap(
            {"correlation": {"D1": {"F1": 0.5}}}, top_n=0
        )


def test_plot_coranking_matrix_max_k():
    Q = np.random.rand(100, 100)
    fig, ax = viz.plot_coranking_matrix(Q, max_k=20)
    plt.close(fig)


def test_plot_coranking_matrix_non_square():
    with pytest.raises(ValueError, match="square"):
        viz.plot_coranking_matrix(np.random.rand(3, 4))
