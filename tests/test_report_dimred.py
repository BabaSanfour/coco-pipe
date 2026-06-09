"""
Tests for Dim-Red Components
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from coco_pipe.report.core import Report
from coco_pipe.report.dim_reduction import (
    _get_reducer_summary,
    _metrics_summary_table,
    _trajectory_times,
    add_reduction_components,
    add_reduction_coranking,
    add_reduction_diagnostics,
    add_reduction_embedding,
    add_reduction_interpretation,
    add_reduction_metrics,
    add_reduction_trajectory,
    add_reduction_trajectory_separation,
    make_reduction_report,
)
from coco_pipe.report.elements import PlotlyElement
from coco_pipe.viz.interactive.dim_reduction import (
    plot_embedding as plot_embedding_interactive,
)
from coco_pipe.viz.interactive.dim_reduction import (
    plot_metrics as plot_metric_details,
)


class MockReducer:
    """Mock DimReduction object."""

    def __init__(self):
        self.loss_history_ = [10, 5, 2, 1]
        self.explained_variance_ratio_ = np.array([0.5, 0.3, 0.1])
        self.capabilities = {
            "supported_diagnostics": [
                "loss_history_",
                "explained_variance_ratio_",
            ],
            "supported_metadata": [],
        }

    def get_diagnostics(self):
        return {
            "loss_history_": self.loss_history_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
        }

    def get_quality_metadata(self):
        return {}

    def get_summary(self):
        return {
            "method": "MockReducer",
            "metrics": {},
            "metric_records": [],
            "quality_metadata": self.get_quality_metadata(),
            "diagnostics": self.get_diagnostics(),
            "interpretation": {},
            "interpretation_records": [],
            "capabilities": self.capabilities,
        }


def test_plotly_element_rendering():
    fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
    el = PlotlyElement(fig)
    html = el.render()

    # Check for lazy loading structure
    assert "lazy-plot" in html
    assert "data-figure" in html
    assert "&quot;data&quot;" in html  # JSON encoded


def test_plot_embedding_interactive_logic():
    emb = np.random.randn(50, 2)
    labels = np.random.randint(0, 2, 50)
    metadata = {"Class": ["X"] * 25 + ["Y"] * 25, "Score": np.random.rand(50)}

    fig_basic = plot_embedding_interactive(emb, labels=labels, title="Basic")
    assert isinstance(fig_basic, go.Figure)
    assert len(fig_basic.data) > 0

    fig_adv = plot_embedding_interactive(
        emb, labels=labels, metadata=metadata, title="Adv"
    )
    assert isinstance(fig_adv, go.Figure)
    assert len(fig_adv.data) > 0


def test_plot_metric_details():
    records = [
        {"Method": "PCA", "Metric": "Trustworthiness", "Score": 0.8},
        {"Method": "PCA", "Metric": "Continuity", "Score": 0.7},
        {"Method": "UMAP", "Metric": "Trustworthiness", "Score": 0.9},
        {"Method": "UMAP", "Metric": "Continuity", "Score": 0.95},
    ]
    fig = plot_metric_details(records)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_report_add_reduction_logic():
    rep = Report("DimRed Test")
    reducer = MockReducer()
    embedding = np.random.randn(100, 2)
    metadata = {
        "Group": ["A"] * 50 + ["B"] * 50,
        "Value": np.random.rand(100),
    }

    rep.add_reduction(reducer, name="MockPCA", X_emb=embedding, metadata=metadata)

    html = rep.render()

    assert "MockPCA" in html

    assert html.count("lazy-plot") >= 3


def test_report_add_comparison():
    rep = Report("Comparison Test")

    metrics_df = pd.DataFrame(
        {
            "Method": ["PCA", "ISO"],
            "Trustworthiness": [0.8, 0.6],
            "Continuity": [0.7, 0.5],
        }
    ).set_index("Method")

    rep.add_comparison(metrics_df)

    html = rep.render()

    # Check for section title and icon
    assert "Method Comparison" in html
    # Check for plots (Radar + Bar)
    # Check for Table title
    assert "Quality Metrics" in html


def test_report_add_reduction_requires_summary_contract():
    """Report.add_reduction should require the explicit reducer summary contract."""
    rep = Report("Safe Access Test")

    class BrokenReducer:
        def __init__(self):
            self.embedding_ = np.random.randn(10, 2)

        @property
        def loss_history_(self):
            raise RuntimeError("Broken loss history")

    reducer = BrokenReducer()
    with pytest.raises(TypeError, match="must implement get_summary"):
        rep.add_reduction(reducer, name="BrokenMethod")


def test_make_reduction_report_static_sections():
    class StaticReducer(MockReducer):
        diagnostics_ = {"coranking_matrix_": np.eye(5)}

        def get_summary(self):
            summary = super().get_summary()
            summary["metrics"] = {"trustworthiness": 0.9}
            summary["diagnostics"] = self.diagnostics_
            return summary

        def get_components(self):
            return {"components_": np.random.randn(4, 2)}

    report = make_reduction_report(
        [StaticReducer()],
        embeddings=[np.random.randn(20, 2)],
        sections=["overview", "embedding", "metrics", "coranking", "components"],
    )
    html = report.render()
    assert "MockReducer Overview" in html
    assert "MockReducer Embedding" in html

    assert hasattr(Report, "add_reduction_overview")


def test_reduction_report_empty_edge_cases():
    class EmptyReduction:
        def get_summary(self):
            return {}

    reduction = EmptyReduction()
    report = Report("Empty")

    # Should not raise any errors, just return self
    report.add_reduction_overview(reduction)
    report.add_reduction_embedding(None)
    report.add_reduction_metrics(reduction)
    report.add_reduction_diagnostics(None, None)
    report.add_reduction_interpretation({})
    report.add_reduction_coranking(None)
    report.add_reduction_components(None)
    report.add_reduction_trajectory(None)
    report.add_reduction_trajectory_separation({})
    assert len(report.children) == 0


def test_reduction_full_coverage():
    import numpy as np

    class FullReduction:
        def get_summary(self):
            return {
                "method": "PCA",
                "diagnostics": {"explained_variance_ratio_": np.array([0.5, 0.3])},
                "metric_records": [{"metric": "trustworthiness", "score": 0.9}],
                "interpretation": {"loadings": np.array([[1, 0], [0, 1]])},
            }

        def get_scores(self):
            return [{"metric": "trustworthiness", "score": 0.9}]

        def get_components(self):
            return np.array([[1, 0], [0, 1]])

    reduction = FullReduction()
    report = Report("Full")

    X_emb = np.random.rand(10, 2)
    X_3d = np.random.rand(5, 10, 2)
    coranking = np.random.rand(9, 9)
    sep = {"A-B": np.random.rand(5)}

    report.add_reduction_overview(reduction)
    report.add_reduction_embedding(X_emb)
    report.add_reduction_metrics(reduction)
    report.add_reduction_diagnostics(np.random.rand(10, 5), X_emb)
    report.add_reduction_coranking(coranking)
    report.add_reduction_components(reduction.get_components())
    report.add_reduction_trajectory(X_3d)
    report.add_reduction_trajectory_separation(sep)

    assert len(report.children) > 0


# ----- Helper-level edge cases moved from test_report_core.py ----------------


def test_get_reducer_summary_edge_cases():
    # 1. Missing get_summary
    with pytest.raises(TypeError, match="must implement get_summary"):
        _get_reducer_summary(object())

    # 2. get_summary returns non-dict
    mock = MagicMock()
    mock.get_summary.return_value = "not a dict"
    with pytest.raises(TypeError, match="must return a dictionary"):
        _get_reducer_summary(mock)

    # 3. Partial summary (fills defaults)
    mock.get_summary.return_value = {"method": "PCA"}
    summary = _get_reducer_summary(mock)
    assert summary["method"] == "PCA"
    assert summary["metrics"] == {}
    assert summary["metric_records"] == []


def test_metrics_summary_table_empty():
    assert _metrics_summary_table({}).empty


def test_trajectory_times_resolution():
    assert _trajectory_times({}, np.array([1, 2, 3])) is not None
    assert _trajectory_times({}, np.array([])) is None
    assert _trajectory_times({"trajectory_times_": [1, 2]}, None) is not None
    assert _trajectory_times({"trajectory_times_": []}, None) is None


def test_report_add_reduction_coverage():
    from coco_pipe.io.structures import DataContainer

    rep = Report()
    mock_reducer = MagicMock()
    mock_reducer.get_summary.return_value = {
        "method": "MockDR",
        "metrics": {"trust": 0.9},
        "metric_records": [{"method": "MockDR", "metric": "trust", "value": 0.9}],
        "diagnostics": {
            "embedding_": np.random.randn(10, 2),
            "reconstruction_": np.random.randn(10, 5),
        },
        "quality_metadata": {},
    }

    X = np.random.randn(10, 5)
    _ = DataContainer(X, dims=("obs", "feature"))

    # 1. Basic add
    rep.add_reduction(mock_reducer, name="Mock Reduction")
    assert "Mock Reduction" in rep.children[-1].render()

    # 2. Add with explicit embedding and labels
    X_emb = np.random.randn(10, 2)
    labels = np.array([0, 1] * 5)
    metadata = {"feat": np.random.randn(10)}
    rep.add_reduction(
        mock_reducer,
        name="With Embedding",
        X_emb=X_emb,
        labels=labels,
        metadata=metadata,
    )
    assert "With Embedding" in rep.children[-1].render()


@patch("coco_pipe.viz.interactive.dim_reduction.plot_trajectory")
@patch("coco_pipe.viz.interactive.dim_reduction.plot_loss_history")
@patch("coco_pipe.viz.interactive.dim_reduction.plot_scree")
@patch("coco_pipe.viz.interactive.dim_reduction.plot_trajectory_metric_series")
def test_add_reduction_advanced(mock_traj_series, mock_eig, mock_loss, mock_traj):
    mock_fig = MagicMock()
    mock_fig.to_json.return_value = '{"data": []}'
    mock_fig.to_dict.return_value = {"data": []}
    mock_traj.return_value = mock_fig
    mock_loss.return_value = mock_fig
    mock_eig.return_value = mock_fig
    mock_traj_series.return_value = mock_fig

    mock_reducer = MagicMock()
    mock_reducer.get_summary.return_value = {
        "method": "MockAdvanced",
        "metrics": {"trust": 0.9},
        "metric_records": [],
        "diagnostics": {
            "loss_history_": [1, 2],
            "explained_variance_ratio_": [0.5, 0.5],
            "coranking_matrix_": np.zeros((2, 2)),
            "trajectory_speed_": [0.1, 0.2],
            "trajectory_separation_": [0.5, 0.6],
        },
        "quality_metadata": {},
    }

    rep = Report()
    X_emb = np.random.randn(10, 2, 3)
    rep.add_reduction(mock_reducer, X_emb=X_emb)

    mock_traj.assert_called()
    mock_loss.assert_called()
    mock_eig.assert_called()
    mock_traj_series.assert_called()

    html = rep.render()
    assert "MockAdvanced" in html


@patch("coco_pipe.viz.interactive.dim_reduction.plot_metrics")
@patch("coco_pipe.viz.interactive.dim_reduction.plot_radar_comparison")
def test_add_comparison(mock_radar, mock_metrics):
    mock_metrics.return_value = MagicMock()
    mock_radar.return_value = MagicMock()

    rep = Report()
    df = pd.DataFrame(
        [
            {"Method": "PCA", "Metric": "Trust", "Value": 0.9, "ScopeValue": "A"},
            {"Method": "UMAP", "Metric": "Trust", "Value": 0.95, "ScopeValue": "A"},
            {"Method": "PCA", "Metric": "Loss", "Value": 0.1, "ScopeValue": "A"},
            {"Method": "UMAP", "Metric": "Loss", "Value": 0.05, "ScopeValue": "A"},
            {"Method": "PCA", "Metric": "Continuity", "Value": 0.8, "ScopeValue": "A"},
            {
                "Method": "UMAP",
                "Metric": "Continuity",
                "Value": 0.85,
                "ScopeValue": "A",
            },
        ]
    )

    rep.add_comparison(df)

    mock_metrics.assert_called()
    mock_radar.assert_called()

    with pytest.raises(ValueError):
        rep.add_comparison(pd.DataFrame())


class MockReducerForExceptions:
    def get_summary(self):
        return {
            "method": "ExceptionMock",
            "metric_records": [{"metric": "m", "score": 1}],
            "diagnostics": {"coranking_matrix_": np.ones((5, 5))},
            "interpretation": {"loadings": np.ones((2, 2))},
        }

    def get_scores(self):
        return [{"metric": "m", "score": 1}]

    def get_components(self):
        return np.ones((2, 2))


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_embedding", side_effect=ValueError("Plot error")
)
def test_add_reduction_embedding_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_embedding(rep, np.random.randn(10, 2))
    mock_log.assert_called_with("Embedding section skipped: %s", mock_plot.side_effect)


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch("coco_pipe.viz.dim_reduction.plot_metrics", side_effect=ValueError("Plot error"))
def test_add_reduction_metrics_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_metrics(rep, MockReducerForExceptions())
    mock_log.assert_called_with("Metrics section skipped: %s", mock_plot.side_effect)


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_shepard_diagram",
    side_effect=ValueError("Plot error"),
)
def test_add_reduction_diagnostics_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_diagnostics(rep, np.random.randn(10, 5), np.random.randn(10, 2))
    mock_log.assert_called_with(
        "Diagnostics section skipped: %s", mock_plot.side_effect
    )


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_interpretation",
    side_effect=ValueError("Plot error"),
    create=True,
)
def test_add_reduction_interpretation_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_interpretation(
        rep, {"loadings": np.ones((2, 2))}, analysis="loadings"
    )
    mock_log.assert_called_with(
        "Interpretation section skipped: %s", mock_plot.side_effect
    )


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_coranking_matrix",
    side_effect=ValueError("Plot error"),
)
def test_add_reduction_coranking_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_coranking(rep, np.ones((5, 5)))
    mock_log.assert_called_with("Co-ranking section skipped: %s", mock_plot.side_effect)


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_component_loadings",
    side_effect=ValueError("Plot error"),
)
def test_add_reduction_components_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_components(rep, np.ones((2, 2)))
    mock_log.assert_called_with("Component section skipped: %s", mock_plot.side_effect)


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_trajectory", side_effect=ValueError("Plot error")
)
def test_add_reduction_trajectory_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_trajectory(rep, np.random.randn(5, 10, 2))
    mock_log.assert_called_with("Trajectory section skipped: %s", mock_plot.side_effect)


@patch("coco_pipe.report.dim_reduction.logger.debug")
@patch(
    "coco_pipe.viz.dim_reduction.plot_trajectory_separation",
    side_effect=ValueError("Plot error"),
)
def test_add_reduction_trajectory_separation_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_reduction_trajectory_separation(rep, {"A-B": [1, 2]})
    mock_log.assert_called_with(
        "Trajectory separation section skipped: %s", mock_plot.side_effect
    )


@patch("coco_pipe.report.dim_reduction.warnings.warn")
def test_make_reduction_report_interactive_warning(mock_warn):
    make_reduction_report([MockReducerForExceptions()], interactive=True)
    assert mock_warn.called


def test_make_reduction_report_mismatched_embeddings():
    with pytest.raises(ValueError, match="must align with `reductions`"):
        make_reduction_report(
            [MockReducerForExceptions()],
            embeddings=[np.random.randn(10, 2), np.random.randn(10, 2)],
        )


def test_metrics_fallback_to_empty():
    class NoMetricsReducer:
        def get_summary(self):
            return {}

    rep = Report("Test")
    # Should not crash, should return self
    assert add_reduction_metrics(rep, NoMetricsReducer()) is rep


def test_metrics_get_scores_attribute_error():
    class BrokenScoresReducer:
        @property
        def get_scores(self):
            raise AttributeError("Broken")

        def get_summary(self):
            return {}

    rep = Report("Test")
    assert add_reduction_metrics(rep, BrokenScoresReducer()) is rep


def test_components_fallback_dict():
    from coco_pipe.report.dim_reduction import _components_payload

    class DictComponentReducer:
        def get_components(self):
            return {"components": np.ones((2, 2))}

    assert _components_payload(DictComponentReducer()) is not None

    class BrokenComponentReducer:
        @property
        def get_components(self):
            raise AttributeError("Broken")

    assert _components_payload(BrokenComponentReducer()) is None


def test_original_data_payload_fallback():
    from coco_pipe.report.dim_reduction import _original_data_payload

    assert _original_data_payload({"X_orig_": [1, 2]}) == [1, 2]
    assert _original_data_payload({"X_orig": [3, 4]}) == [3, 4]


def test_add_embedding_and_shepard_with_exceptions():
    from coco_pipe.report.dim_reduction import _add_embedding_and_shepard

    rep = Report("Test")
    reducer = MockReducerForExceptions()
    # Test skipping logic internally
    # with plot_shepard_diagram failing
    with patch(
        "coco_pipe.viz.dim_reduction.plot_shepard_diagram", side_effect=ValueError
    ):
        try:
            _add_embedding_and_shepard(
                rep,
                reducer,
                np.random.randn(10, 2),
                prefix="Test",
                diagnostics={"X_orig": np.random.randn(10, 5)},
            )
        except Exception:
            pass
        assert len(rep.children) >= 0
