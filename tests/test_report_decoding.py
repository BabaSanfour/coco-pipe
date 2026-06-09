from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from coco_pipe.report.api import from_experiment_result
from coco_pipe.report.core import Report
from coco_pipe.report.decoding import (
    add_decoding_diagnostics,
    add_decoding_features,
    add_decoding_neural_artifacts,
    add_decoding_statistical_assessment,
    add_decoding_temporal,
    add_decoding_topomaps,
    make_decoding_report,
)
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)


def test_decoding_report_builder_renders_html(tmp_path):
    result = make_synthetic_result()
    report = make_decoding_report(
        result,
        sections=["overview", "performance", "provenance", "features", "temporal"],
    )
    html = report.render()
    parsed = BeautifulSoup(html, "html.parser")
    assert parsed.find("html") is not None
    assert "Overview" in html
    assert "Performance" in html

    path = tmp_path / "report.html"
    report.save(str(path))
    BeautifulSoup(path.read_text(), "html.parser")


def test_all_decoding_report_methods_run():
    result = make_synthetic_result()
    report = Report("Diagnostics")
    report.add_decoding_overview(result)
    report.add_decoding_summary(result)
    report.add_decoding_diagnostics(result)
    report.add_decoding_statistical_assessment(result)
    report.add_decoding_neural_artifacts(result)
    report.add_decoding_performance(result)
    report.add_decoding_features(result)
    report.add_decoding_temporal(result)
    html = report.render()
    assert "Diagnostics" in html
    assert "Temporal Decoding" in html


def test_decoding_report_feature_metadata_is_explicit():
    result = make_synthetic_result()
    result.meta["feature_metadata"] = make_synthetic_feature_metadata()

    without_explicit = make_decoding_report(result, sections=["features"]).render()
    assert "Feature Metadata" not in without_explicit

    with_explicit = make_decoding_report(
        result,
        feature_metadata=make_synthetic_feature_metadata(),
        sections=["features"],
    ).render()
    assert "Feature Metadata" in with_explicit


def test_decoding_report_api_and_core_reexports():
    result = make_synthetic_result()
    report = from_experiment_result(result, sections=["overview", "performance"])
    assert isinstance(report, Report)
    # add_decoding_* are now dynamically registered extensions
    assert hasattr(report, "add_decoding_temporal")
    assert hasattr(report, "add_decoding_summary")


def test_decoding_report_rejects_unknown_sections():
    result = make_synthetic_result()
    with pytest.raises(ValueError, match="Unknown decoding report section"):
        make_decoding_report(result, sections=["performnace"])


def test_decoding_report_empty_edge_cases():
    class EmptyResult:
        def get_temporal_score_summary(self):
            return pd.DataFrame()

        def summary(self):
            return pd.DataFrame()

        def get_detailed_scores(self):
            return pd.DataFrame()

        def get_fit_diagnostics(self):
            return pd.DataFrame()

        def get_confusion_matrices(self, model=None):
            return pd.DataFrame()

        def get_statistical_assessment(self):
            return pd.DataFrame()

        def get_model_artifacts(self):
            return pd.DataFrame()

    result = EmptyResult()
    report = Report("Empty")
    import pandas as pd

    # Should not raise any errors, just return self
    report.add_decoding_temporal(result)
    report.add_decoding_summary(result)
    report.add_decoding_diagnostics(result)
    report.add_decoding_statistical_assessment(result)
    report.add_decoding_neural_artifacts(result)
    assert len(report.children) == 0


def test_decoding_full_coverage():
    import pandas as pd

    class FullResult:
        def get_temporal_score_summary(self):
            return pd.DataFrame(
                {
                    "Metric": ["accuracy"],
                    "Model": ["A"],
                    "Time": [0.1],
                    "TrainTime": [0.1],
                    "TestTime": [0.1],
                    "Score": [1.0],
                }
            )

        def summary(self):
            return pd.DataFrame({"Metric": ["accuracy"], "Score": [1.0]})

        def get_detailed_scores(self):
            return pd.DataFrame(
                {"Metric": ["accuracy"], "Model": ["A"], "Fold": [0], "Score": [1.0]}
            )

        def get_fit_diagnostics(self):
            return pd.DataFrame(
                {
                    "Model": ["A"],
                    "Fold": [0],
                    "FitTime": [0.1],
                    "PredictTime": [0.1],
                    "TotalTime": [0.2],
                    "WarningMessage": ["test"],
                    "Stage": ["train"],
                    "WarningCategory": ["test"],
                }
            )

        def get_confusion_matrices(self, model=None):
            return pd.DataFrame(
                {"TrueLabel": ["A"], "PredictedLabel": ["A"], "Count": [10]}
            )

        def get_statistical_assessment(self):
            return pd.DataFrame(
                {
                    "Metric": ["accuracy"],
                    "Model": ["A"],
                    "Time": [0.1],
                    "Score": [1.0],
                    "NullScore": [0.5],
                    "PValue": [0.01],
                }
            )

        def get_model_artifacts(self):
            return pd.DataFrame({"Model": ["A"], "Epoch": [1], "Loss": [0.1]})

        # properties for ROC/PR
        classes_ = np.array([0, 1])

        def predict_proba(self, *args, **kwargs):
            return np.array([[0.1, 0.9]])

        def predict(self, *args, **kwargs):
            return np.array([1])

    result = FullResult()
    report = Report("Full")
    report.add_decoding_temporal(result, metric="accuracy", model="A")
    report.add_decoding_summary(result)
    report.add_decoding_diagnostics(result, metric="accuracy", model="A")
    report.add_decoding_statistical_assessment(result, metric="accuracy", model="A")
    report.add_decoding_neural_artifacts(result, model="A")

    # Try features
    feature_meta = pd.DataFrame({"Feature": ["F1"], "FeatureFamily": ["EEG"]})
    report.add_decoding_features(result, feature_metadata=feature_meta)

    # Topomaps
    report.add_decoding_topomaps(result, feature_metadata=feature_meta)

    assert len(report.children) > 0


class MockResultForExceptions:
    def get_temporal_score_summary(self):
        return pd.DataFrame({"Metric": ["a"], "Score": [1], "Time": [0.1]})

    def summary(self):
        return pd.DataFrame({"Metric": ["a"], "Score": [1]})

    def get_detailed_scores(self):
        return pd.DataFrame({"Metric": ["a"], "Score": [1]})

    def get_fit_diagnostics(self):
        return pd.DataFrame({"Model": ["A"]})

    def get_confusion_matrices(self, model=None):
        return pd.DataFrame({"TrueLabel": ["A"], "Count": [1]})

    def get_statistical_assessment(self):
        return pd.DataFrame({"Metric": ["a"]})

    def get_model_artifacts(self):
        return pd.DataFrame({"Model": ["A"]})


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_temporal_score_curve",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_temporal_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_temporal(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_decoding_scores", side_effect=ValueError("Plot error")
)
def test_add_decoding_performance_exception(mock_plot, mock_log):
    rep = Report("Test")
    from coco_pipe.report.decoding import add_decoding_performance

    add_decoding_performance(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_fit_diagnostics", side_effect=ValueError("Plot error")
)
def test_add_decoding_diagnostics_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_diagnostics(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_temporal_statistical_assessment",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_statistical_assessment_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_statistical_assessment(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch(
    "coco_pipe.viz.decoding.plot_training_history", side_effect=ValueError("Plot error")
)
def test_add_decoding_neural_artifacts_exception(mock_plot):
    rep = Report("Test")
    # This should just not raise an exception
    add_decoding_neural_artifacts(rep, MockResultForExceptions())


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_feature_importance",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_features_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_features(
        rep,
        MockResultForExceptions(),
        feature_metadata=pd.DataFrame({"Feature": ["A"]}),
    )
    mock_log.assert_called()


@patch("coco_pipe.report.decoding.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_feature_sensor_profile",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_topomaps_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_topomaps(
        rep,
        MockResultForExceptions(),
        feature_metadata=pd.DataFrame({"FeatureFamily": ["A"]}),
    )
    mock_log.assert_called()


@patch("coco_pipe.report.decoding.warnings.warn")
def test_make_decoding_report_interactive_warning(mock_warn):
    make_decoding_report(MockResultForExceptions(), interactive=True)
    assert mock_warn.called
