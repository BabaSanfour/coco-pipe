from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from coco_pipe.io.quality import QCResult
from coco_pipe.report.api import from_experiment_result
from coco_pipe.report.core import Report
from coco_pipe.report.decoding import (
    DECODING_PRESETS,
    SectionDataUnavailable,
    _accepted_kwargs,
    _result_frame,
    add_decoding_diagnostics,
    add_decoding_features,
    add_decoding_neural_artifacts,
    add_decoding_statistical_assessment,
    add_decoding_temporal,
    add_decoding_topomaps,
    build_caveats_section,
    build_configuration_section,
    build_cv_section,
    build_decoding_sections,
    build_export_inventory_section,
    build_features_section,
    build_fit_diagnostics_section,
    build_neural_section,
    build_probability_section,
    build_provenance_section,
    build_statistical_section,
    build_temporal_section,
    build_topomaps_section,
    build_tuning_section,
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


@pytest.mark.parametrize("interactive", [False, True])
def test_full_decoding_report_covers_all_builders(interactive):
    result = make_synthetic_result()
    metadata = pd.DataFrame(make_synthetic_feature_metadata())
    coords = metadata.drop_duplicates("Sensor").set_index("Sensor")[["x", "y"]]
    report = make_decoding_report(
        result,
        sections="full",
        feature_metadata=metadata,
        coords=coords,
        interactive=interactive,
        verbose=True,
    )
    titles = [section.title for section in report.children]
    for expected in (
        "Sensor Maps",
        "Fit Diagnostics",
        "Hyperparameter Tuning",
        "Neural Artifacts",
        "Configuration",
        "Provenance",
        "Export Inventory",
    ):
        assert expected in titles, (expected, interactive)
    # rendering must succeed end-to-end for both modalities
    assert "<html" in report.render()


def test_decoding_report_renders_qc_result():
    result = make_synthetic_result()
    qc_result = QCResult(
        n_obs_in=10,
        n_obs_out=8,
        n_subjects_in=5,
        n_subjects_out=4,
    )

    html = make_decoding_report(
        result,
        sections=["overview"],
        qc_result=qc_result,
    ).render()

    assert "Data Quality (QC)" in html
    assert "QC Funnel Summary" in html


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


# Plot failures are swallowed and logged by the shared `plot_or_none` helper, so
# these tests patch the logger in `coco_pipe.report._utils`, not in `decoding`.
@patch("coco_pipe.report._utils.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_temporal_score_curve",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_temporal_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_temporal(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch("coco_pipe.report._utils.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_decoding_scores", side_effect=ValueError("Plot error")
)
def test_add_decoding_performance_exception(mock_plot, mock_log):
    rep = Report("Test")
    from coco_pipe.report.decoding import add_decoding_performance

    add_decoding_performance(rep, MockResultForExceptions())
    mock_log.assert_called()


# `add_decoding_diagnostics` now composes the CV + probability blocks; patch a plot
# that the CV builder actually calls (fit diagnostics moved to its own section).
@patch("coco_pipe.report._utils.logger.debug")
@patch(
    "coco_pipe.viz.decoding.plot_fold_score_dispersion",
    side_effect=ValueError("Plot error"),
)
def test_add_decoding_diagnostics_exception(mock_plot, mock_log):
    rep = Report("Test")
    add_decoding_diagnostics(rep, MockResultForExceptions())
    mock_log.assert_called()


@patch("coco_pipe.report._utils.logger.debug")
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
    mock_plot.assert_not_called()


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
    mock_plot.assert_not_called()


def _walk_elements(element):
    """Yield an element and all of its descendants."""
    yield element
    for child in getattr(element, "children", []):
        yield from _walk_elements(child)


def test_make_decoding_report_interactive_uses_plotly():
    from coco_pipe.report.elements import ImageElement, PlotlyElement

    result = make_synthetic_result()
    sections = ["cv", "performance", "statistical"]

    interactive = make_decoding_report(result, sections=sections, interactive=True)
    elements = list(_walk_elements(interactive))
    assert any(isinstance(e, PlotlyElement) for e in elements)
    assert "lazy-plot" in interactive.render()

    static = make_decoding_report(result, sections=sections, interactive=False)
    static_elements = list(_walk_elements(static))
    assert any(isinstance(e, ImageElement) for e in static_elements)
    assert not any(isinstance(e, PlotlyElement) for e in static_elements)


def test_decoding_presets_do_not_duplicate_diagnostics():
    assert len(DECODING_PRESETS["default"]) == len(set(DECODING_PRESETS["default"]))
    assert "cv_summary" not in DECODING_PRESETS["default"]
    assert "confusion_probability" not in DECODING_PRESETS["default"]
    assert DECODING_PRESETS["default"].count("cv") == 1
    assert DECODING_PRESETS["default"].count("probability") == 1
    assert DECODING_PRESETS["default"].count("fit_diagnostics") == 1


def test_decoding_sections_force_headless_matplotlib_backend():
    result = make_synthetic_result()

    with (
        patch("matplotlib.get_backend", return_value="MacOSX"),
        patch("matplotlib.use") as use_backend,
    ):
        build_decoding_sections(result, sections=[])

    use_backend.assert_called_once_with("Agg", force=True)


def test_builders_split_cv_from_probability():
    result = make_synthetic_result()

    cv = build_cv_section(result)
    probability = build_probability_section(result)

    assert cv.title == "Cross-Validation"
    assert probability.title == "Confusion and Probability"
    assert "Confusion matrix" not in cv.render()
    assert "Fold score dispersion" not in probability.render()


def test_empty_builder_raises_data_unavailable():
    class Empty:
        def get_detailed_scores(self):
            return pd.DataFrame()

    with pytest.raises(SectionDataUnavailable):
        build_cv_section(Empty())


def test_decoding_report_error_policy():
    result = make_synthetic_result()

    with patch.dict(
        "coco_pipe.report.decoding.DECODING_SECTION_BUILDERS",
        {"overview": lambda result: (_ for _ in ()).throw(RuntimeError("boom"))},
    ):
        with pytest.raises(RuntimeError, match="boom"):
            build_decoding_sections(result, sections=["overview"], on_error="raise")
        with pytest.warns(RuntimeWarning, match="boom"):
            assert not build_decoding_sections(
                result, sections=["overview"], on_error="warn"
            )
        placeholder = build_decoding_sections(
            result, sections=["overview"], on_error="placeholder"
        )
        assert len(placeholder) == 1
        assert placeholder[0].status == "WARN"


def test_accepted_kwargs_and_result_frame_branches():
    def variadic(**kwargs):
        pass

    assert _accepted_kwargs(variadic, {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    # normal path: unknown keys are silently dropped
    def fixed(x, y):
        pass

    assert _accepted_kwargs(fixed, {"x": 10, "z": 99}) == {"x": 10}

    class NotCallable:
        summary = "a_string"

    with pytest.raises(SectionDataUnavailable):
        _result_frame(NotCallable(), "summary", required=True)
    assert _result_frame(NotCallable(), "summary", required=False).empty

    class Broken:
        def summary(self):
            raise ValueError("oops")

    assert _result_frame(Broken(), "summary", required=False).empty
    with pytest.raises(ValueError, match="oops"):
        _result_frame(Broken(), "summary", required=True)


def test_section_builders_empty_and_filter_raises():
    """Smoke-test every SectionDataUnavailable guard that fires when data are
    absent or a metric/model filter empties the frame."""
    result = make_synthetic_result()

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_cv_section(result, metric="nonexistent")

    class NoProbResult:
        raw = {}

        def get_confusion_matrices(self, model=None):
            return pd.DataFrame()

    with pytest.raises(SectionDataUnavailable):
        build_probability_section(NoProbResult())

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_statistical_section(result, metric="nonexistent")

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_temporal_section(result, metric="nonexistent")

    class NoTimeResult:
        def get_temporal_score_summary(self, model=None):
            return pd.DataFrame({"Metric": ["accuracy"], "Score": [0.9]})

    with pytest.raises(SectionDataUnavailable, match="not temporally resolved"):
        build_temporal_section(NoTimeResult())

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_fit_diagnostics_section(result, model="ghost_model")

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_tuning_section(result, model="ghost_model")

    with pytest.raises(SectionDataUnavailable, match="No matching"):
        build_neural_section(result, model="ghost_model")

    class NoConfig:
        config = {}

    with pytest.raises(SectionDataUnavailable, match="configuration"):
        build_configuration_section(NoConfig())

    class NoMeta:
        meta = {}

    with pytest.raises(SectionDataUnavailable, match="Provenance"):
        build_provenance_section(NoMeta())

    with pytest.raises(ValueError, match="Unknown decoding report preset"):
        build_decoding_sections(result, sections="not_a_preset")

    with pytest.raises(ValueError, match="on_error must be"):
        build_decoding_sections(result, sections=[], on_error="silently_ignore")


def test_section_builders_single_item_and_extra_branches():
    result = make_synthetic_result(n_models=1)

    prob = build_probability_section(result, model="model_1")
    assert prob is not None and prob.children

    feat = build_features_section(result, model="model_1")
    assert feat is not None and feat.children

    class NoModelCol:
        def get_feature_importances(self, model=None):
            return pd.DataFrame({"Feature": ["F1"], "Score": [0.5]})

        def get_feature_stability(self, model=None):
            return pd.DataFrame()

    feat_no_model = build_features_section(NoModelCol())
    assert feat_no_model is not None

    stat = build_statistical_section(
        make_synthetic_result(), metric="temporal_accuracy"
    )
    assert stat is not None

    temp = build_temporal_section(
        make_synthetic_result(), metric="generalization_accuracy"
    )
    assert temp is not None

    meta = pd.DataFrame(
        {"FeatureFamily": ["EEG"], "Sensor": ["Fp1"], "x": [0.0], "y": [0.0]}
    )
    with patch("coco_pipe.report._utils._plot_or_none", return_value=None):
        with pytest.raises(SectionDataUnavailable, match="No sensor maps"):
            build_topomaps_section(
                make_synthetic_result(),
                feature_metadata=meta,
                coords=meta.set_index("Sensor")[["x", "y"]],
            )


def test_caveats_and_export_inventory_branches():
    class Plain:
        pass

    sec = build_caveats_section(Plain(), feature_metadata=None)
    assert "sensor-wise feature plots" in sec.render()

    class WithEmptyProb:
        def get_probability_diagnostics(self):
            return pd.DataFrame()

    sec2 = build_caveats_section(WithEmptyProb(), feature_metadata=None)
    assert "Probability diagnostics" in sec2.render()

    with pytest.raises(SectionDataUnavailable, match="No report caveats"):
        build_caveats_section(Plain(), feature_metadata=pd.DataFrame({"x": [1]}))

    class BrokenAccessor:
        def summary(self):
            raise ValueError("broken")

        def get_detailed_scores(self):
            return pd.DataFrame({"Score": [1.0]})

    inv = build_export_inventory_section(BrokenAccessor())
    assert inv is not None and "summary" in inv.render()

    class Empty:
        pass

    with pytest.raises(SectionDataUnavailable, match="No result exports"):
        build_export_inventory_section(Empty())


def test_make_decoding_report_output_path(tmp_path):
    """L1095: output_path triggers report.save."""
    path = tmp_path / "out.html"
    make_decoding_report(
        make_synthetic_result(), sections=["overview"], output_path=str(path)
    )
    assert path.exists() and "<html" in path.read_text()
