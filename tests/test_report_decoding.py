import logging
from typing import ClassVar

import matplotlib
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from coco_pipe.io.quality import QCResult
from coco_pipe.report.core import Report
from coco_pipe.report.decoding import (
    DECODING_PRESETS,
    SectionDataUnavailable,
    _accepted_kwargs,
    _null_note,
    _result_frame,
    add_decoding_features,
    add_decoding_neural_artifacts,
    add_decoding_performance,
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
    make_decoding_result_report,
    render_unit_reports,
)
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)


def test_decoding_report_builder_renders_html(tmp_path):
    result = make_synthetic_result()
    report = make_decoding_result_report(
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
    report = make_decoding_result_report(
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

    html = make_decoding_result_report(
        result,
        sections=["overview"],
        qc_result=qc_result,
    ).render()

    assert "Data Quality (QC)" in html
    assert "QC Funnel Summary" in html


def test_render_unit_reports_from_successful_records(tmp_path):
    output_dir = tmp_path / "flat"
    make_synthetic_result(n_models=1, n_times=1).save(output_dir / "result.joblib")

    render_unit_reports(
        [
            {
                "status": "success",
                "analysis_mode": "flat",
                "target": "adhd",
                "unit_name": "all",
                "selection_mode": "baseline",
                "output_dir": str(output_dir),
            },
            {
                "status": "success",
                "analysis_mode": "sensor",
                "output_dir": str(tmp_path / "sensor"),
            },
        ],
        modes=["flat"],
        asset_urls={
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
        title_fn=lambda record: f"{record['target']} {record['unit_name']}",
    )

    assert (output_dir / "report.html").exists()
    assert not (tmp_path / "sensor" / "report.html").exists()


def test_report_methods_append_their_sections_in_order():
    result = make_synthetic_result()
    report = Report("Diagnostics")
    chained = (
        report.add_decoding_overview(result)
        .add_decoding_summary(result)
        .add_decoding_statistical_assessment(result)
        .add_decoding_neural_artifacts(result)
        .add_decoding_performance(result)
        .add_decoding_features(result)
        .add_decoding_temporal(result)
    )
    assert chained is report
    assert [section.title for section in report.children] == [
        "Overview",
        "Decoding Summary",
        "Statistical Assessment",
        "Neural Artifacts",
        "Performance",
        "Features",
        "Temporal Decoding",
    ]
    # a filter that matches nothing adds no section
    assert not Report("Empty").add_decoding_temporal(result, metric="ghost").children


def test_decoding_report_feature_metadata_is_explicit():
    result = make_synthetic_result()
    result.meta["feature_metadata"] = make_synthetic_feature_metadata()

    without_explicit = make_decoding_result_report(
        result, sections=["features"]
    ).render()
    assert "Feature Metadata" not in without_explicit

    with_explicit = make_decoding_result_report(
        result,
        feature_metadata=make_synthetic_feature_metadata(),
        sections=["features"],
    ).render()
    assert "Feature Metadata" in with_explicit


def test_decoding_report_rejects_unknown_sections():
    result = make_synthetic_result()
    with pytest.raises(ValueError, match="Unknown decoding report section"):
        make_decoding_result_report(result, sections=["unknown_section_name"])


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

    # Should not raise any errors, just return self
    report.add_decoding_temporal(result)
    report.add_decoding_summary(result)
    report.add_decoding_statistical_assessment(result)
    report.add_decoding_neural_artifacts(result)
    assert len(report.children) == 0


def _walk_elements(element):
    """Yield an element and all of its descendants."""
    yield element
    for child in getattr(element, "children", []):
        yield from _walk_elements(child)


def test_make_decoding_report_interactive_uses_plotly():
    from coco_pipe.report.elements import ImageElement, PlotlyElement

    result = make_synthetic_result()
    sections = ["cv", "performance", "statistical"]

    interactive = make_decoding_result_report(
        result, sections=sections, interactive=True
    )
    elements = list(_walk_elements(interactive))
    assert any(isinstance(e, PlotlyElement) for e in elements)
    assert "lazy-plot" in interactive.render()

    static = make_decoding_result_report(result, sections=sections, interactive=False)
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


class MalformedResult:
    """Frames that hold data but lack the columns the plots need."""

    def get_temporal_score_summary(self):
        return pd.DataFrame({"Metric": ["a"], "Score": [1], "Time": [0.1]})

    def get_detailed_scores(self):
        return pd.DataFrame({"Metric": ["a"], "Score": [1]})

    def get_statistical_assessment(self):
        return pd.DataFrame({"Metric": ["a"]})

    def get_model_artifacts(self):
        return pd.DataFrame({"Model": ["A"]})


@pytest.mark.parametrize(
    "add, title",
    [
        (add_decoding_temporal, "Temporal Decoding"),
        (add_decoding_statistical_assessment, "Statistical Assessment"),
        (add_decoding_neural_artifacts, "Neural Artifacts"),
    ],
)
def test_plot_that_cannot_be_drawn_leaves_the_data(add, title, caplog):
    from io import StringIO

    from coco_pipe.report.elements import (
        DownloadAssetElement,
        ImageElement,
        PlotlyElement,
    )

    report = Report("Test")
    with caplog.at_level(logging.DEBUG, logger="coco_pipe.report._utils"):
        add(report, MalformedResult())

    (section,) = report.children
    assert section.title == title
    elements = list(_walk_elements(section))
    assert not any(isinstance(e, ImageElement | PlotlyElement) for e in elements)
    (download,) = [e for e in elements if isinstance(e, DownloadAssetElement)]
    assert download.filename.endswith(".csv")
    assert len(pd.read_csv(StringIO(download.data))) == 1
    assert "missing required columns" in caplog.text


@pytest.mark.parametrize(
    "add, kwargs",
    [
        # no plot can be drawn, and the section holds nothing else
        (add_decoding_performance, {}),
        # no feature importances
        (add_decoding_features, {"feature_metadata": pd.DataFrame({"Feature": ["A"]})}),
        # no sensor positions
        (
            add_decoding_topomaps,
            {"feature_metadata": pd.DataFrame({"FeatureFamily": ["A"]})},
        ),
    ],
)
def test_section_without_content_is_left_out(add, kwargs):
    report = Report("Test")
    assert add(report, MalformedResult(), **kwargs) is report
    assert not report.children


def test_decoding_sections_force_headless_matplotlib_backend():
    matplotlib.use("svg", force=True)
    try:
        build_decoding_sections(make_synthetic_result(), sections=[])
        assert matplotlib.get_backend().lower() == "agg"
    finally:
        matplotlib.use("Agg", force=True)


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
    class Broken:
        def get_detailed_scores(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        build_decoding_sections(Broken(), sections=["cv"], on_error="raise")
    with pytest.warns(RuntimeWarning, match="section 'cv' failed: boom"):
        assert not build_decoding_sections(Broken(), sections=["cv"], on_error="warn")
    (placeholder,) = build_decoding_sections(
        Broken(), sections=["cv"], on_error="placeholder"
    )
    assert placeholder.status == "WARN"
    assert "boom" in placeholder.render()


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
        raw: ClassVar[dict] = {}

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
        config: ClassVar[dict] = {}

    with pytest.raises(SectionDataUnavailable, match="configuration"):
        build_configuration_section(NoConfig())

    class NoMeta:
        meta: ClassVar[dict] = {}

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
    # the metadata names no feature of the result, so no family can be drawn
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
    make_decoding_result_report(
        make_synthetic_result(), sections=["overview"], output_path=str(path)
    )
    assert path.exists() and "<html" in path.read_text()


def test_statistical_section_states_what_the_null_permuted():
    permuted = pd.DataFrame(
        {
            "Model": ["A"],
            "Metric": ["accuracy"],
            "Observed": [0.7],
            "PValue": [0.01],
            "NullMethod": ["permutation_full_pipeline"],
        }
    )
    within = {"permutation_scheme": "within_strata", "strata_column": "Subject"}
    blocks = {
        "permutation_scheme": "blocks_within_strata",
        "strata_column": "Subject",
        "block_column": "Run",
    }
    notes = {
        "units": _null_note(
            {"statistical_assessment": {"resolved_unit_of_inference": "group"}},
            permuted,
        ),
        "within": _null_note({"statistical_assessment": within}, permuted),
        "blocks": _null_note({"statistical_assessment": blocks}, permuted),
        "sign_flip": _null_note({}, permuted.assign(NullMethod="sign_flip_group")),
    }
    assert "between whole 'group' units" in notes["units"]
    assert "shuffled within each 'Subject' stratum" in notes["within"]
    assert (
        "'Run' blocks shuffled between the blocks of each 'Subject'"
        in (notes["blocks"])
    )
    assert "flipped at random across groups" in notes["sign_flip"]
    assert _null_note({}, permuted.assign(NullMethod="binomial")) == ""

    class Result:
        meta: ClassVar[dict] = {"statistical_assessment": blocks}

        def get_statistical_assessment(self):
            return permuted

    html = build_statistical_section(Result()).render()
    assert "shuffled between the blocks of each" in html
