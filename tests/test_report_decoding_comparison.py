from pathlib import Path

import pandas as pd
import pytest

from coco_pipe.io.quality import QCResult
from coco_pipe.report.api import from_experiment_results
from coco_pipe.report.decoding_comparison import (
    ResultCollection,
    _load_result,
    build_comparison_section,
    build_result_tabs,
    collect_results,
    make_experiment_results_report,
)
from coco_pipe.report.elements import AccordionElement, TabsElement
from tests.fixtures.synthetic_result import make_synthetic_result


def _items():
    return [
        (
            {
                "scope": scope,
                "analysis_mode": "sensor",
                "unit_name": sensor,
            },
            make_synthetic_result(n_models=2, n_times=1),
        )
        for scope, sensor in (
            ("EO", "Fp1"),
            ("EO", "Fp2"),
            ("EO", "Cz"),
        )
    ]


def test_collect_results_tags_summary_and_records_load_failures(tmp_path):
    items = _items()
    items.append(
        (
            {
                "scope": "EO",
                "analysis_mode": "sensor",
                "unit_name": "bad",
            },
            tmp_path / "missing.joblib",
        )
    )

    collection = collect_results(
        items,
        by=("scope", "analysis_mode", "unit_name"),
    )

    assert len(collection.results) == 3
    assert {"scope", "analysis_mode", "unit_name", "Model"}.issubset(
        collection.summary.columns
    )
    assert set(collection.summary["_status"]) == {"success", "failed"}
    assert len(collection.filter(unit_name="Fp1").results) == 1


def test_comparison_builders_cover_heatmap_and_sensor_topomap():
    collection = collect_results(
        _items(),
        by=("scope", "analysis_mode", "unit_name"),
    )
    heatmap = build_comparison_section(
        collection,
        kind="axis_heatmap",
        axis="unit_name",
        column="Model",
        group_by=("scope",),
        title="Sensor Accuracy",
    )
    coords = pd.DataFrame(
        {
            "ch_name": ["Fp1", "Fp2", "Cz"],
            "x": [-0.3, 0.3, 0.0],
            "y": [0.4, 0.4, 0.0],
        }
    )
    topomap = build_comparison_section(
        collection,
        kind="sensor_topomap",
        axis="unit_name",
        group_by=("scope", "Model"),
        coords=coords,
        title="Sensor Topomap",
    )

    assert heatmap is not None
    assert topomap is not None
    assert heatmap.children
    assert topomap.children


def test_build_result_tabs_and_public_multi_result_api(tmp_path):
    collection = collect_results(
        _items()[:2],
        by=("scope", "analysis_mode", "unit_name"),
    )

    nested = build_result_tabs(collection, sections=["overview", "model_summary"])
    report = from_experiment_results(
        _items()[:2],
        by=("scope", "analysis_mode", "unit_name"),
        comparisons=[
            {
                "kind": "axis_heatmap",
                "axis": "unit_name",
                "column": "Model",
                "title": "Comparison",
            }
        ],
        per_result=["overview"],
        asset_urls={
            "plotly": "about:blank",
            "tailwind": "about:blank",
            "pako": "about:blank",
        },
        output_path=tmp_path / "comparison.html",
    )

    assert isinstance(nested, TabsElement)
    assert [section.title for section in report.children] == [
        "Comparison",
        "Per-Result Diagnostics",
    ]
    assert Path(tmp_path / "comparison.html").exists()


def test_collect_results_metrics_filter_and_duplicate_source():
    import pytest

    shared = make_synthetic_result(n_models=2, n_times=1)
    # Same key + same source is deduplicated silently.
    collection = collect_results(
        [({"scope": "EO"}, shared), ({"scope": "EO"}, shared)],
        by=("scope",),
        metrics=["accuracy"],
    )
    assert len(collection.results) == 1
    kept = [c for c in collection.summary.columns if c.startswith("accuracy_")]
    assert kept  # metrics filter retained accuracy_* columns
    assert not any(c.startswith("roc_auc_") for c in collection.summary.columns)

    # Same key but different sources is a hard error.
    with pytest.raises(ValueError, match="multiple sources"):
        collect_results(
            [
                ({"scope": "EO"}, make_synthetic_result(n_models=1, n_times=1)),
                ({"scope": "EO"}, make_synthetic_result(n_models=1, n_times=1)),
            ],
            by=("scope",),
        )


def test_collect_results_missing_axis_field_raises():
    import pytest

    with pytest.raises(ValueError, match="missing axis fields"):
        collect_results(
            [({"scope": "EO"}, make_synthetic_result(n_models=1, n_times=1))],
            by=("scope", "unit_name"),
        )


def test_comparison_builders_cover_remaining_kinds():
    collection = collect_results(
        _items(),
        by=("scope", "analysis_mode", "unit_name"),
    )
    for kind in ("model_bars", "head_to_head", "spread", "metric_matrix"):
        section = build_comparison_section(
            collection,
            kind=kind,
            axis="unit_name",
            title=kind,
        )
        assert section is not None, kind
        assert section.children, kind

    grid = build_comparison_section(
        collection,
        kind="grid_heatmap",
        row="unit_name",
        column="Model",
        title="Grid",
    )
    assert grid is not None and grid.children


def test_comparison_unknown_kind_warns_and_returns_none():
    collection = collect_results(
        _items(),
        by=("scope", "analysis_mode", "unit_name"),
    )
    section = build_comparison_section(
        collection, kind="not_a_real_kind", axis="unit_name"
    )
    assert section is None


def test_paired_delta_success_path():
    collection = collect_results(
        [
            ({"comparison": name}, make_synthetic_result(n_models=1, n_times=1))
            for name in ("A", "B")
        ],
        by=("comparison",),
    )
    collection.summary = pd.DataFrame(
        {
            "comparison": ["A", "B"],
            "delta": [0.02, -0.01],
            "ci_lower": [0.0, -0.03],
            "ci_upper": [0.04, 0.01],
            "_status": ["success", "success"],
        }
    )
    section = build_comparison_section(
        collection, kind="paired_delta", axis="comparison", value="delta"
    )
    assert section is not None and section.children


def test_showcase_preset_renders_multiple_comparison_views():
    from coco_pipe.report import make_experiment_results_report

    report = make_experiment_results_report(
        _items(),
        by=("scope", "analysis_mode", "unit_name"),
        comparisons="showcase",
        per_result="compact",
    )
    titles = [section.title for section in report.children]
    assert any("Model Comparison" in title for title in titles)
    # heatmap / spread / metric matrix keyed on the first by-axis ("scope")
    assert (
        sum(
            any(token in title for token in ("×", "Spread", "Matrix"))
            for title in titles
        )
        >= 2
    )
    assert "Per-Result Diagnostics" in titles


def test_paired_delta_rejects_mismatched_cv_designs():
    frame = pd.DataFrame(
        {
            "comparison": ["A", "B"],
            "delta": [0.02, -0.01],
            "cv_signature": ["groupkfold-5", "groupkfold-10"],
            "_status": ["success", "success"],
        }
    )
    collection = collect_results(
        [
            (
                {"comparison": name},
                make_synthetic_result(n_models=1, n_times=1),
            )
            for name in ("A", "B")
        ],
        by=("comparison",),
    )
    collection.summary = frame

    section = build_comparison_section(
        collection,
        kind="paired_delta",
        axis="comparison",
        value="delta",
    )

    assert section is not None
    assert "Incompatible comparison design" in section.render()


def test_collect_results_and_result_collection_edge_cases(tmp_path):
    summary = pd.DataFrame({"scope": ["EO"], "accuracy_mean": [0.9]})
    coll = ResultCollection(by=("scope",), results={}, contexts={}, summary=summary)
    assert "_status" not in coll.successful_summary.columns
    with pytest.raises(Exception):
        _load_result(tmp_path)  # directory exists, result.joblib inside does not

    with pytest.raises(ValueError, match="by must contain"):
        collect_results([], by=())

    result = make_synthetic_result(n_models=2, n_times=1)
    coll2 = collect_results([({**{"scope": "EO"}}, result)], by=("scope",))
    assert "Model" in coll2.summary.columns or len(coll2.results) == 1

    class EmptyResult:
        raw = {}

        def summary(self):
            return pd.DataFrame()

    coll3 = collect_results([({"scope": "EO"}, EmptyResult())], by=("scope",))
    assert "empty" in coll3.summary["_status"].values


def _two_item_collection():
    return collect_results(
        [
            (
                {"scope": f"run{i}", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            )
            for i in range(2)
        ],
        by=("scope", "model"),
    )


def test_build_comparison_section_guard_branches():
    coll = _two_item_collection()

    with pytest.raises(ValueError, match="on_error must be"):
        build_comparison_section(coll, kind="model_bars", on_error="placeholder")

    failed = coll.summary.copy()
    failed["_status"] = "failed"
    coll_failed = ResultCollection(coll.by, coll.results, coll.contexts, failed)
    assert build_comparison_section(coll_failed, kind="model_bars") is None

    nan_coll = ResultCollection(
        coll.by, coll.results, coll.contexts, coll.summary.copy()
    )
    nan_coll.summary["accuracy_mean"] = "not_a_number"
    assert (
        build_comparison_section(
            nan_coll, kind="model_bars", axis="scope", value="accuracy_mean"
        )
        is None
    )

    assert (
        build_comparison_section(
            coll, kind="axis_heatmap", row="ghost_col", column="Model"
        )
        is None
    )

    nan_pivot = ResultCollection(
        coll.by, coll.results, coll.contexts, coll.summary.copy()
    )
    nan_pivot.summary["accuracy_mean"] = float("nan")
    assert (
        build_comparison_section(
            nan_pivot,
            kind="axis_heatmap",
            row="scope",
            column="Model",
            value="accuracy_mean",
        )
        is None
    )

    assert (
        build_comparison_section(
            coll, kind="sensor_topomap", axis="scope", info=None, coords=None
        )
        is None
    )

    assert build_comparison_section(coll, kind="model_bars", axis="ghost_col") is None

    assert build_comparison_section(coll, kind="spread", axis="ghost_col") is None

    no_mean = ResultCollection(
        coll.by, coll.results, coll.contexts, coll.summary.copy()
    )
    no_mean.summary = no_mean.summary.drop(
        columns=[c for c in no_mean.summary.columns if c.endswith("_mean")],
        errors="ignore",
    )
    assert build_comparison_section(no_mean, kind="metric_matrix", axis="scope") is None


def test_build_comparison_section_on_error_raise_branches():
    coll = _two_item_collection()

    with pytest.raises(ValueError, match="Unknown decoding comparison kind"):
        build_comparison_section(coll, kind="totally_unknown", on_error="raise")

    frame = pd.DataFrame(
        {
            "comparison": ["A", "B"],
            "delta": [0.02, -0.01],
            "cv_signature": ["groupkfold-5", "groupkfold-10"],
            "_status": ["success", "success"],
        }
    )
    mismatch_coll = collect_results(
        [
            ({"comparison": name}, make_synthetic_result(n_models=1, n_times=1))
            for name in ("A", "B")
        ],
        by=("comparison",),
    )
    mismatch_coll.summary = frame
    with pytest.raises(ValueError, match="Paired delta"):
        build_comparison_section(
            mismatch_coll,
            kind="paired_delta",
            axis="comparison",
            value="delta",
            on_error="raise",
        )


def test_build_result_tabs_return_types_and_report_branches(tmp_path):
    single = collect_results(
        [
            (
                {"scope": "run0", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            )
        ],
        by=("scope", "model"),
    )
    multi = _two_item_collection()

    assert isinstance(
        build_result_tabs(single, sections=["overview"]), AccordionElement
    )

    assert isinstance(build_result_tabs(multi, sections=["overview"]), TabsElement)

    report_no_nest = make_experiment_results_report(
        [
            (
                {"scope": f"run{i}", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            )
            for i in range(2)
        ],
        by=("scope", "model"),
        comparisons=[],
        per_result="compact",
        nest=False,
    )
    assert "Per-Result Diagnostics" not in [s.title for s in report_no_nest.children]

    report_no_per = make_experiment_results_report(
        [
            (
                {"scope": f"run{i}", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            )
            for i in range(2)
        ],
        by=("scope", "model"),
        comparisons=[],
        per_result=None,
        nest=True,
    )
    assert "Per-Result Diagnostics" not in [s.title for s in report_no_per.children]

    qc = QCResult(n_obs_in=10, n_obs_out=8, n_subjects_in=5, n_subjects_out=4)
    report_qc = make_experiment_results_report(
        [
            (
                {"scope": f"run{i}", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            )
            for i in range(2)
        ],
        by=("scope", "model"),
        comparisons=[],
        per_result=None,
        nest=False,
        qc_result=qc,
    )
    assert any(
        "QC" in t or "Quality" in t or "Data" in t
        for t in [s.title for s in report_qc.children]
    )

    report_fail = make_experiment_results_report(
        [
            (
                {"scope": "run0", "model": "SVM"},
                make_synthetic_result(n_models=1, n_times=1),
            ),
            ({"scope": "bad", "model": "SVM"}, tmp_path / "nonexistent.joblib"),
        ],
        by=("scope", "model"),
        comparisons=[],
        per_result=None,
        nest=False,
    )
    assert "Collection Status" in [s.title for s in report_fail.children]
