from pathlib import Path

import pandas as pd

from coco_pipe.report.api import from_experiment_results
from coco_pipe.report.decoding_comparison import (
    build_comparison_section,
    build_result_tabs,
    collect_results,
)
from coco_pipe.report.elements import TabsElement
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
