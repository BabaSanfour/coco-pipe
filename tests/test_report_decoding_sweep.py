import pandas as pd
import pytest

from coco_pipe.report.core import Report, Section
from coco_pipe.report.decoding import make_decoding_report
from coco_pipe.report.decoding_sweep import (
    add_scientific_overview,
    build_classical_taxonomy_sections,
    build_comparison_section,
    collect_mode_results,
    collect_results,
    enrich_head_to_head_frame,
    failures_section,
    feature_selection_section,
    flat_section,
    hp_tuning_section,
    leaderboard_section,
    make_decoding_sweep_report,
    make_head_to_head_report,
    paired_delta_vs_baseline,
    prepare_sweep_frame,
    summary_collection,
)
from coco_pipe.report.elements import InteractiveTableElement
from tests.fixtures.synthetic_result import make_synthetic_result


def test_summary_collection_adds_descriptor_family_and_sensor():
    frame = pd.DataFrame(
        [
            {
                "status": "success",
                "scope": "EO",
                "target": "adhd",
                "analysis_mode": "descriptor_sensor",
                "unit_name": "log_abs_alpha",
                "unit_key": "log_abs_alpha_Fz",
                "model": "rf",
                "primary_metric": 0.62,
            }
        ]
    )

    collection = summary_collection(
        frame,
        descriptor_family={"log_abs_alpha": "log_abs"},
        include_sensor=True,
    )

    assert collection is not None
    row = collection.summary.iloc[0]
    assert row["Model"] == "rf"
    assert row["descriptor_family"] == "log_abs"
    assert row["sensor"] == "Fz"


def test_paired_delta_vs_baseline_skips_incompatible_groups():
    frame = pd.DataFrame(
        [
            {
                "scope": "EO",
                "target": "adhd",
                "comparison_family": "descriptor_flat_baseline",
                "model_label": "logreg",
                "primary_metric": 0.6,
            },
            {
                "scope": "EO",
                "target": "adhd",
                "comparison_family": "foundation_linear_probe",
                "model_label": "labram",
                "primary_metric": 0.67,
            },
            {
                "scope": "EC",
                "target": "adhd",
                "comparison_family": "descriptor_flat_baseline",
                "model_label": "logreg",
                "primary_metric": 0.61,
            },
            {
                "scope": "EC",
                "target": "adhd",
                "comparison_family": "foundation_linear_probe",
                "model_label": "labram",
                "primary_metric": 0.7,
            },
        ]
    )

    delta = paired_delta_vs_baseline(
        frame,
        group_columns=("scope", "target"),
        baseline_family="descriptor_flat_baseline",
        compatible_lookup={("EO", "adhd"): True, ("EC", "adhd"): False},
    )

    assert len(delta) == 1
    assert delta.loc[0, "scope"] == "EO"
    assert (
        delta.loc[0, "comparison_pair"] == "foundation_linear_probe | labram - logreg"
    )
    assert delta.loc[0, "comparison_delta"] == pytest.approx(0.07)


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


def _interactive_tables(element):
    """Recursively collect InteractiveTableElement instances under *element*."""
    found = []
    if isinstance(element, InteractiveTableElement):
        found.append(element)
    for child in getattr(element, "children", []):
        found.extend(_interactive_tables(child))
    return found


def _sweep_frame():
    """A small classical decoding-sweep result frame with successes + a failure."""
    return pd.DataFrame(
        [
            {
                "status": "success",
                "scope": "EO",
                "condition": "rest",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Fz",
                "family": "sensor",
                "subfamily": "frontal",
                "model": "rf",
                "model_key": "rf",
                "train_mode": "scratch",
                "selection_mode": "baseline",
                "primary_metric_name": "balanced_accuracy",
                "primary_metric": 0.71,
                "p_value": 0.01,
                "p_value_fdr": 0.02,
                "significant_fdr": True,
            },
            {
                "status": "success",
                "scope": "EO",
                "condition": "rest",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Cz",
                "family": "sensor",
                "subfamily": "central",
                "model": "svm",
                "model_key": "svm",
                "train_mode": "scratch",
                "selection_mode": "sfs",
                "primary_metric_name": "balanced_accuracy",
                "primary_metric": 0.66,
                "p_value": 0.04,
                "p_value_fdr": 0.05,
                "significant_fdr": False,
            },
            {
                "status": "failed",
                "scope": "EO",
                "condition": "rest",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Pz",
                "family": "sensor",
                "subfamily": "parietal",
                "model": "rf",
                "model_key": "rf",
                "train_mode": "scratch",
                "selection_mode": "baseline",
                "primary_metric_name": "balanced_accuracy",
                "primary_metric": float("nan"),
                "reason": "not enough samples",
            },
        ]
    )


def test_leaderboard_section_ranks_and_applies_labels():
    section = leaderboard_section(
        _sweep_frame(),
        title="Primary Leaderboard",
        table_title="Best per unit",
        filters={"analysis_mode": "sensor"},
        comparison_axis="model",
        group_by=("scope",),
    )

    assert section is not None
    tables = _interactive_tables(section)
    assert tables, "leaderboard should render an interactive table"
    columns = list(tables[0].data.columns)
    # Columns are relabeled for display (no raw sweep column names remain).
    assert "Primary Score" in columns
    assert "primary_metric" not in columns
    assert section.children[-1] is not tables[0]  # score-comparison sub-section added


def test_leaderboard_section_returns_none_when_filter_column_missing():
    assert (
        leaderboard_section(
            _sweep_frame(),
            title="t",
            table_title="tt",
            filters={"does_not_exist": 1},
            comparison_axis="model",
            group_by=(),
        )
        is None
    )


def test_failures_section_lists_only_non_success_rows():
    section = failures_section(_sweep_frame())

    assert section is not None
    tables = _interactive_tables(section)
    assert len(tables) == 1
    display = tables[0].data
    assert len(display) == 1  # only the failed Pz row
    assert "Status" in display.columns
    assert "status" not in display.columns


def test_failures_section_returns_none_when_all_successful():
    frame = _sweep_frame()
    frame = frame[frame["status"] == "success"].reset_index(drop=True)
    assert failures_section(frame) is None


def test_feature_selection_section_selects_non_baseline_rows():
    section = feature_selection_section(_sweep_frame(), feature_metadata=None)

    assert section is not None
    tables = _interactive_tables(section)
    assert tables
    display = tables[0].data
    assert len(display) == 1  # only the sfs (non-baseline, successful) Cz row
    assert "Feature Selection" in display.columns


def test_feature_selection_section_returns_none_without_non_baseline():
    frame = _sweep_frame()
    frame["selection_mode"] = "baseline"
    assert feature_selection_section(frame, feature_metadata=None) is None


def test_add_scientific_overview_builds_cards_and_strategy():
    report = Report("Overview")
    add_scientific_overview(
        report,
        _sweep_frame(),
        dataset_name="DemoDS",
        kind="classical",
        strategy_note="Flat baseline is the primary comparison.",
    )

    assert len(report.children) == 1
    html = report.render()
    assert "Scientific Overview" in html
    assert "DemoDS" in html
    assert "Flat baseline is the primary comparison." in html


def test_add_scientific_overview_empty_frame_uses_message():
    report = Report("Overview")
    add_scientific_overview(
        report,
        pd.DataFrame(),
        dataset_name="DemoDS",
        kind="foundation",
        strategy_note="unused",
        empty_message="No units were produced for this run.",
    )

    assert len(report.children) == 1
    assert "No units were produced for this run." in report.render()


def test_flat_section_builds_from_persisted_results(tmp_path):
    frame_rows = []
    for unit in ("Fz", "Cz"):
        output_dir = tmp_path / unit
        make_synthetic_result(n_models=2, n_times=1).save(output_dir / "result.joblib")
        frame_rows.append(
            {
                "status": "success",
                "scope": "EO",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": unit,
                "output_dir": str(output_dir),
            }
        )

    section = flat_section(
        pd.DataFrame(frame_rows),
        scope_label="EO",
        feature_metadata=None,
        kind="model_bars",
        title_prefix="Sensors",
        group_by=(),
        axis="unit_name",
    )

    assert section is not None


def test_flat_section_returns_none_without_successful_results():
    frame = pd.DataFrame(
        [
            {
                "status": "failed",
                "scope": "EO",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Fz",
                "output_dir": "/nonexistent",
            }
        ]
    )
    assert (
        flat_section(
            frame,
            scope_label="EO",
            feature_metadata=None,
            kind="model_bars",
            title_prefix="Sensors",
            group_by=(),
            axis="unit_name",
        )
        is None
    )


def test_make_decoding_sweep_report_runs_skeleton_in_order():
    frame = _sweep_frame()
    seen = {}

    def body(report, body_frame):
        seen["called"] = True
        report.add_section(Section("Custom Body"))

    report = make_decoding_sweep_report(
        [],
        frame=frame,
        title="Sweep",
        kind="classical",
        dataset_name="DemoDS",
        strategy_note="Flat baseline is primary.",
        leaderboards=[
            {
                "title": "Primary Leaderboard",
                "table_title": "Flat baseline",
                "filters": {"analysis_mode": "sensor"},
                "comparison_axis": "model",
                "group_by": ("scope",),
            }
        ],
        body=body,
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    titles = [child.title for child in report.children]
    assert seen.get("called") is True
    assert titles[0] == "Scientific Overview"
    assert "Custom Body" in titles
    # failures section (the frame has a failed row) comes last
    assert titles[-1] == "Skipped and Failed Units"


def test_make_decoding_sweep_report_empty_skips_body():
    called = {"body": False}

    def body(report, body_frame):
        called["body"] = True

    report = make_decoding_sweep_report(
        [],
        title="Sweep",
        kind="foundation",
        strategy_note="note",
        empty_message="No units were produced.",
        body=body,
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    assert called["body"] is False
    assert "No units were produced." in report.render()


def _classical_rows(tmp_path, scope="EO"):
    rows = []
    for unit in ("Fz", "Cz"):
        output_dir = tmp_path / f"{scope}_{unit}"
        make_synthetic_result(n_models=2, n_times=1).save(output_dir / "result.joblib")
        rows.append(
            {
                "status": "success",
                "scope": scope,
                "target": "adhd",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
                "unit_name": unit,
                "model": "logreg",
                "balanced_accuracy_mean": 0.7,
                "output_dir": str(output_dir),
            }
        )
    return rows


def test_make_decoding_report_classical_builds_overview_and_taxonomy(tmp_path):
    report = make_decoding_report(
        _classical_rows(tmp_path),
        title="Classical",
        scope_order=["EO"],
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    titles = [child.title for child in report.children]
    assert titles[0] == "Scientific Overview"
    # the classical taxonomy body renders one section per scope
    assert "EO" in titles


def test_make_decoding_report_flags_missing_configured_scope(tmp_path):
    # A configured scope with no rows is kept and flagged, not dropped.
    report = make_decoding_report(
        _classical_rows(tmp_path),
        title="Classical",
        scope_order=["EO", "EC"],
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    assert "The full analysis is missing or incomplete." in report.render()


def test_build_classical_taxonomy_sections_needs_analysis_mode():
    report = Report(title="x", asset_urls={"plotly": "", "tailwind": "", "pako": ""})
    build_classical_taxonomy_sections(report, pd.DataFrame([{"scope": "EO"}]))
    assert report.children == []


def test_hp_tuning_section_renders_for_tuned_runs(tmp_path):
    section = hp_tuning_section(pd.DataFrame(_classical_rows(tmp_path)))
    assert section is not None
    assert section.title == "Hyperparameter Tuning"


def test_hp_tuning_section_returns_none_without_output_dirs():
    # No loadable artifacts -> nothing to inspect -> section omitted.
    frame = pd.DataFrame(
        [{"status": "success", "analysis_mode": "flat", "primary_metric": 0.7}]
    )
    assert hp_tuning_section(frame) is None


def test_prepare_sweep_frame_defaults_scope_and_primary_metric():
    frame = prepare_sweep_frame(
        [{"target": "adhd", "balanced_accuracy_mean": 0.7}],
        scope_from=None,
        default_scope="all",
    )
    assert (frame["scope"] == "all").all()
    assert "primary_metric" in frame.columns


def _head_to_head_frame():
    return pd.DataFrame(
        [
            {
                "comparison_family": "descriptor_flat_baseline",
                "scope": "EO",
                "condition": "EO",
                "target": "adhd",
                "model": "logreg",
                "primary_metric": 0.60,
                "primary_metric_name": "balanced_accuracy",
            },
            {
                "comparison_family": "foundation_linear_probe",
                "scope": "EO",
                "condition": "EO",
                "target": "adhd",
                "model_key": "labram",
                "primary_metric": 0.70,
                "primary_metric_name": "balanced_accuracy",
            },
        ]
    )


def test_enrich_head_to_head_frame_derives_labels():
    enriched = enrich_head_to_head_frame(_head_to_head_frame())
    assert "comparison_label" in enriched.columns
    assert "model_label" in enriched.columns
    assert set(enriched["model_label"]) == {"logreg", "labram"}


def test_make_head_to_head_report_builds_sections():
    report = make_head_to_head_report(
        _head_to_head_frame(),
        baseline_family="descriptor_flat_baseline",
        group_columns=("scope", "target"),
        title="Head-to-Head",
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    html = report.render()
    assert "Head-to-Head Comparison" in html
    assert "Head-to-Head Accuracy" in html
    assert "Paired Delta vs descriptor_flat_baseline" in html


def test_make_head_to_head_report_handles_empty():
    report = make_head_to_head_report(
        pd.DataFrame(),
        baseline_family="descriptor_flat_baseline",
        title="Head-to-Head",
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    assert "Head-to-Head Comparison" in report.render()


def test_collect_mode_results_loads_successful_unique_artifacts(tmp_path):
    output_dir = tmp_path / "unit"
    make_synthetic_result(n_models=1, n_times=1).save(output_dir / "result.joblib")
    frame = pd.DataFrame(
        [
            {
                "status": "success",
                "scope": "EO",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Fz",
                "output_dir": str(output_dir),
            },
            {
                "status": "failed",
                "scope": "EO",
                "target": "adhd",
                "analysis_mode": "sensor",
                "unit_name": "Cz",
                "output_dir": str(tmp_path / "failed"),
            },
        ]
    )

    collection = collect_mode_results(
        frame,
        context_columns=("scope", "target", "analysis_mode", "unit_name"),
    )

    assert collection is not None
    assert len(collection.results) == 1
    assert set(collection.summary["unit_name"]) == {"Fz"}
