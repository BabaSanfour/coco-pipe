import pandas as pd

from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY
from coco_pipe.report.core import Section
from coco_pipe.report.dim_reduction_sweep import (
    add_reduction_best_run_cards,
    build_reduction_condition_ranking_section,
    build_reduction_eval_results_section,
    build_reduction_rollup_report,
    merge_fit_eval,
    rank_reduction_runs,
)

_ASSETS = {"plotly": "", "tailwind": "", "pako": ""}


def _fit_runs():
    return pd.DataFrame(
        [
            {
                "fit_id": "a",
                "scope": "condition",
                "condition": "EO",
                "reducer": "pca",
                "status": "success",
                "trustworthiness": 0.80,
            },
            {
                "fit_id": "b",
                "scope": "condition",
                "condition": "EO",
                "reducer": "umap",
                "status": "success",
                "trustworthiness": 0.90,
            },
            {
                "fit_id": "c",
                "scope": "condition",
                "condition": "EC",
                "reducer": "pca",
                "status": "success",
                "trustworthiness": 0.70,
            },
            {
                "fit_id": "d",
                "scope": "condition",
                "condition": "EO",
                "reducer": "pca",
                "status": "failed",
                "trustworthiness": 0.99,
            },
        ]
    )


def _eval_runs():
    return pd.DataFrame(
        [
            {
                "fit_id": "a",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_METRIC_KEY: 0.60,
            },
            {
                "fit_id": "b",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_METRIC_KEY: 0.50,
            },
        ]
    )


def test_merge_fit_eval_left_merges_success_metric():
    merged = merge_fit_eval(_fit_runs(), _eval_runs(), eval_name="sep")
    assert SEPARATION_METRIC_KEY in merged.columns
    assert len(merged) == len(_fit_runs())  # left merge, no row multiplication
    assert merged.loc[merged["fit_id"] == "a", SEPARATION_METRIC_KEY].iloc[0] == 0.60


def test_merge_fit_eval_returns_fit_when_no_eval():
    fit = _fit_runs()
    out = merge_fit_eval(fit, None)
    assert list(out.columns) == list(fit.columns)
    assert SEPARATION_METRIC_KEY not in out.columns


def test_rank_reduction_runs_best_per_group_excludes_failures():
    best = rank_reduction_runs(
        _fit_runs(),
        _eval_runs(),
        selection_metric="trustworthiness",
        group_by=("scope", "condition"),
    )
    groups = set(zip(best["scope"], best["condition"], strict=False))
    assert groups == {("condition", "EO"), ("condition", "EC")}
    # EO winner is the umap run (0.90), not the failed pca run (0.99)
    assert best.loc[best["condition"] == "EO", "fit_id"].iloc[0] == "b"
    assert SEPARATION_METRIC_KEY in best.columns  # eval metric merged in


def test_rank_reduction_runs_reducer_filter():
    best = rank_reduction_runs(
        _fit_runs(),
        _eval_runs(),
        selection_metric="trustworthiness",
        group_by=("scope", "condition"),
        reducers=["pca"],
    )
    # restricted to pca -> EO winner becomes run "a"
    assert best.loc[best["condition"] == "EO", "fit_id"].iloc[0] == "a"


def test_rank_reduction_runs_empty_without_success():
    fit = _fit_runs()
    fit["status"] = "failed"
    assert rank_reduction_runs(fit, None, selection_metric="trustworthiness").empty


def test_build_reduction_rollup_report_builds_sections():
    leaderboard = pd.DataFrame(
        [
            {
                "analysis_mode": "flat",
                "condition": "EO",
                "reducer": "pca",
                "trustworthiness": 0.8,
                SEPARATION_METRIC_KEY: 0.6,
            },
            {
                "analysis_mode": "sensor",
                "condition": "EC",
                "reducer": "umap",
                "trustworthiness": 0.7,
                SEPARATION_METRIC_KEY: 0.65,
            },
        ]
    )
    report = build_reduction_rollup_report(
        leaderboard,
        title="Roll-up",
        mode_label_map={"flat": "Flat", "sensor": "Sensor"},
        link_rows=[{"analysis_mode": "flat", "report": "x"}],
        task_failures=[{"analysis_mode": "descriptor", "reason": "no runs"}],
        asset_urls=_ASSETS,
    )
    titles = [section.title for section in report.children]
    assert "Roll-up Overview" in titles
    assert "Leaderboard" in titles
    assert "Per-mode reports" in titles
    assert "Task Failures" in titles


def test_build_reduction_rollup_report_empty_leaderboard():
    report = build_reduction_rollup_report(
        pd.DataFrame(), title="Roll-up", asset_urls=_ASSETS
    )
    assert "No successful runs" in report.render()


def _condition_runs():
    return pd.DataFrame(
        [
            {
                "condition": "EO",
                "reducer": "pca",
                "status": "success",
                "n_components": 5,
                "trustworthiness": 0.80,
                "continuity": 0.7,
                SEPARATION_METRIC_KEY: 0.6,
            },
            {
                "condition": "EO",
                "reducer": "umap",
                "status": "success",
                "n_components": 2,
                "trustworthiness": 0.85,
                "continuity": 0.75,
                SEPARATION_METRIC_KEY: 0.65,
            },
            {
                "condition": "EC",
                "reducer": "pca",
                "status": "success",
                "n_components": 5,
                "trustworthiness": 0.70,
                "continuity": 0.6,
                SEPARATION_METRIC_KEY: 0.55,
            },
            {
                "condition": "EC",
                "reducer": "umap",
                "status": "success",
                "n_components": 3,
                "trustworthiness": 0.78,
                "continuity": 0.68,
                SEPARATION_METRIC_KEY: 0.62,
            },
        ]
    )


def test_add_reduction_best_run_cards_one_block_per_group():
    section = Section("Overview")
    runs = _condition_runs()
    runs.loc[len(runs)] = {
        "condition": "EO",
        "reducer": "pca",
        "status": "failed",
        "n_components": 9,
        "trustworthiness": 0.99,
        "continuity": 0.99,
        SEPARATION_METRIC_KEY: 0.99,
    }
    add_reduction_best_run_cards(section, runs, selection_metric="trustworthiness")
    # one callout + one columns element per successful condition (EO, EC)
    assert len(section.children) == 4
    rendered = section.render()
    assert "Peak Performance" in rendered
    # the failed 0.99 run must not win EO
    assert "umap" in rendered


def test_add_reduction_best_run_cards_noop_without_success():
    section = Section("Overview")
    runs = _condition_runs()
    runs["status"] = "failed"
    add_reduction_best_run_cards(section, runs, selection_metric="trustworthiness")
    assert section.children == []


def test_build_reduction_eval_results_section():
    eval_frame = pd.DataFrame(
        [
            {
                "scope": "condition",
                "condition": "EO",
                "reducer": "pca",
                "eval_name": "sep",
                SEPARATION_METRIC_KEY: 0.6,
            }
        ]
    )
    section = build_reduction_eval_results_section(eval_frame)
    assert section is not None and section.title == "Evaluation Results"
    assert build_reduction_eval_results_section(pd.DataFrame()) is None


def test_build_reduction_condition_ranking_section_table_and_tabs():
    section = build_reduction_condition_ranking_section(
        _condition_runs(),
        conditions=["EO", "EC"],
        reducers=["pca", "umap"],
        selection_metric="trustworthiness",
    )
    assert section is not None and section.title == "Condition Ranking"
    # multi-condition + multi-reducer -> cross-condition bar and reducer radar tabs
    rendered = section.render()
    assert "Cross-Condition Summary" in rendered
    assert "Reducer Profile" in rendered


def test_build_reduction_condition_ranking_section_empty():
    assert (
        build_reduction_condition_ranking_section(
            pd.DataFrame(),
            conditions=["EO"],
            reducers=["pca"],
            selection_metric="trustworthiness",
        )
        is None
    )
