import json

import numpy as np
import pandas as pd

from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY, SEPARATION_RF_METRIC_KEY
from coco_pipe.dim_reduction.artifacts import save_fit_artifact
from coco_pipe.io import DataContainer
from coco_pipe.report.core import Report, Section
from coco_pipe.report.dim_reduction_sweep import (
    DimReductionReportContext,
    add_reduction_best_run_cards,
    build_best_fit_plots,
    build_dataset_report,
    build_meta_dict,
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


def test_merge_fit_eval_left_merges_rf_and_logreg_metrics():
    eval_runs = pd.DataFrame(
        [
            {
                "fit_id": "a",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_RF_METRIC_KEY: 0.55,
                SEPARATION_METRIC_KEY: 0.60,
            },
            {
                "fit_id": "b",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_RF_METRIC_KEY: 0.80,
                SEPARATION_METRIC_KEY: 0.50,
            },
        ]
    )

    merged = merge_fit_eval(_fit_runs(), eval_runs, eval_name="sep")

    assert {SEPARATION_RF_METRIC_KEY, SEPARATION_METRIC_KEY} <= set(merged.columns)
    assert merged.loc[merged["fit_id"] == "b", SEPARATION_RF_METRIC_KEY].iloc[0] == 0.80
    assert merged.loc[merged["fit_id"] == "b", SEPARATION_METRIC_KEY].iloc[0] == 0.50


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


def test_rank_reduction_runs_can_select_by_rf_before_lr_or_geometry():
    fit = pd.DataFrame(
        [
            {
                "fit_id": "a",
                "scope": "condition",
                "condition": "EO",
                "reducer": "pca",
                "status": "success",
                "trustworthiness": 0.95,
            },
            {
                "fit_id": "b",
                "scope": "condition",
                "condition": "EO",
                "reducer": "umap",
                "status": "success",
                "trustworthiness": 0.60,
            },
        ]
    )
    eval_runs = pd.DataFrame(
        [
            {
                "fit_id": "a",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_RF_METRIC_KEY: 0.50,
                SEPARATION_METRIC_KEY: 0.90,
            },
            {
                "fit_id": "b",
                "status": "success",
                "eval_name": "sep",
                SEPARATION_RF_METRIC_KEY: 0.75,
                SEPARATION_METRIC_KEY: 0.40,
            },
        ]
    )

    best = rank_reduction_runs(
        fit,
        eval_runs,
        selection_metric=SEPARATION_RF_METRIC_KEY,
        group_by=("scope", "condition"),
    )

    assert best.loc[0, "fit_id"] == "b"
    assert best.loc[0, SEPARATION_RF_METRIC_KEY] == 0.75
    assert best.loc[0, SEPARATION_METRIC_KEY] == 0.40


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


_FEATURES = ["f0", "f1", "f2", "f3"]
_CHANNELS = ["Fz", "Cz", "Pz", "Oz"]


def _eye_state(meta):
    if "condition" not in meta:
        return None
    arr = np.array(
        ["OPEN" if str(v).endswith("o") else "CLOSED" for v in meta["condition"]],
        dtype=object,
    )
    return arr if len(np.unique(arr)) > 1 else None


def _container(condition, *, features=_FEATURES):
    """Small (6 obs by n_feat) container with condition + group metadata."""
    n_obs = 6
    ids = np.array([f"{condition}-s{i}" for i in range(n_obs)], dtype=object)
    conditions = np.array(
        [f"{condition}o" if i % 2 else f"{condition}c" for i in range(n_obs)]
    )
    return DataContainer(
        X=np.random.default_rng(0).normal(size=(n_obs, len(features))),
        dims=("obs", "feature"),
        coords={
            "feature": np.asarray(features),
            "condition": conditions,
            "group": np.array(["A", "B", "A", "B", "A", "B"]),
            "age": np.arange(n_obs),  # excluded via excluded_normalized
        },
        y=np.array([0, 1, 0, 1, 0, 1]),
        ids=ids,
        meta={"loaded_obs": n_obs},
    )


def _ctx(tmp_path, **overrides):
    defaults = {
        "analysis_mode": "flat",
        "selection_metric": "trustworthiness",
        "reducers": ["pca"],
        "conditions": ["EO", "EC"],
        "container_builder": lambda condition: _container(condition),
        "output_root": tmp_path,
        "eval_specs": (),
        "interactive": False,
        "input_mode": "raw",
        "representation": "sensors",
        "excluded_normalized": frozenset({"age"}),
        "excluded_normalized_substrings": ("psychostimulant",),
        "excluded_suffixes": ("_bool",),
        "meta_extractors": {"eye_state": _eye_state},
        "topomap_channels": frozenset(c.lower() for c in _CHANNELS),
        "unit_labels": {"flat": "global"},
        "fit_failure_columns": ["scope", "condition", "reducer", "status", "error"],
        "eval_failure_columns": ["scope", "condition", "eval_name", "status", "error"],
        "report_title": "Test Report",
        "dataset_name": "ds",
    }
    defaults.update(overrides)
    return DimReductionReportContext(**defaults)


def _artifact(n_components, n_features):
    return {
        "embedding": np.random.default_rng(1).normal(size=(6, n_components)),
        "ids": np.array([f"s{i}" for i in range(6)], dtype=object),
        "diagnostics": {
            "components": np.random.default_rng(2).normal(
                size=(n_components, n_features)
            )
        },
    }


# --- build_meta_dict --------------------------------------------------------


def test_build_meta_dict_applies_exclusions_and_extractors(tmp_path):
    ctx = _ctx(tmp_path)
    container = _container("EO")
    meta = build_meta_dict(container, container.ids, ctx)

    assert "group" in meta  # ordinary metadata kept
    assert "y" in meta  # target attached from container.y
    assert "age" not in meta  # excluded via excluded_normalized
    assert "feature" not in meta  # feature column always dropped
    assert set(np.unique(meta["eye_state"])) == {"OPEN", "CLOSED"}  # extractor ran


def test_build_meta_dict_skips_extractor_without_condition(tmp_path):
    ctx = _ctx(tmp_path)
    container = DataContainer(
        X=np.zeros((3, 2)),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "group": ["A", "B", "A"]},
        ids=np.array(["x", "y", "z"], dtype=object),
    )
    meta = build_meta_dict(container, container.ids, ctx)
    assert "eye_state" not in meta
    assert "group" in meta


# --- build_best_fit_plots ---------------------------------------------------


def test_best_fit_plots_uses_loadings_for_non_channel_features(tmp_path):
    ctx = _ctx(tmp_path)
    element = build_best_fit_plots(
        "title", _artifact(2, 4), {}, ctx, feature_names=_FEATURES
    )
    assert element is not None  # embedding + loadings columns


def test_best_fit_plots_topomap_gate_for_channel_features(tmp_path):
    ctx = _ctx(tmp_path)
    # Channel feature names pass the topomap gate; render must not raise.
    element = build_best_fit_plots(
        "title", _artifact(2, 4), {}, ctx, feature_names=_CHANNELS
    )
    assert element is not None


def test_best_fit_plots_none_for_1d_embedding(tmp_path):
    ctx = _ctx(tmp_path)
    artifact = {"embedding": np.zeros((6,)), "ids": np.arange(6), "diagnostics": {}}
    assert build_best_fit_plots("t", artifact, {}, ctx) is None


# --- build_dataset_report (end to end) --------------------------------------


def _write_flat_runs(tmp_path):
    """Write two conditions x pca(n=2,3) artifacts + fit/eval inventories."""
    fit_rows = []
    eval_rows = []
    for condition in ("EO", "EC"):
        for n in (2, 3):
            fit_id = f"{condition}-pca-{n}"
            rel = f"artifacts/{fit_id}"
            save_fit_artifact(
                tmp_path / rel,
                embedding=np.random.default_rng(n).normal(size=(6, n)),
                ids=np.array([f"{condition}-s{i}" for i in range(6)], dtype=object),
                fit_payload={"reducer": "pca", "n_components": n},
                metrics_payload={},
                diagnostics={
                    "components": np.random.default_rng(n).normal(size=(n, 4))
                },
            )
            fit_rows.append(
                {
                    "fit_id": fit_id,
                    "scope": "condition",
                    "condition": condition,
                    "analysis_mode": "flat",
                    "reducer": "pca",
                    "n_components": n,
                    "status": "success",
                    "artifact_path": rel,
                    "trustworthiness": 0.8 + 0.01 * n,
                    "continuity": 0.7,
                }
            )
            eval_rows.append(
                {
                    "fit_id": fit_id,
                    "scope": "condition",
                    "condition": condition,
                    "analysis_mode": "flat",
                    "eval_name": "dx",
                    "target_col": "group",
                    "reducer": "pca",
                    "n_components": n,
                    "status": "success",
                    SEPARATION_METRIC_KEY: 0.6,
                }
            )
    # A failed fit to exercise the failure section.
    fit_rows.append(
        {
            "fit_id": "EO-pca-bad",
            "scope": "condition",
            "condition": "EO",
            "reducer": "pca",
            "n_components": 9,
            "status": "failed",
            "error": "boom",
        }
    )
    fit_path = tmp_path / "fit_runs.json"
    eval_path = tmp_path / "eval_runs.json"
    fit_path.write_text(json.dumps(fit_rows), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_rows), encoding="utf-8")
    return fit_path, eval_path


def test_build_dataset_report_flat_end_to_end(tmp_path):
    fit_path, eval_path = _write_flat_runs(tmp_path)
    ctx = _ctx(tmp_path)

    seen = []

    def _overview_extras(section):
        seen.append(section.title)

    report = build_dataset_report(
        ctx,
        fit_runs_path=fit_path,
        eval_runs_path=eval_path,
        dataset_stats=[{"scope": "condition", "condition": "EO", "samples_used": 6}],
        overview_extras=_overview_extras,
    )

    assert isinstance(report, Report)
    titles = [s.title for s in report.children if isinstance(s, Section)]
    assert titles[0] == "Overview"
    assert "EO" in titles and "EC" in titles  # per-condition sections
    assert "Fit Failures" in titles  # failure section rendered
    assert seen == ["Overview"]  # overview_extras hook invoked with the section
    assert "Test Report" in report.render()  # renders to HTML without error
