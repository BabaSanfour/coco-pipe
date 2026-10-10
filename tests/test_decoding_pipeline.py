"""Tests for the decoding per-unit runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding import (
    ChanceAssessmentConfig,
    ClassicalModelConfig,
    CVConfig,
    DecodingUnit,
    ExperimentConfig,
    StatisticalAssessmentConfig,
    TuningConfig,
    execute_decoding_sweep,
    execute_decoding_sweep_streaming,
    load_sweep_records,
    read_run_manifest,
    read_sweep,
    run_decoding_unit,
)
from coco_pipe.decoding.pipeline import allocate_inner_jobs, allocate_outer_inner
from coco_pipe.utils import run_task_batch


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        tag="unit",
        random_state=42,
        models={
            "logreg": ClassicalModelConfig(
                estimator="LogisticRegression",
                params={"solver": "liblinear", "max_iter": 200},
                input_kind="tabular",
            )
        },
        cv=CVConfig(
            strategy="stratified_group_kfold",
            n_splits=2,
            shuffle=True,
            random_state=42,
            group_key="group_id",
        ),
        statistical_assessment=StatisticalAssessmentConfig(
            enabled=True,
            metrics=["accuracy"],
            chance=ChanceAssessmentConfig(
                method="binomial", store_null_distribution=False
            ),
            unit_of_inference="custom",
            custom_unit_column="group_id",
        ),
        metrics=["accuracy"],
        n_jobs=1,
    )


def _unit(tmp_path, **overrides) -> DecodingUnit:
    rng = np.random.default_rng(3)
    n = 20
    groups = np.repeat([f"p{idx:02d}" for idx in range(10)], 2)
    y = np.repeat([0] * 5 + [1] * 5, 2)
    X = rng.normal(size=(n, 4))
    X[:, 0] += y * 1.5
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{idx}" for idx in range(n)],
            "group_id": groups,
            "Subject": groups,
            "Session": ["01"] * n,
        }
    )
    context = {"scope": "cond", "target": "adhd", "analysis_mode": "flat"}
    defaults = {
        "experiment_config": _experiment_config(),
        "X": X,
        "y": y,
        "output_dir": tmp_path / "unit",
        "context": context,
        "run_config": {"tag": "unit", **context},
        "groups": groups,
        "feature_names": ["a", "b", "c", "d"],
        "sample_ids": metadata["sample_id"],
        "sample_metadata": metadata,
        "inferential_unit": "group_id",
    }
    return DecodingUnit(**{**defaults, **overrides})


def test_run_decoding_unit_fresh_then_resume(tmp_path):
    records = run_decoding_unit(_unit(tmp_path))
    assert records and all(r["status"] == "success" for r in records)
    assert sorted(path.name for path in (tmp_path / "unit").iterdir()) == [
        "result.joblib",
        "run_manifest.json",
    ]

    resumed = run_decoding_unit(_unit(tmp_path))
    assert all(r["reason"] == "resumed" for r in resumed)


def test_resume_refuses_a_fit_made_on_other_data(tmp_path):
    # Same config, different rows: the stale fit must not be reused.
    run_decoding_unit(_unit(tmp_path))
    unit = _unit(tmp_path)
    shifted = _unit(tmp_path, X=unit.X + 1.0)
    with pytest.raises(RuntimeError, match="input data changed"):
        run_decoding_unit(shifted)
    relabelled = _unit(tmp_path, sample_ids=[f"t{idx}" for idx in range(20)])
    with pytest.raises(RuntimeError, match="input data changed"):
        run_decoding_unit(relabelled)


def test_records_carry_cv_design_and_sample_set(tmp_path):
    fresh = run_decoding_unit(_unit(tmp_path))
    resumed = run_decoding_unit(_unit(tmp_path))
    for record in (fresh[0], resumed[0]):
        assert record["cv_strategy"] == "stratified_group_kfold"
        assert record["effective_n_splits"] == 2
        assert record["cv_random_state"] == 42
        assert record["n_samples"] == 20
        assert record["n_groups"] == 10
    assert fresh[0]["cohort_signature"] == resumed[0]["cohort_signature"]

    # The sample-set signature ignores row order but not membership.
    unit = _unit(tmp_path)
    order = np.arange(20)[::-1]
    reordered = _unit(
        tmp_path / "reordered",
        X=unit.X[order],
        y=unit.y[order],
        groups=unit.groups[order],
        sample_ids=unit.sample_ids.iloc[order].reset_index(drop=True),
        sample_metadata=unit.sample_metadata.iloc[order].reset_index(drop=True),
    )
    assert (
        run_decoding_unit(reordered)[0]["cohort_signature"]
        == fresh[0]["cohort_signature"]
    )


def test_run_decoding_unit_records_failure(tmp_path):
    records = run_decoding_unit(_unit(tmp_path, y=np.array([0, 1, 0])), errors="record")
    assert records[0]["status"] == "failed"
    assert read_run_manifest(tmp_path / "unit")["status"] == "failed"


def test_run_decoding_unit_raises_on_config_hash_mismatch(tmp_path):
    run_decoding_unit(_unit(tmp_path))
    drifted = _unit(tmp_path, run_config={"tag": "unit", "drift": True})
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_decoding_unit(drifted)


def test_units_run_through_task_batch(tmp_path):
    units = [_unit(tmp_path / f"u{idx}") for idx in range(3)]
    results = run_task_batch(units, run_decoding_unit, max_workers=2)
    assert len(results) == 3
    assert all(records and records[0]["status"] == "success" for records in results)


def test_streaming_sweep_leaves_a_partial_inventory_when_it_breaks(tmp_path):
    def batches():
        yield [_unit(tmp_path / "first")]
        raise RuntimeError("job killed")

    with pytest.raises(RuntimeError, match="job killed"):
        execute_decoding_sweep_streaming(
            batches(), [], config={"n_jobs": 1}, output_root=tmp_path / "sweep"
        )

    assert (tmp_path / "sweep" / "_PARTIAL").exists()
    (record,) = load_sweep_records(tmp_path / "sweep")
    assert record["status"] == "success"
    assert record["output_dir"] == str(tmp_path / "first" / "unit")


def test_read_sweep_gives_the_table_and_loads_units_on_demand(tmp_path):
    units = [_unit(tmp_path / name) for name in ("a", "b")]
    execute_decoding_sweep(
        units, [], config={"n_jobs": 1}, output_root=tmp_path / "sweep"
    )

    sweep = read_sweep(tmp_path / "sweep")

    assert sweep.status == "SUCCESS"  # without a primary mask any success counts
    assert sweep.results["output_dir"].tolist() == [str(u.output_dir) for u in units]
    assert sweep.results["accuracy_mean"].notna().all()
    one = sweep.results.iloc[[1]]
    predictions = sweep.predictions(one)
    assert set(predictions["output_dir"]) == {str(units[1].output_dir)}
    assert len(predictions) == len(units[1].y)
    assert sweep.result(units[0].output_dir).summary().index.tolist() == ["logreg"]


def test_job_budget_is_shared_between_units():
    assert allocate_outer_inner(32, 2580) == (32, 1)
    assert allocate_outer_inner(32, 4) == (4, 8)
    assert allocate_outer_inner(32, 1) == (1, 32)


def test_allocate_inner_jobs_splits_budget_across_fold_and_tuning_levels():
    cfg = _experiment_config().model_copy(
        update={
            "cv": CVConfig(
                strategy="stratified_group_kfold",
                n_splits=5,
                shuffle=True,
                random_state=42,
                group_key="group_id",
            ),
            "tuning": TuningConfig(enabled=True, scoring="accuracy"),
        }
    )

    reallocated = allocate_inner_jobs(cfg, 32)

    # Handing 32 to both the 5-fold outer loop and each fold's grid search
    # would oversubscribe by 5x; the two levels must instead multiply to at
    # most the original budget.
    assert reallocated.n_jobs == 5
    assert reallocated.tuning.n_jobs == 6
    assert reallocated.n_jobs * reallocated.tuning.n_jobs <= 32


def test_allocate_inner_jobs_with_more_splits_than_jobs():
    cfg = _experiment_config().model_copy(
        update={
            "cv": CVConfig(
                strategy="stratified_group_kfold",
                n_splits=10,
                shuffle=True,
                random_state=42,
                group_key="group_id",
            ),
            "tuning": TuningConfig(enabled=True, scoring="accuracy"),
        }
    )

    reallocated = allocate_inner_jobs(cfg, 4)

    assert reallocated.n_jobs == 4
    assert reallocated.tuning.n_jobs == 1
