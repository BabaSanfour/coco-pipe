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
    run_decoding_unit,
)
from coco_pipe.decoding.pipeline import allocate_inner_jobs
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
    assert (tmp_path / "unit" / "_SUCCESS").exists()
    assert (tmp_path / "unit" / "predictions.csv").exists()

    resumed = run_decoding_unit(_unit(tmp_path))
    assert all(r["reason"] == "resumed" for r in resumed)


def test_run_decoding_unit_records_failure(tmp_path):
    records = run_decoding_unit(_unit(tmp_path, y=np.array([0, 1, 0])), errors="record")
    assert records[0]["status"] == "failed"
    assert (tmp_path / "unit" / "_FAILED").exists()


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
