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
    run_decoding_unit,
)
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


def test_unit_records_surfaces_model_failure(tmp_path):
    """Regression test for issue #21.

    When a model's result carries an 'error' key it is absent from
    ExperimentResult.summary().  _unit_records must still return one
    record per model (status='failed') so callers never receive an
    empty list or hit KeyError: 'Model' when indexing into records.
    """
    from pathlib import Path
    from unittest.mock import MagicMock

    import pandas as pd

    from coco_pipe.decoding.pipeline import _unit_records

    # Build a mock ExperimentResult where 'bad_model' failed.
    result = MagicMock()
    result.raw = {
        "good_model": {
            "metrics": {"accuracy": {"mean": 0.9, "std": 0.05, "folds": [0.9]}}
        },
        "bad_model": {"error": "Degenerate Test Fold: Only one class found (1)."},
    }
    # summary() returns only the good model (bad_model is skipped via 'error' check).
    result.summary.return_value = pd.DataFrame(
        [{"accuracy_mean": 0.9, "accuracy_std": 0.05}],
        index=pd.Index(["good_model"], name="Model"),
    )
    result.get_statistical_assessment.return_value = pd.DataFrame()

    records = _unit_records(
        result,
        context={"scope": "test"},
        output_dir=Path(tmp_path),
        include_p_values=False,
        metrics=["accuracy"],
    )

    # Must have exactly one record per model.
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"

    by_model = {r["model"]: r for r in records}
    assert "good_model" in by_model
    assert "bad_model" in by_model

    assert by_model["good_model"]["status"] == "success"
    assert by_model["bad_model"]["status"] == "failed"
    assert "Degenerate" in by_model["bad_model"]["reason"]
