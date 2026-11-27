import json

import numpy as np
import pytest

from coco_pipe.dim_reduction.artifacts import (
    EVAL_METRIC_COLUMNS,
    FIT_METRIC_COLUMNS,
    SEPARATION_METRIC_KEY,
    SEPARATION_RF_METRIC_KEY,
    _embedding_container,
    build_availability_record,
    build_record,
    load_fit_artifact,
    load_fit_runs,
    save_fit_artifact,
    update_runs,
    write_run_status,
)
from coco_pipe.io.structures import DataContainer


def test_save_and_load_fit_artifact(tmp_path):
    embedding = np.random.rand(10, 2)
    ids = np.arange(10)
    fit_payload = {"reducer": "PCA", "artifact_stem": "test_fit"}
    metrics = {"trustworthiness": 0.9}
    diagnostics = {"explained_variance": [0.5, 0.4]}

    path = tmp_path / "test_fit.npz"
    save_fit_artifact(path, embedding, ids, fit_payload, metrics, diagnostics)

    # One file per fit.
    assert [p.name for p in tmp_path.iterdir()] == ["test_fit.npz"]

    loaded = load_fit_artifact(path)
    np.testing.assert_allclose(loaded["embedding"], embedding)
    np.testing.assert_array_equal(loaded["ids"], ids)
    assert loaded["fit"] == fit_payload
    assert loaded["metrics"] == metrics
    assert (
        loaded["diagnostics"]["explained_variance"] == diagnostics["explained_variance"]
    )
    assert loaded["manifest"]["artifact_stem"] == "test_fit"
    assert loaded["path"] == path

    # The embedding is also exposed as a DataContainer view (carry structure),
    # while the raw embedding/ids arrays remain for direct access.
    container = loaded["embedding_container"]
    assert isinstance(container, DataContainer)
    assert container.dims == ("obs", "component")
    np.testing.assert_allclose(container.X, embedding)
    np.testing.assert_array_equal(container.ids, ids)
    assert list(container.coords["component"]) == ["component_1", "component_2"]
    assert container.meta["fit"] == fit_payload


def test_embedding_container_returns_none_for_non_2d():
    # Native trajectory tensors do not map onto a component axis.
    traj = np.zeros((4, 10, 2))
    assert _embedding_container(traj, np.arange(4), {"reducer": "PCA"}) is None


def test_update_and_load_runs(tmp_path):
    runs_file = tmp_path / "runs.json"

    with pytest.raises(RuntimeError, match="No fit runs found"):
        load_fit_runs(runs_file)

    record1 = {"fit_id": "1", "score": 0.5, "scope": "A"}
    update_runs(runs_file, record1, key_fields=["fit_id"])

    loaded = load_fit_runs(runs_file)
    assert len(loaded) == 1
    assert loaded[0] == record1

    # Update existing
    record1_updated = {"fit_id": "1", "score": 0.8, "scope": "A"}
    update_runs(runs_file, record1_updated, key_fields=["fit_id"])

    # Insert new
    record2 = {"fit_id": "2", "score": 0.9, "scope": "B"}
    update_runs(runs_file, record2, key_fields=["fit_id"])

    loaded = load_fit_runs(runs_file)
    assert len(loaded) == 2
    assert {r["fit_id"]: r["score"] for r in loaded} == {"1": 0.8, "2": 0.9}

    # Test corrupted file
    runs_file.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="Expected list payload"):
        load_fit_runs(runs_file)
    with pytest.raises(ValueError, match="Expected list payload"):
        update_runs(runs_file, record2, key_fields=["fit_id"])


def test_build_records(tmp_path):
    payload = {"fit_id": "1", "metrics": {"trust": 0.9}, "artifacts": ["file"]}
    metrics_payload = {"trustworthiness": 0.95}

    # Fit record success
    rec = build_record(
        payload, tmp_path / "art", tmp_path, FIT_METRIC_COLUMNS, metrics_payload
    )
    assert rec["status"] == "success"
    assert rec["trustworthiness"] == 0.95
    assert np.isnan(rec["lcmc"])
    assert "metrics" not in rec
    assert "artifacts" not in rec

    # Fit record with error
    rec2 = build_record(
        payload, tmp_path / "art", tmp_path, FIT_METRIC_COLUMNS, None, error="failed"
    )
    assert rec2["status"] == "failed"
    assert rec2["error"] == "failed"
    assert np.isnan(rec2["trustworthiness"])

    # Eval record keeps RF first and promotes both separation probes.
    assert EVAL_METRIC_COLUMNS == [SEPARATION_RF_METRIC_KEY, SEPARATION_METRIC_KEY]
    rec3 = build_record(
        payload,
        tmp_path / "art",
        tmp_path,
        EVAL_METRIC_COLUMNS,
        {SEPARATION_RF_METRIC_KEY: 0.7, SEPARATION_METRIC_KEY: 0.8},
    )
    assert rec3[SEPARATION_RF_METRIC_KEY] == 0.7
    assert rec3[SEPARATION_METRIC_KEY] == 0.8


def test_write_run_status(tmp_path):
    fit_runs = tmp_path / "fit_runs.json"
    eval_runs = tmp_path / "eval_runs.json"

    # Create dummy runs
    update_runs(fit_runs, {"fit_id": "1", "status": "success"}, ["fit_id"])
    update_runs(eval_runs, {"fit_id": "1", "status": "failed"}, ["fit_id"])

    payload = write_run_status(
        tmp_path, fit_runs, eval_runs, run_metadata={"foo": "bar"}
    )

    assert (tmp_path / "run_summary.json").exists()
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert payload == summary
    assert summary["status"] == "partial"
    assert summary["foo"] == "bar"
    assert (tmp_path / "_RUN_PARTIAL").exists()
    assert not (tmp_path / "_RUN_SUCCESS").exists()

    # Test full fail
    write_run_status(
        tmp_path, tmp_path / "none.json", tmp_path / "none.json", fatal_error="err"
    )
    assert (tmp_path / "_RUN_FAILED").exists()

    # Test success
    fit_runs.unlink()
    eval_runs.unlink()
    update_runs(fit_runs, {"fit_id": "1", "status": "success"}, ["fit_id"])
    write_run_status(tmp_path, fit_runs, eval_runs)
    assert (tmp_path / "_RUN_SUCCESS").exists()

    custom_summary = tmp_path / "runs" / "run_summary.json"
    write_run_status(tmp_path, fit_runs, eval_runs, run_summary_path=custom_summary)
    assert custom_summary.exists()


def test_availability_record():
    class DummyContainer:
        def __init__(self, X):
            self.X = X

    container = DummyContainer(np.zeros((10, 5)))
    rec = build_availability_record(
        scope="all",
        condition="cond",
        unit_spec={"unit_type": "all", "unit_name": "all", "unit_key": "all"},
        container=container,
        requested_components=[2, 3, 10],
        valid_components=[2, 3],
    )
    assert rec["n_samples"] == 10
    assert rec["n_features"] == 5
    assert rec["skipped_n_components"] == [10]
