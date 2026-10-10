import pytest

from coco_pipe.decoding.persistence import (
    completed_for_config,
    config_hash,
    load_completed_result_records,
    write_run_manifest,
    write_run_status,
)


def test_completed_for_config(tmp_path):
    config = {"a": 1}
    assert completed_for_config(tmp_path, config) is False

    write_run_manifest(tmp_path, {"status": "failed", "reason": "boom"})
    assert completed_for_config(tmp_path, config) is False

    write_run_manifest(
        tmp_path, {"status": "success", "config_hash": config_hash(config)}
    )
    assert completed_for_config(tmp_path, config) is True
    with pytest.raises(RuntimeError, match="Config hash mismatch"):
        completed_for_config(tmp_path, {"a": 2})


def test_load_completed_result_records(tmp_path):
    with pytest.raises(RuntimeError, match="no summary to resume from"):
        load_completed_result_records(tmp_path, {})

    write_run_manifest(
        tmp_path,
        {
            "status": "success",
            "summary": [{"Model": "model_A", "accuracy": 0.9}],
            "statistical_assessment": [
                {"Model": "model_A", "Metric": "accuracy", "PValue": 0.05}
            ],
        },
    )
    (record,) = load_completed_result_records(tmp_path, {"context": "test"})
    assert record["model"] == "model_A"
    assert record["context"] == "test"
    assert record["status"] == "success"
    assert record["reason"] == "resumed"
    assert record["p_value"] == 0.05
    assert record["accuracy"] == 0.9


def test_write_run_status(tmp_path):
    marker = write_run_status(tmp_path, "SUCCESS")
    assert marker.exists()
    assert marker.name == "_SUCCESS"

    marker = write_run_status(tmp_path, "PARTIAL")
    assert marker.exists()
    assert marker.name == "_PARTIAL"
    assert not (tmp_path / "_SUCCESS").exists()

    with pytest.raises(ValueError, match="must be SUCCESS, PARTIAL, or FAILED"):
        write_run_status(tmp_path, "UNKNOWN")
