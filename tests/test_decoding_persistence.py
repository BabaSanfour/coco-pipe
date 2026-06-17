import json

import pandas as pd
import pytest

from coco_pipe.decoding.persistence import (
    completed_for_config,
    load_completed_result_records,
    write_run_status,
)


def test_completed_for_config_success(tmp_path):
    (tmp_path / "_SUCCESS").touch()
    manifest = {"config_hash": "dummyhash"}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    from unittest.mock import patch

    with patch("coco_pipe.decoding.persistence.config_hash", return_value="dummyhash"):
        assert completed_for_config(tmp_path, {"a": 1}) is True


def test_completed_for_config_no_success_marker(tmp_path):
    assert completed_for_config(tmp_path, {"a": 1}) is False


def test_completed_for_config_no_manifest(tmp_path):
    (tmp_path / "_SUCCESS").touch()
    with pytest.raises(RuntimeError, match="has no run manifest"):
        completed_for_config(tmp_path, {"a": 1})


def test_completed_for_config_hash_mismatch(tmp_path):
    (tmp_path / "_SUCCESS").touch()
    manifest = {"config_hash": "wronghash"}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    from unittest.mock import patch

    with patch("coco_pipe.decoding.persistence.config_hash", return_value="dummyhash"):
        with pytest.raises(RuntimeError, match="Config hash mismatch"):
            completed_for_config(tmp_path, {"a": 1})


def test_load_completed_result_records_success(tmp_path):
    summary_df = pd.DataFrame([{"Model": "model_A", "accuracy": 0.9}])
    summary_df.to_csv(tmp_path / "summary.csv", index=False)

    stats_df = pd.DataFrame(
        [{"Model": "model_A", "Metric": "accuracy", "PValue": 0.05}]
    )
    stats_df.to_csv(tmp_path / "statistical_assessment.csv", index=False)

    records = load_completed_result_records(tmp_path, {"context": "test"})
    assert len(records) == 1
    assert records[0]["model"] == "model_A"
    assert records[0]["context"] == "test"
    assert records[0]["status"] == "success"
    assert records[0]["p_value"] == 0.05
    assert records[0]["accuracy"] == 0.9


def test_load_completed_result_records_no_summary(tmp_path):
    with pytest.raises(RuntimeError, match="has no summary table"):
        load_completed_result_records(tmp_path, {})


def test_load_completed_result_records_no_model_column(tmp_path):
    pd.DataFrame([{"acc": 0.9}]).to_csv(tmp_path / "summary.csv", index=False)
    with pytest.raises(RuntimeError, match="has no Model column"):
        load_completed_result_records(tmp_path, {})


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
