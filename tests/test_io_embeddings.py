import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding.foundation_models import FoundationEmbeddingResult
from coco_pipe.io.embeddings import (
    _json_value,
    combined_embedding_table_path,
    discover_embedding_derivatives,
    embedding_observation_id,
    load_combined_embedding_table,
    load_embedding_derivatives,
    save_embedding_derivative,
    save_embedding_outputs,
    validate_embedding_derivative,
    write_embedding_dataset_description,
    write_embedding_manifest,
)


def _native_token_metadata(**metadata):
    return {
        **metadata,
        "token_layout": "native",
        "token_layout_version": 1,
        "token_source": "test_native_output",
        "token_axes": ["window", "token", "feature"],
        "token_observation_axes": ["token"],
        "token_feature_axis": "feature",
    }


def test_embedding_derivative_round_trip(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        recording_embedding=np.arange(4, dtype=float),
        window_start=np.array([0, 100, 200]),
        window_stop=np.array([100, 200, 300]),
        window_index=np.arange(3),
        metadata={
            "model_key": "cbramod",
            "recording_id": "sub-01_ses-01_run-01",
            "subject": "01",
            "session": "01",
            "patient_group_id": "p01",
            "preprocessing_provenance": {"reference": "average"},
        },
    )
    path = (
        tmp_path
        / "sub-01"
        / "ses-01"
        / "eeg"
        / "sub-01_ses-01_desc-cbramodMean_embedding.npz"
    )
    save_embedding_derivative(result, path)

    metadata = validate_embedding_derivative(path)
    assert metadata["model_key"] == "cbramod"
    assert json.loads(path.with_suffix(".json").read_text())["recording_id"].startswith(
        "sub-01"
    )

    recording = load_embedding_derivatives(tmp_path)
    assert recording.X.shape == (1, 4)
    assert recording.coords["patient_group_id"].tolist() == ["p01"]

    epochs = load_embedding_derivatives(tmp_path, representation="epoch")
    assert epochs.X.shape == (3, 4)
    assert epochs.coords["window_index"].tolist() == [0, 1, 2]
    assert epochs.meta["artifact_metadata"][str(path)] == metadata
    assert epochs.meta["artifact_metadata"][str(path)]["preprocessing_provenance"] == {
        "reference": "average"
    }


def test_token_derivative_round_trip(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        recording_embedding=np.arange(4, dtype=float),
        window_start=np.array([0, 100, 200]),
        window_stop=np.array([100, 200, 300]),
        window_index=np.arange(3),
        metadata=_native_token_metadata(
            model_key="cbramod",
            recording_id="sub-01_ses-01_run-01",
            subject="01",
        ),
        token_embeddings=np.arange(60, dtype=float).reshape(3, 5, 4),
    )
    token_path = tmp_path / "sub-01_desc-cbramod_tokens.npz"
    save_embedding_derivative(result, token_path)

    tokens = load_embedding_derivatives(
        tmp_path, representation="token", model_key="cbramod"
    )
    assert tokens.X.shape == (3, 5, 4)
    assert tokens.dims == ("obs", "token", "feature")
    assert tokens.ids.tolist() == [
        "sub-01_ses-01_run-01_epoch-0000",
        "sub-01_ses-01_run-01_epoch-0001",
        "sub-01_ses-01_run-01_epoch-0002",
    ]
    np.testing.assert_allclose(tokens.X[1], result.token_embeddings[1])

    # A token payload must be saved to a *_tokens.npz path.
    with pytest.raises(ValueError, match="token artifact must be saved"):
        save_embedding_derivative(
            result, tmp_path / "sub-01_desc-cbramod_embedding.npz"
        )


def test_native_four_dimensional_token_derivative_preserves_layout_and_dtype(tmp_path):
    native = np.arange(2 * 3 * 4 * 5, dtype=np.float16).reshape(2, 3, 4, 5)
    result = FoundationEmbeddingResult(
        window_embeddings=np.ones((2, 5)),
        recording_embedding=np.ones(5),
        window_start=np.array([0, 100]),
        window_stop=np.array([100, 200]),
        window_index=np.arange(2),
        metadata={
            "model_key": "reve",
            "recording_id": "sub-01",
            "token_layout": "native",
            "token_layout_version": 1,
            "token_source": "reve_backbone_output",
            "token_axes": ["window", "channel", "time_patch", "feature"],
            "token_observation_axes": ["channel", "time_patch"],
            "token_feature_axis": "feature",
        },
        token_embeddings=native,
    )
    path = tmp_path / "sub-01_desc-reveNative_tokens.npz"

    save_embedding_derivative(result, path)
    loaded = load_embedding_derivatives(path, representation="token", model_key="reve")

    assert loaded.dims == ("obs", "channel", "time_patch", "feature")
    assert loaded.X.dtype == native.dtype
    np.testing.assert_array_equal(loaded.X, native)


def test_embedding_observation_id_includes_condition():
    assert (
        embedding_observation_id(
            {"recording_id": "sub-01_run-01", "condition": "EO"},
            "unrelated_embedding.npz",
            2,
        )
        == "sub-01_run-01_condition-EO_epoch-0002"
    )


def test_save_embedding_outputs_writes_independent_pooled_and_token_artifacts(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        recording_embedding=np.arange(4, dtype=float),
        window_start=np.array([0, 100, 200]),
        window_stop=np.array([100, 200, 300]),
        window_index=np.arange(3),
        metadata=_native_token_metadata(
            model_key="cbramod",
            recording_id="sub-01",
            within_window_pooling="mean",
            recording_pooling="mean",
            normalize_embeddings=True,
        ),
        token_embeddings=np.arange(60, dtype=float).reshape(3, 5, 4),
    )
    embedding_path = tmp_path / "sub-01_desc-cbramod_embedding.npz"
    requested_token_path = tmp_path / "independent_desc-cbramod_tokens.npz"

    pooled_path, token_path = save_embedding_outputs(
        result,
        embedding_path,
        token_path=requested_token_path,
        pooled_metadata={
            "model_key": "cbramod_pool-attention",
            "source_model_key": "cbramod",
        },
    )

    assert pooled_path == embedding_path
    assert token_path == requested_token_path
    token_metadata = validate_embedding_derivative(token_path)
    pooled_metadata = validate_embedding_derivative(pooled_path)
    assert pooled_metadata["model_key"] == "cbramod_pool-attention"
    assert pooled_metadata["source_model_key"] == "cbramod"
    assert token_metadata["model_key"] == "cbramod"
    assert "source_model_key" not in token_metadata
    assert "within_window_pooling" not in token_metadata
    assert "recording_pooling" not in token_metadata
    assert "normalize_embeddings" not in token_metadata
    with np.load(pooled_path, allow_pickle=False) as payload:
        assert "token_embeddings" not in payload.files
    np.testing.assert_allclose(
        load_embedding_derivatives(tmp_path, representation="epoch").X,
        result.window_embeddings,
    )
    np.testing.assert_allclose(
        load_embedding_derivatives(tmp_path, representation="token").X,
        result.token_embeddings,
    )

    # Complete outputs resume, while a partial pair is safely repaired.
    assert save_embedding_outputs(result, embedding_path, token_path=token_path) == (
        pooled_path,
        token_path,
    )
    token_path.with_suffix(".json").unlink()
    assert save_embedding_outputs(result, embedding_path, token_path=token_path) == (
        pooled_path,
        token_path,
    )
    validate_embedding_derivative(token_path)

    with pytest.raises(ValueError, match="token_path is required"):
        save_embedding_outputs(result, tmp_path / "missing-token-path_embedding.npz")


def test_manifest_discovery_partitions_by_kind(tmp_path):
    """A manifest listing both kinds resolves each to the right artifacts."""

    def _make(model_key, subject, *, tokens):
        result = FoundationEmbeddingResult(
            window_embeddings=np.ones((2, 4)),
            recording_embedding=np.ones(4),
            window_start=np.array([0, 100]),
            window_stop=np.array([100, 200]),
            window_index=np.array([0, 1]),
            metadata=(
                _native_token_metadata(
                    model_key=model_key, recording_id=f"sub-{subject}"
                )
                if tokens
                else {"model_key": model_key, "recording_id": f"sub-{subject}"}
            ),
            token_embeddings=np.ones((2, 5, 4)) if tokens else None,
        )
        suffix = "tokens" if tokens else "embedding"
        path = tmp_path / f"sub-{subject}_desc-{model_key}_{suffix}.npz"
        save_embedding_derivative(result, path)
        return path

    emb = _make("cbramod", "01", tokens=False)
    tok = _make("cbramod", "01", tokens=True)
    write_embedding_manifest(
        tmp_path,
        [
            {"status": "success", "artifact_path": emb.name},
            {"status": "success", "artifact_path": tok.name},
        ],
    )

    assert discover_embedding_derivatives(tmp_path, kind="embedding") == [emb]
    assert discover_embedding_derivatives(tmp_path, kind="token") == [tok]


def test_pooled_and_token_fallback_observation_ids_match(tmp_path):
    metadata = {"model_key": "demo"}
    pooled = tmp_path / "sub-01_ses-01_run-01_embedding.npz"
    tokens = tmp_path / "sub-01_ses-01_run-01_tokens.npz"
    assert embedding_observation_id(metadata, pooled, 3) == (
        embedding_observation_id(metadata, tokens, 3)
    )


def test_loader_requires_one_model_space(tmp_path):
    for model_key, subject in (("cbramod", "01"), ("reve", "02")):
        result = FoundationEmbeddingResult(
            window_embeddings=np.ones((1, 4)),
            recording_embedding=np.ones(4),
            window_start=np.array([0]),
            window_stop=np.array([100]),
            window_index=np.array([0]),
            metadata={
                "model_key": model_key,
                "recording_id": f"sub-{subject}_ses-01_run-01",
                "subject": subject,
            },
        )
        save_embedding_derivative(
            result,
            tmp_path / f"sub-{subject}_desc-{model_key}_embedding.npz",
        )

    with pytest.raises(ValueError, match="multiple foundation models"):
        load_embedding_derivatives(tmp_path)

    selected = load_embedding_derivatives(tmp_path, model_key="reve")
    assert selected.X.shape == (1, 4)
    assert selected.meta["model_key"] == "reve"


def test_sidecar_stringifies_unknown_metadata_objects(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.ones((1, 2)),
        recording_embedding=np.ones(2),
        window_start=np.array([0]),
        window_stop=np.array([10]),
        window_index=np.array([0]),
        metadata={"model_key": "cbramod", "opaque": object()},
    )
    path = tmp_path / "sub-01_desc-cbramod_embedding.npz"
    save_embedding_derivative(result, path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["opaque"].startswith("<builtins.object:")


def test_json_value_types():
    @dataclass
    class Dummy:
        a: int

    assert _json_value(Dummy(1)) == {"a": 1}
    assert _json_value(np.array([1, 2])) == [1, 2]
    assert _json_value(np.int64(1)) == 1
    assert _json_value(Path("foo/bar")) == "foo/bar"


def test_validate_embedding_errors(tmp_path):
    path = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        validate_embedding_derivative(path)

    path.touch()
    with pytest.raises(FileNotFoundError):
        validate_embedding_derivative(path)

    sidecar = tmp_path / "missing.json"
    sidecar.write_text("not a dict")

    # Missing arrays
    np.savez(path, dummy=np.array([1]))
    with pytest.raises(ValueError, match="missing arrays"):
        validate_embedding_derivative(path)

    # Bad window_embeddings shape
    np.savez(
        path,
        window_embeddings=np.array([1]),
        recording_embedding=np.array([1]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    with pytest.raises(ValueError, match="window_embeddings must be 2-D"):
        validate_embedding_derivative(path)

    # Bad recording_embedding shape
    np.savez(
        path,
        window_embeddings=np.array([[1]]),
        recording_embedding=np.array([[1]]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    with pytest.raises(ValueError, match="recording_embedding must match"):
        validate_embedding_derivative(path)

    # Bad length
    np.savez(
        path,
        window_embeddings=np.array([[1]]),
        recording_embedding=np.array([1]),
        window_start=[1, 2],
        window_stop=[1],
        window_index=[1],
    )
    with pytest.raises(ValueError, match="length does not match"):
        validate_embedding_derivative(path)

    # Metadata not a dict
    np.savez(
        path,
        window_embeddings=np.array([[1]]),
        recording_embedding=np.array([1]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    sidecar.write_text("[]")
    with pytest.raises(ValueError, match="Expected an object"):
        validate_embedding_derivative(path)


def test_save_embedding_errors(tmp_path):
    class DummyResult:
        window_embeddings = np.array([[1]])
        recording_embedding = np.array([1])
        window_start: ClassVar[list] = [0]
        window_stop: ClassVar[list] = [1]
        window_index: ClassVar[list] = [0]

    with pytest.raises(ValueError, match=r"must end in \.npz"):
        save_embedding_derivative(DummyResult(), tmp_path / "bad.txt")

    # An embedding payload must be saved to an *_embedding.npz path.
    with pytest.raises(ValueError, match="embedding artifact must be saved"):
        save_embedding_derivative(DummyResult(), tmp_path / "test.npz")

    path = tmp_path / "test_embedding.npz"
    save_embedding_derivative(DummyResult(), path)
    with pytest.raises(FileExistsError):
        save_embedding_derivative(DummyResult(), path)


def test_write_manifest_and_discover(tmp_path):
    records = [
        {"status": "success", "artifact_path": "sub-1_embedding.npz"},
        {"status": "success", "artifact_path": "/absolute/path.npz"},
    ]
    write_embedding_manifest(tmp_path, records)

    # Create the referenced files
    (tmp_path / "sub-1_embedding.npz").touch()

    paths = discover_embedding_derivatives(tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "sub-1_embedding.npz"

    # Filter by model key with missing sidecar should ignore it
    paths = discover_embedding_derivatives(tmp_path, model_key="foo")
    assert len(paths) == 0


def test_discover_finds_model_variant_missing_from_manifest(tmp_path):
    raw = tmp_path / "sub-1_embedding.npz"
    aligned = tmp_path / "sub-1_proc-alignleace_embedding.npz"
    raw.touch()
    aligned.touch()
    raw.with_suffix(".json").write_text(json.dumps({"model_key": "demo"}))
    aligned.with_suffix(".json").write_text(
        json.dumps({"model_key": "demo__align-leace"})
    )
    write_embedding_manifest(
        tmp_path,
        [{"status": "success", "artifact_path": str(raw)}],
    )

    assert discover_embedding_derivatives(tmp_path, model_key="demo__align-leace") == [
        aligned
    ]


def test_load_embedding_errors(tmp_path):
    with pytest.raises(ValueError, match="representation must be"):
        load_embedding_derivatives([tmp_path], representation="bad")

    with pytest.raises(FileNotFoundError):
        load_embedding_derivatives([tmp_path / "none.npz"])

    # Valid artifact 1
    path1 = tmp_path / "1_embedding.npz"
    sidecar1 = tmp_path / "1_embedding.json"
    np.savez(
        path1,
        window_embeddings=np.array([[1, 2]]),
        recording_embedding=np.array([1, 2]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    sidecar1.write_text('{"model_key": "M1"}')

    # Valid artifact 2 with wrong dims
    path2 = tmp_path / "2_embedding.npz"
    sidecar2 = tmp_path / "2_embedding.json"
    np.savez(
        path2,
        window_embeddings=np.array([[1]]),
        recording_embedding=np.array([1]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    sidecar2.write_text('{"model_key": "M1"}')

    with pytest.raises(ValueError, match="Embedding shapes differ"):
        load_embedding_derivatives([path1, path2])

    # Same dims but different model_key
    path3 = tmp_path / "3_embedding.npz"
    sidecar3 = tmp_path / "3_embedding.json"
    np.savez(
        path3,
        window_embeddings=np.array([[1, 2]]),
        recording_embedding=np.array([1, 2]),
        window_start=[1],
        window_stop=[1],
        window_index=[1],
    )
    sidecar3.write_text('{"model_key": "M2"}')

    with pytest.raises(
        ValueError, match="multiple foundation models cannot be combined"
    ):
        load_embedding_derivatives([path1, path3])

    with pytest.raises(ValueError, match="Requested model_key="):
        load_embedding_derivatives([path1], model_key="M2")

    # Aggregate by missing coordinate
    with pytest.raises(KeyError):
        load_embedding_derivatives([path1], aggregate_by="missing_coord")


def test_write_dataset_description(tmp_path):
    write_embedding_dataset_description(
        tmp_path, "Test", "1.0", [{"Name": "Agent"}], [{"Name": "Source"}]
    )
    data = json.loads((tmp_path / "dataset_description.json").read_text())
    assert data["Name"] == "Test"
    assert data["SourceDatasets"][0]["Name"] == "Source"


def test_json_value_fallback():
    class CustomObj:
        def __str__(self):
            return "custom"

    assert "<" in _json_value(CustomObj())


def test_discover_embeddings_extra(tmp_path):
    # Manifest with bad status / missing artifact_path
    records = [{"status": "failed"}, {"status": "success"}]  # missing artifact_path
    write_embedding_manifest(tmp_path, records)
    paths = discover_embedding_derivatives(tmp_path)
    assert len(paths) == 0

    # Manifest bad JSON
    (tmp_path / "run_manifest.json").write_text("bad json")
    paths = discover_embedding_derivatives(tmp_path)
    assert len(paths) == 0


def test_load_embeddings_window(tmp_path):
    path1 = tmp_path / "1_embedding.npz"
    sidecar1 = tmp_path / "1_embedding.json"
    np.savez(
        path1,
        window_embeddings=np.array([[1, 2], [3, 4]]),
        recording_embedding=np.array([1, 2]),
        window_start=[1, 2],
        window_stop=[2, 3],
        window_index=[1, 2],
    )
    sidecar1.write_text('{"model_key": "M1"}')

    # Canonical "epoch" representation (one row per epoch).
    ds = load_embedding_derivatives(tmp_path, representation="epoch")
    assert ds.shape == (2, 2)
    assert ds.meta["representation"] == "epoch"

    # The old "window" name is no longer accepted.
    with pytest.raises(ValueError, match=r"epoch.*recording"):
        load_embedding_derivatives(tmp_path, representation="window")


def test_load_embeddings_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embedding_derivatives(tmp_path)


def test_load_embedding_derivatives_aggregate_by(tmp_path):
    result = FoundationEmbeddingResult(
        window_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        recording_embedding=np.arange(4, dtype=float),
        window_start=np.array([0, 100, 200]),
        window_stop=np.array([100, 200, 300]),
        window_index=np.arange(3),
        metadata={"model_key": "cbramod", "recording_id": "sub-01", "subject": "01"},
    )
    path = tmp_path / "sub-01" / "eeg" / "sub-01_desc-x_embedding.npz"
    save_embedding_derivative(result, path)

    # Three epochs, all subject "01" -> aggregating by subject yields one row.
    agg = load_embedding_derivatives(
        tmp_path, representation="epoch", aggregate_by="subject"
    )
    assert agg.X.shape[0] == 1


def _write_combined_table(tmp_path, model_key, condition, representation):
    combined_dir = tmp_path / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "subject": ["sub-0001", "sub-0001", "sub-0002"],
            "recording_id": ["sub-0001_run-01", "sub-0001_run-02", "sub-0002_run-01"],
            "condition": [condition] * 3,
            "model_key": [model_key] * 3,
            "embedding_0000": [0.1, 0.2, 0.3],
            "embedding_0001": [1.0, 1.1, 1.2],
        }
    )
    path = combined_embedding_table_path(tmp_path, model_key, condition, representation)
    frame.to_parquet(path, index=False)
    return path


def test_load_combined_embedding_table_roundtrip(tmp_path):
    _write_combined_table(tmp_path, "cbramod", "EO_baseline", "recording")
    container = load_combined_embedding_table(
        tmp_path, "cbramod", "EO_baseline", "recording"
    )
    assert container.dims == ("obs", "feature")
    assert container.X.shape == (3, 2)
    assert list(container.coords["feature"]) == ["embedding_0000", "embedding_0001"]
    assert "subject" in container.coords
    assert container.meta["model_key"] == "cbramod"
    assert container.meta["source"] == "combined_table"


def test_load_combined_embedding_table_aggregate_by_subject(tmp_path):
    _write_combined_table(tmp_path, "cbramod", "EO_baseline", "recording")
    agg = load_combined_embedding_table(
        tmp_path, "cbramod", "EO_baseline", "recording", aggregate_by="subject"
    )
    assert agg.X.shape[0] == 2  # two unique subjects


def test_load_combined_embedding_table_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_combined_embedding_table(tmp_path, "cbramod", "EO_baseline", "epoch")
