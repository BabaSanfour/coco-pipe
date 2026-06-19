import numpy as np
import pytest

from coco_pipe.dim_reduction.artifacts import load_fit_artifact
from coco_pipe.dim_reduction.pipeline import (
    build_auto_pooled_eval_spec,
    build_eval_request,
    build_fit_request,
    prepare_eval_inputs,
    run_eval,
    run_fit,
    valid_component_sweep,
    valid_n_components_for_container,
)
from coco_pipe.io.structures import DataContainer


@pytest.fixture
def dummy_container():
    n_samples = 100
    X = np.random.rand(n_samples, 5)
    coords = {
        "subject": [f"sub{i//2}" for i in range(n_samples)],
        "target": ["A" if i % 2 == 0 else "B" for i in range(n_samples)],
        "group": [f"G{i//10}" for i in range(n_samples)],
    }
    ids = np.arange(n_samples).astype(str)
    return DataContainer(
        X=X, coords=coords, ids=ids, meta={}, dims=("observation", "feature")
    )


def test_run_fit(tmp_path, dummy_container):
    fit_payload = {
        "reducer": "PCA",
        "n_components": 2,
        "scope": "test",
        "condition": "test",
        "unit_key": "test_unit",
    }
    out_path = tmp_path / "fit_out"

    # Run fit
    record = run_fit(fit_payload, dummy_container, out_path, tmp_path, overwrite=True)
    assert record["status"] == "success"
    assert (out_path / "_SUCCESS").exists()

    # Run fit again without overwrite (loads artifact)
    record2 = run_fit(fit_payload, dummy_container, out_path, tmp_path, overwrite=False)
    assert record2["status"] == "success"

    # Run fit again with overwrite
    record3 = run_fit(fit_payload, dummy_container, out_path, tmp_path, overwrite=True)
    assert record3["status"] == "success"


def test_run_fit_errors(tmp_path):
    out_path = tmp_path / "fit_out"
    with pytest.raises(ValueError, match="2D matrix"):
        run_fit(
            {"reducer": "PCA", "n_components": 2},
            DataContainer(X=np.zeros(10), dims=("obs",)),
            out_path,
            tmp_path,
            True,
        )

    with pytest.raises(ValueError, match="ids to be present"):
        run_fit(
            {"reducer": "PCA", "n_components": 2},
            DataContainer(X=np.zeros((10, 2)), dims=("obs", "feat")),
            out_path,
            tmp_path,
            True,
        )
    record = run_fit(
        {"reducer": "PCA", "n_components": 2},
        DataContainer(X=np.zeros(10), dims=("obs",)),
        out_path,
        tmp_path,
        True,
        errors="record",
    )
    assert record["status"] == "failed"


def test_run_eval(tmp_path, dummy_container):
    # Setup fit artifact first
    fit_payload = {
        "fit_id": "test_fit_id",
        "reducer": "PCA",
        "n_components": 2,
        "scope": "test",
        "condition": "test",
        "analysis_mode": "single",
        "unit_type": "all",
        "unit_name": "all",
        "unit_key": "all",
        "input_mode": "raw",
        "representation": "power",
    }
    fit_out = tmp_path / "fit_out"
    run_fit(fit_payload, dummy_container, fit_out, tmp_path, True)

    fit_artifact = load_fit_artifact(fit_out)

    eval_spec = {
        "name": "test_eval",
        "target_col": "target",
        "group_col": "group",
        "filters": [],
        "label_map": {},
    }
    eval_out = tmp_path / "eval_out"

    # Run eval
    record = run_eval(
        fit_artifact,
        dummy_container,
        eval_spec,
        eval_out,
        tmp_path,
        overwrite=True,
    )
    assert record["status"] == "success"
    assert (eval_out / "_SUCCESS").exists()

    # Run eval without overwrite
    record2 = run_eval(
        fit_artifact,
        dummy_container,
        eval_spec,
        eval_out,
        tmp_path,
        overwrite=False,
    )
    assert record2["status"] == "success"

    # run_eval is tolerant of a DataContainer embedding in the fit artifact.
    assert isinstance(fit_artifact["embedding_container"], DataContainer)
    container_artifact = {
        **fit_artifact,
        "embedding": fit_artifact["embedding_container"],
    }
    record3 = run_eval(
        container_artifact,
        dummy_container,
        eval_spec,
        tmp_path / "eval_out_container",
        tmp_path,
        overwrite=True,
    )
    assert record3["status"] == "success"

    bad_record = run_eval(
        fit_artifact,
        dummy_container,
        {**eval_spec, "target_col": "missing_col"},
        tmp_path / "eval_out_failed",
        tmp_path,
        overwrite=True,
        errors="record",
    )
    assert bad_record["status"] == "failed"


def test_build_auto_pooled_eval_spec():
    assert build_auto_pooled_eval_spec(["cond1"], True) is None
    assert build_auto_pooled_eval_spec(["cond1", "cond2"], False) is None
    spec = build_auto_pooled_eval_spec(["cond1", "cond2"], True)
    assert spec is not None
    assert spec["name"] == "condition_separation"


def testprepare_eval_inputs(dummy_container):
    fit_ids = dummy_container.ids.copy()
    eval_spec = {
        "name": "test",
        "target_col": "target",
        "group_col": "group",
        "filters": [{"column": "subject", "values": ["sub1"]}],
        "label_map": {"A": "Class_A"},
    }
    idx, sel_ids, labels, groups = prepare_eval_inputs(
        dummy_container, fit_ids, eval_spec
    )
    assert len(sel_ids) == 2
    assert "Class_A" in labels

    # Test errors
    with pytest.raises(ValueError, match="missing_col"):
        prepare_eval_inputs(
            dummy_container,
            fit_ids,
            {**eval_spec, "filters": [{"column": "missing_col", "values": ["1"]}]},
        )

    with pytest.raises(RuntimeError, match="could not be aligned"):
        prepare_eval_inputs(dummy_container, np.array(["missing_id"]), eval_spec)


def test_valid_component_sweep(dummy_container):
    assert valid_n_components_for_container(dummy_container, 2) is True
    assert (
        valid_n_components_for_container(dummy_container, 10) is False
    )  # X is 10x5, max is 5
    assert (
        valid_n_components_for_container(
            DataContainer(X=np.zeros(10), dims=("obs",)), 2
        )
        is False
    )
    assert (
        valid_n_components_for_container(
            DataContainer(X=np.zeros(10), dims=("obs",)), 2
        )
        is False
    )

    valid = valid_component_sweep(dummy_container, [2, 10])
    assert valid == [2]


def test_build_fit_request(tmp_path, dummy_container):
    unit_spec = {
        "unit_type": "all",
        "unit_name": "all",
        "unit_key": "all",
        "container": dummy_container,
    }
    input_signature = {
        "input_mode": "raw",
        "representation": "power",
        "analysis_mode": "single",
    }

    request = build_fit_request(
        container=dummy_container,
        scope="test",
        condition="test",
        unit_spec=unit_spec,
        reducer="PCA",
        n_components=2,
        input_signature=input_signature,
        output_root=tmp_path,
        overwrite=True,
        subject_col="subject",
    )
    assert request["out_path"].parent == tmp_path / "artifacts" / "fits"
    assert request["fit_payload"]["input_mode"] == "raw"
    assert request["fit_payload"]["container_signature"]["matrix_shape"] == [100, 5]
    res = run_fit(**request)
    assert res["status"] == "success"

    request["fit_payload"]["n_components"] = -1
    res2 = run_fit(**request, errors="record")
    assert res2["status"] == "failed"


def test_build_fit_request_identity_tracks_container_and_unit(
    tmp_path, dummy_container
):
    input_signature = {
        "input_mode": "descriptors",
        "representation": "features",
        "analysis_mode": "descriptor_sensor",
    }

    def fit_id(container, unit_key, subfamily=None):
        request = build_fit_request(
            container=container,
            scope="condition",
            condition="EO",
            unit_spec={
                "unit_type": "descriptor",
                "unit_name": "alpha",
                "unit_key": unit_key,
                "family": "band",
                "subfamily": subfamily,
                "container": container,
            },
            reducer="PCA",
            n_components=2,
            input_signature=input_signature,
            output_root=tmp_path,
        )
        return request["fit_payload"]["fit_id"]

    changed = DataContainer(
        X=np.asarray(dummy_container.X, dtype=float) + 1.0,
        dims=dummy_container.dims,
        coords=dummy_container.coords,
        ids=dummy_container.ids,
    )
    baseline = fit_id(dummy_container, "alpha_Fz")
    assert baseline != fit_id(changed, "alpha_Fz")
    assert baseline != fit_id(dummy_container, "alpha_Cz")
    assert baseline != fit_id(dummy_container, "alpha_Fz", "log_abs")


def test_build_eval_request(tmp_path, dummy_container):
    unit_spec = {
        "unit_type": "all",
        "unit_name": "all",
        "unit_key": "all",
        "container": dummy_container,
    }
    fit_request = build_fit_request(
        container=dummy_container,
        scope="test",
        condition="test",
        unit_spec=unit_spec,
        reducer="PCA",
        n_components=2,
        input_signature={
            "input_mode": "raw",
            "representation": "power",
            "analysis_mode": "single",
        },
        output_root=tmp_path,
        overwrite=True,
        subject_col="subject",
    )
    fit_res = run_fit(**fit_request)
    fit_artifact = load_fit_artifact(fit_request["out_path"])

    eval_spec = {
        "name": "test_eval",
        "target_col": "target",
        "group_col": "group",
        "filters": [],
        "label_map": {},
    }

    request = build_eval_request(
        fit_record=fit_res,
        fit_artifact=fit_artifact,
        eval_spec=eval_spec,
        container=dummy_container,
        output_root=tmp_path,
        overwrite=True,
    )
    assert request["out_path"].parent == tmp_path / "artifacts" / "evals"
    res = run_eval(**request)
    assert res["status"] == "success"

    request["eval_spec"] = {**eval_spec, "target_col": "missing_col"}
    request["out_path"] = tmp_path / "eval_out_failed"
    res2 = run_eval(**request, errors="record")
    assert res2["status"] == "failed"

    descriptor_request = build_fit_request(
        container=dummy_container,
        scope="test",
        condition="test",
        unit_spec=unit_spec,
        reducer="PCA",
        n_components=2,
        input_signature={
            "input_mode": "descriptors",
            "representation": "features",
            "analysis_mode": "single",
            "descriptor_table_path": "a",
            "descriptor_feature_columns_path": "b",
        },
        output_root=tmp_path,
        overwrite=True,
        subject_col="subject",
    )
    assert descriptor_request["fit_payload"]["input_mode"] == "descriptors"


def test_prepare_eval_inputs_no_ids():
    with pytest.raises(ValueError, match="ids to be present"):
        prepare_eval_inputs(
            DataContainer(X=np.zeros((5, 5)), dims=("obs", "feat")), np.array([]), {}
        )


def test_coco_pipe_lazy_getattr():
    import coco_pipe

    # UMAPReducer should be lazily resolved
    assert coco_pipe.UMAPReducer is not None

    with pytest.raises(
        AttributeError, match="module 'coco_pipe' has no attribute 'InvalidAttr'"
    ):
        _ = coco_pipe.InvalidAttr
