import numpy as np
import pytest

from coco_pipe.dim_reduction.pipeline import (
    _build_eval_task,
    _build_fit_task,
    _execute_eval_task,
    _execute_fit_task,
    _prepare_eval_inputs,
    _valid_component_sweep,
    _valid_n_components_for_container,
    build_auto_pooled_eval_spec,
    run_eval,
    run_fit,
)
from coco_pipe.io.structures import DataContainer


class MockArgs:
    def __init__(self, **kwargs):
        self.input_mode = "raw"
        self.representation = "power"
        self.analysis_mode = "single"
        self.balance_target = False
        self.bids_root = "/tmp/bids"
        self.use_derivatives = False
        self.segment_duration = 1.0
        self.overlap = 0.0
        self.desc = None
        self.filter_col = []
        self.filter_val = []
        self.overwrite = True
        self.subject_col = "subject"
        for k, v in kwargs.items():
            setattr(self, k, v)


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

    from coco_pipe.dim_reduction.artifacts import load_fit_artifact

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
        fit_payload,
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
        fit_payload,
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
        fit_payload,
        container_artifact,
        dummy_container,
        eval_spec,
        tmp_path / "eval_out_container",
        tmp_path,
        overwrite=True,
    )
    assert record3["status"] == "success"


def test_build_auto_pooled_eval_spec():
    assert build_auto_pooled_eval_spec(["cond1"], True) is None
    assert build_auto_pooled_eval_spec(["cond1", "cond2"], False) is None
    spec = build_auto_pooled_eval_spec(["cond1", "cond2"], True)
    assert spec is not None
    assert spec["name"] == "condition_separation"


def test_prepare_eval_inputs(dummy_container):
    fit_ids = dummy_container.ids.copy()
    eval_spec = {
        "name": "test",
        "target_col": "target",
        "group_col": "group",
        "filters": [{"column": "subject", "values": ["sub1"]}],
        "label_map": {"A": "Class_A"},
    }
    idx, sel_ids, labels, groups = _prepare_eval_inputs(
        dummy_container, fit_ids, eval_spec
    )
    assert len(sel_ids) == 2
    assert "Class_A" in labels

    # Test errors
    with pytest.raises(ValueError, match="missing_col"):
        _prepare_eval_inputs(
            dummy_container,
            fit_ids,
            {**eval_spec, "filters": [{"column": "missing_col", "values": ["1"]}]},
        )

    with pytest.raises(RuntimeError, match="could not be aligned"):
        _prepare_eval_inputs(dummy_container, np.array(["missing_id"]), eval_spec)


def test_valid_component_sweep(dummy_container):
    assert _valid_n_components_for_container(dummy_container, 2) is True
    assert (
        _valid_n_components_for_container(dummy_container, 10) is False
    )  # X is 10x5, max is 5
    assert (
        _valid_n_components_for_container(
            DataContainer(X=np.zeros(10), dims=("obs",)), 2
        )
        is False
    )
    assert (
        _valid_n_components_for_container(
            DataContainer(X=np.zeros(10), dims=("obs",)), 2
        )
        is False
    )

    valid = _valid_component_sweep(dummy_container, [2, 10])
    assert valid == [2]


def test_execute_fit_task(tmp_path, dummy_container):
    args = MockArgs()
    unit_spec = {
        "unit_type": "all",
        "unit_name": "all",
        "unit_key": "all",
        "container": dummy_container,
    }

    task = _build_fit_task(args, "test", "test", unit_spec, "PCA", 2, tmp_path)
    res = _execute_fit_task(task)
    assert res["status"] == "success"

    # test failure
    task["fit_payload"]["n_components"] = -1
    res2 = _execute_fit_task(task)
    assert res2["status"] == "failed"


def test_execute_eval_task(tmp_path, dummy_container):
    args = MockArgs()
    unit_spec = {
        "unit_type": "all",
        "unit_name": "all",
        "unit_key": "all",
        "container": dummy_container,
    }
    fit_task = _build_fit_task(args, "test", "test", unit_spec, "PCA", 2, tmp_path)
    fit_res = _execute_fit_task(fit_task)

    eval_spec = {
        "name": "test_eval",
        "target_col": "target",
        "group_col": "group",
        "filters": [],
        "label_map": {},
    }

    task = _build_eval_task(fit_res, eval_spec, dummy_container, tmp_path, True)
    res = _execute_eval_task(task)
    assert res["status"] == "success"

    # Test failure
    task["eval_spec"]["target_col"] = "missing_col"
    res2 = _execute_eval_task(task)
    assert res2["status"] == "failed"

    # Test with descriptor mode args
    args_desc = MockArgs(
        input_mode="descriptors",
        descriptor_table_path="a",
        descriptor_feature_columns_path="b",
    )
    task_desc = _build_fit_task(
        args_desc, "test", "test", unit_spec, "PCA", 2, tmp_path
    )
    assert task_desc["fit_payload"]["input_mode"] == "descriptors"


def test_prepare_eval_inputs_no_ids():
    with pytest.raises(ValueError, match="ids to be present"):
        _prepare_eval_inputs(
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
