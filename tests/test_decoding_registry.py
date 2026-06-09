import re

import pytest

from coco_pipe.decoding._specs import canonical_estimator_name
from coco_pipe.decoding.registry import (
    EstimatorSpec,
    get_capabilities,
    get_estimator_cls,
    get_estimator_spec,
    get_selector_capabilities,
    list_capabilities,
    list_estimator_specs,
    register_estimator,
    register_estimator_spec,
    resolve_estimator_capabilities,
    resolve_estimator_spec,
)


def test_manual_registration():
    @register_estimator("TestModel")
    class TestModel:
        pass

    cls = get_estimator_cls("TestModel")
    assert cls is TestModel


def test_registration_overwrite_warning():
    @register_estimator("WarningModel")
    class Model1:
        pass

    with pytest.warns(UserWarning, match="Overwriting existing estimator registry"):

        @register_estimator("WarningModel")
        class Model2:
            pass


def test_get_estimator_cls_not_found():
    # Direct string check on the exception message
    with pytest.raises(
        pytest.importorskip("coco_pipe.decoding.registry").EstimatorNotFoundError
    ) as excinfo:
        get_estimator_cls("LogisticRegresion")  # Typo
    err_msg = str(excinfo.value)
    assert "Did you mean:" in err_msg
    assert "LogisticRegression" in err_msg


def test_lazy_load_from_spec():
    # LogisticRegression should be loadable from spec even if not in registry dict yet
    cls = get_estimator_cls("LogisticRegression")
    assert cls.__name__ == "LogisticRegression"


def test_capabilities_methods():
    caps = get_capabilities("LogisticRegression")
    assert caps.method == "LogisticRegression"
    assert "classification" in caps.tasks
    assert not caps.supports_task("regression")
    assert caps.has_response("predict")
    assert caps.to_dict()["method"] == "LogisticRegression"

    # Test canonical lookup
    spec = get_estimator_spec("LogisticRegression")
    assert spec.to_dict()["name"] == "LogisticRegression"

    # Test list_capabilities
    all_caps = list_capabilities()
    assert "LogisticRegression" in all_caps


def test_selector_capabilities():
    caps = get_selector_capabilities("k_best")
    assert caps.method == "k_best"
    assert caps.to_dict()["method"] == "k_best"

    with pytest.raises(
        ValueError, match="No decoding capabilities registered for selector"
    ):
        get_selector_capabilities("invalid_selector")


def test_spec_lookup():
    spec = get_estimator_spec("LogisticRegression")
    assert spec.name == "LogisticRegression"
    assert spec.import_path == "sklearn.linear_model"

    all_specs = list_estimator_specs()
    assert "LogisticRegression" in all_specs

    # Test missing spec
    with pytest.raises(ValueError, match="No decoding estimator spec registered"):
        get_estimator_spec("InvalidModel")


def test_register_new_spec():
    new_spec = EstimatorSpec(
        name="NewModel",
        import_path="sklearn.dummy:DummyClassifier",
        family="dummy",
        task=("classification",),
    )
    register_estimator_spec(new_spec)
    cls = get_estimator_cls("NewModel")
    assert cls.__name__ == "DummyClassifier"


def test_invalid_import_path():
    new_spec = EstimatorSpec(
        name="InvalidPath",
        import_path="nonexistent.module",
        family="linear",
        task=("regression",),
    )
    register_estimator_spec(new_spec)
    with pytest.raises(ImportError):
        get_estimator_cls("InvalidPath")


def test_missing_class_in_module():
    new_spec = EstimatorSpec(
        name="MissingClass",
        import_path="sklearn.linear_model:NonExistentClass",
        family="linear",
        task=("regression",),
    )
    register_estimator_spec(new_spec)
    with pytest.raises(Exception):  # EstimatorNotFoundError
        get_estimator_cls("MissingClass")


def test_resolve_estimator_spec():
    from types import SimpleNamespace

    # Classical
    cfg = SimpleNamespace(
        kind="classical", estimator="logistic_regression", method="LogisticRegression"
    )
    spec = resolve_estimator_spec(cfg)
    assert spec.name == "LogisticRegression"

    # Foundation (reve)
    cfg_f = SimpleNamespace(kind="reve", method="REVEModel")
    spec_f = resolve_estimator_spec(cfg_f)
    assert spec_f.name == "REVEModel"

    # Temporal
    cfg_t = SimpleNamespace(
        kind="temporal",
        wrapper="sliding",
        base=cfg,
        method="SlidingEstimator",
        base_estimator=cfg,
    )
    spec_t = resolve_estimator_spec(cfg_t)
    assert spec_t.name == "SlidingEstimator"

    # Canonical
    assert canonical_estimator_name("lda") == "LinearDiscriminantAnalysis"
    assert canonical_estimator_name("unknown") == "unknown"


def test_resolve_estimator_spec_variants():
    from types import SimpleNamespace

    # SVC with probability=False
    cfg_svc = SimpleNamespace(
        method="SVC", kind="classical", estimator="SVC", probability=False
    )
    spec_svc = resolve_estimator_spec(cfg_svc)
    assert not spec_svc.supports_proba

    # SGD with log_loss
    cfg_sgd = SimpleNamespace(
        method="SGDClassifier",
        kind="classical",
        estimator="SGDClassifier",
        loss="log_loss",
    )
    spec_sgd = resolve_estimator_spec(cfg_sgd)
    assert spec_sgd.supports_proba

    # resolve_capabilities
    caps = resolve_estimator_capabilities(cfg_svc)
    assert caps.method == "SVC"


def test_resolve_estimator_spec_temporal_foundation_fixups():
    # 1. Temporal with dict config
    temporal_cfg = {
        "kind": "temporal",
        "wrapper": "sliding",
        "base": {"kind": "classical", "method": "LogisticRegression"},
    }
    spec = resolve_estimator_spec(temporal_cfg)
    assert spec.name == "SlidingEstimator"
    assert spec.supports_proba is True

    # 2. Foundation with dict config
    foundation_cfg = {"kind": "foundation_embedding", "provider": "reve"}
    spec = resolve_estimator_spec(foundation_cfg)
    assert spec.name == "REVEModel"


def test_resolve_estimator_spec_runtime_fixups():
    # SVC probability=False (using dict)
    # Note: registry.py must be updated to use _get_val for this to pass with dicts
    from types import SimpleNamespace

    svc_cfg = SimpleNamespace(kind="classical", method="SVC", probability=False)
    spec = resolve_estimator_spec(svc_cfg)
    assert spec.supports_proba is False
    assert spec.supports_decision_function is True

    # SGDClassifier loss="log_loss"
    sgd_cfg = SimpleNamespace(kind="classical", method="SGDClassifier", loss="log_loss")
    spec = resolve_estimator_spec(sgd_cfg)
    assert spec.supports_proba is True


def test_get_selector_capabilities_error():
    with pytest.raises(
        ValueError, match="No decoding capabilities registered for selector"
    ):
        get_selector_capabilities("non_existent_selector")


def test_register_spec_overwrite_warning():
    spec = EstimatorSpec(
        name="OverwriteModel", import_path="fake", family="linear", task=("regression",)
    )
    register_estimator_spec(spec)
    pass


def test_get_estimator_cls_import_error():
    spec = EstimatorSpec(
        name="ImportErrorModel",
        import_path="non.existent.module:Class",
        family="linear",
        task=("regression",),
    )
    register_estimator_spec(spec)
    with pytest.raises(ImportError, match="Could not load estimator"):
        get_estimator_cls("ImportErrorModel")


def test_get_estimator_cls_not_found_with_matches():
    # Use re.escape to avoid invalid escape sequence warning
    expected = re.escape("Did you mean: ['LogisticRegression']")
    with pytest.raises(Exception, match=expected):
        get_estimator_cls("LogisticRegres")
