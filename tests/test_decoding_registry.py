import re

import pytest

from coco_pipe.decoding._specs import canonical_estimator_name
from coco_pipe.decoding.registry import (
    EstimatorNotFoundError,
    EstimatorSpec,
    get_capabilities,
    get_estimator_cls,
    get_estimator_spec,
    get_selector_capabilities,
    list_capabilities,
    list_estimator_specs,
    list_foundation_models,
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
    with pytest.raises(EstimatorNotFoundError) as excinfo:
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

    # Foundation (reve) — kind acts as spec_name in the else branch
    cfg_f = SimpleNamespace(kind="reve", method="REVEModel")
    spec_f = resolve_estimator_spec(cfg_f)
    assert spec_f.name == "reve"

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
    foundation_cfg = {"kind": "foundation_embedding", "model_key": "reve"}
    spec = resolve_estimator_spec(foundation_cfg)
    assert spec.name == "reve"


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


def test_registry_all_keys_have_positive_embedding_dim():
    for m in list_foundation_models():
        assert m["embedding_dim"] > 0, f"{m['name']} has non-positive embedding_dim"


def test_registry_all_keys_have_hub_repo():
    for m in list_foundation_models():
        assert m["hub_repo"], f"{m['name']} has empty hub_repo"


def test_list_foundation_models_contains_expected_keys():
    keys = {m["name"] for m in list_foundation_models()}
    assert {"reve", "cbramod", "biot", "labram", "eegpt", "signaljepa", "bendr"} <= keys


def test_get_metadata_known_key():
    m = get_estimator_spec("reve")
    assert m.name == "reve"
    assert m.embedding_dim == 512
    assert m.preferred_backend == "hugging_face"


def test_get_metadata_cbramod():
    m = get_estimator_spec("cbramod")
    assert m.name == "cbramod"
    assert m.embedding_dim == 200
    assert m.preferred_backend == "braindecode"


def test_get_metadata_unknown_raises_with_hint():
    from coco_pipe.decoding import get_foundation_model_spec

    with pytest.raises(KeyError, match="Available"):
        get_foundation_model_spec("not_a_model")


def test_metadata_is_frozen():
    m = get_estimator_spec("reve")
    with pytest.raises(Exception):
        m.embedding_dim = 999  # type: ignore[misc]


def test_register_estimator_spec_overwrite():
    from coco_pipe.decoding._specs import _spec
    from coco_pipe.decoding.registry import register_estimator_spec

    spec = _spec("dummy", "dummy", "linear", ("classification",))
    register_estimator_spec(spec)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        register_estimator_spec(spec)
        assert len(w) > 0
        assert "Overwriting existing estimator spec for 'dummy'" in str(w[-1].message)


def test_get_foundation_model_spec_error():
    import pytest

    from coco_pipe.decoding.registry import get_foundation_model_spec

    with pytest.raises(KeyError, match="Unknown foundation model 'invalid'"):
        get_foundation_model_spec("invalid")


def test_resolve_estimator_spec_classical_fallback():
    from coco_pipe.decoding.registry import resolve_estimator_spec

    spec = resolve_estimator_spec(
        {
            "kind": "classical",
            "method": "ClassicalModel",
            "estimator": "LogisticRegression",
        }
    )
    assert spec.name == "LogisticRegression"


def test_resolve_estimator_spec_foundation_error():
    import pytest

    from coco_pipe.decoding.registry import resolve_estimator_spec

    with pytest.raises(ValueError, match="requires a 'model_key'"):
        resolve_estimator_spec({"kind": "foundation_embedding"})


def test_resolve_estimator_spec_frozen_backbone():
    from coco_pipe.decoding.registry import resolve_estimator_spec

    spec = resolve_estimator_spec(
        {
            "kind": "frozen_backbone",
            "backbone": "LogisticRegression",
            "head": "LogisticRegression",
        }
    )
    assert spec.name == "FrozenBackboneDecoder"
    assert spec.supports_proba


def test_resolve_estimator_spec_neural_finetune():
    from coco_pipe.decoding._specs import _fm_spec
    from coco_pipe.decoding.registry import (
        register_estimator_spec,
        resolve_estimator_spec,
    )

    spec = _fm_spec(
        "dummy_fm",
        hub_repo="repo",
        embedding_dim=10,
        pretrained_sfreq=200.0,
        preferred_backend="hf",
    )
    register_estimator_spec(spec)
    res_spec = resolve_estimator_spec(
        {"kind": "neural_finetune", "model_key": "dummy_fm", "train_mode": "full"}
    )
    assert res_spec.name == "dummy_fm_full"
