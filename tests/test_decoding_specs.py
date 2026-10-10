from coco_pipe.decoding._specs import (
    ESTIMATOR_SPECS,
    SELECTOR_CAPABILITIES,
    EstimatorCapabilities,
    EstimatorSpec,
    canonical_estimator_name,
)


def test_estimator_capabilities_methods():
    caps = EstimatorCapabilities(
        method="test",
        tasks=("classification",),
        prediction_interfaces=("predict", "predict_proba"),
    )
    # Check supports_task
    assert caps.supports_task("classification") is True
    assert caps.supports_task("regression") is False

    # Check has_response
    assert caps.has_response("predict") is True
    assert caps.has_response("predict_proba") is True
    assert caps.has_response("decision_function") is False

    # Check to_dict
    d = caps.to_dict()
    assert d["method"] == "test"
    assert d["tasks"] == ("classification",)


def test_estimator_spec_to_capabilities_variants():
    # 1. Temporal Sliding
    spec_sliding = ESTIMATOR_SPECS["SlidingEstimator"]
    caps_sliding = spec_sliding.to_capabilities()
    assert caps_sliding.temporal == "sliding"
    assert caps_sliding.input_ranks == ("3d_temporal",)

    # 2. Temporal Generalizing
    spec_gen = ESTIMATOR_SPECS["GeneralizingEstimator"]
    caps_gen = spec_gen.to_capabilities()
    assert caps_gen.temporal == "generalizing"
    assert caps_gen.input_ranks == ("3d_temporal",)

    # 3. Tokens Rank (Simulated if not in registry)
    spec_tokens = EstimatorSpec(
        name="TokenModel",
        import_path="test:TokenModel",
        family="neural",
        task=("classification",),
        input_kinds=("tokens",),
    )
    caps_tokens = spec_tokens.to_capabilities()
    assert caps_tokens.input_ranks == ("tokens",)


def test_canonical_name_mapping():
    # Test common aliases
    assert canonical_estimator_name("lda") == "LinearDiscriminantAnalysis"
    assert canonical_estimator_name("logistic_regression") == "LogisticRegression"
    assert canonical_estimator_name("ridge") == "Ridge"

    # Test unknown passthrough
    assert canonical_estimator_name("CustomModel") == "CustomModel"
    assert canonical_estimator_name("RandomStuff") == "RandomStuff"


def test_selector_capabilities_to_dict():
    cap = SELECTOR_CAPABILITIES["sfs"]
    d = cap.to_dict()
    assert d["method"] == "sfs"
    assert "input_ranks" in d
    assert "support" in d
