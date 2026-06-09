from coco_pipe.report.provenance import (
    get_environment_info,
    get_git_revision_hash,
    get_package_version,
)


def test_get_git_revision_hash():
    h = get_git_revision_hash()
    assert isinstance(h, str)


def test_get_environment_info():
    info = get_environment_info()
    assert isinstance(info, dict)
    assert "timestamp_utc" in info
    assert "os_platform" in info
    assert "python_version" in info
    assert "git_hash" in info
    assert "coco_pipe_version" in info
    assert "versions" in info
    assert "numpy" in info["versions"]
    assert "pandas" in info["versions"]


def test_get_package_version():
    """Verify package version retrieval and fallback."""
    v_np = get_package_version("numpy")
    assert isinstance(v_np, str)
    assert v_np != "Unknown"

    v_fake = get_package_version("non-existent-package-blah-blah")
    assert v_fake == "Unknown"


def test_experiment_provenance_metadata_integration():
    """Verify that decoded results contain the dynamic version."""
    import numpy as np
    import pandas as pd

    from coco_pipe.decoding.configs import (
        ClassicalModelConfig,
        CVConfig,
        ExperimentConfig,
    )
    from coco_pipe.decoding.experiment import Experiment

    config = ExperimentConfig(
        task="classification",
        models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
        metrics=["accuracy"],
        cv=CVConfig(strategy="kfold", n_splits=2),
        tag="test_meta",
    )

    exp = Experiment(config)
    exp._observation_level = "epoch"
    exp._inferential_unit = "sample"
    exp._sample_metadata = pd.DataFrame()

    meta = exp._build_result_meta(np.zeros((10, 5)), None)
    assert "coco_pipe_version" in meta
    assert isinstance(meta["coco_pipe_version"], str)
