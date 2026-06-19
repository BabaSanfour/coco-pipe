import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from coco_pipe.decoding import (
    ClassicalModelConfig,
    CVConfig,
    Experiment,
    ExperimentConfig,
    FeatureSelectionConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    NeuralFineTuneConfig,
    ReducerConfig,
    benjamini_hochberg,
    correct_sweep_pvalues,
)
from coco_pipe.decoding._engine import extract_metadata
from coco_pipe.decoding._splitters import get_cv_splitter
from coco_pipe.decoding.registry import (
    get_foundation_model_spec,
    resolve_estimator_spec,
)
from coco_pipe.decoding.stats import aggregate_predictions_for_inference


def test_luna_registered_and_neural_configs_resolve():
    assert get_foundation_model_spec("luna").display_name == "LUNA"
    neural = resolve_estimator_spec(
        NeuralFineTuneConfig(model_key="luna", train_mode="linear_probe")
    )
    assert neural.supports_proba
    frozen = resolve_estimator_spec(
        FrozenBackboneDecoderConfig(
            backbone=FoundationEmbeddingModelConfig(model_key="cbramod"),
            head=ClassicalModelConfig(
                estimator="LogisticRegression",
                params={"max_iter": 100},
            ),
        )
    )
    assert frozen.input_kinds == ("epoched",)
    assert frozen.supports_proba


def test_fold_local_pca_is_part_of_each_classical_pipeline():
    config = ExperimentConfig(
        models={
            "lr": ClassicalModelConfig(
                estimator="LogisticRegression",
                params={"max_iter": 1000},
            )
        },
        cv=CVConfig(strategy="stratified", n_splits=2),
        reducer=ReducerConfig(enabled=True, n_components=2),
        metrics=["accuracy"],
        n_jobs=1,
    )
    experiment = Experiment(config)
    pipeline = experiment._prepare_estimator("lr", config.models["lr"])
    assert list(pipeline.named_steps) == ["scaler", "reducer", "clf"]
    X = np.random.default_rng(0).normal(size=(20, 5))
    y = np.array([0, 1] * 10)
    result = experiment.run(X, y)
    assert not result.summary().empty


def test_float_pca_clamps_top_k_to_fold_local_width():
    config = ExperimentConfig(
        models={
            "lr": ClassicalModelConfig(
                estimator="LogisticRegression",
                params={"max_iter": 1000},
            )
        },
        cv=CVConfig(strategy="stratified", n_splits=2),
        reducer=ReducerConfig(enabled=True, n_components=0.8),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="k_best",
            n_features=10,
        ),
        metrics=["accuracy"],
        n_jobs=1,
    )
    X = np.random.default_rng(2).normal(size=(20, 5))
    X[:, 1:] = X[:, :1] + 0.01 * X[:, 1:]
    y = np.array([0, 1] * 10)
    result = Experiment(config).run(X, y)
    assert not result.summary().empty
    pipeline = Experiment(config)._prepare_estimator("lr", config.models["lr"])
    fitted = pipeline.fit(X, y)
    selector = fitted.named_steps["fs"]
    assert selector.k == 10
    assert selector.effective_k_ <= 5


def test_grouped_cv_auto_reduces_fold_count():
    y = np.array([0, 0, 1, 1])
    groups = np.array(["c1", "c2", "a1", "a2"])
    splitter = get_cv_splitter(
        CVConfig(strategy="stratified_group_kfold", n_splits=5),
        groups=groups,
        y=y,
    )
    assert splitter.cv.n_splits == 2


def test_sweep_fdr_is_grouped_by_analysis_family():
    adjusted, rejected = benjamini_hochberg([0.001, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert rejected.tolist() == [True, True, False]

    frame = pd.DataFrame(
        {
            "target": ["a", "a", "b"],
            "analysis_mode": ["sensor", "sensor", "sensor"],
            "p_value": [0.01, 0.2, 0.04],
        }
    )
    corrected = correct_sweep_pvalues(frame, family_columns=["target", "analysis_mode"])
    assert corrected.loc[2, "p_value_fdr"] == 0.04


def test_subject_inference_lookup_is_case_insensitive():
    predictions = pd.DataFrame(
        {
            "Subject": ["p1", "p1", "p2", "p2"],
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 1, 1],
            "y_proba_0": [0.8, 0.7, 0.2, 0.1],
            "y_proba_1": [0.2, 0.3, 0.8, 0.9],
        }
    )
    aggregated = aggregate_predictions_for_inference(
        predictions,
        metric="accuracy",
        unit_of_inference="subject",
        custom_aggregation="mean",
    )
    assert aggregated["InferentialUnitID"].tolist() == ["p1", "p2"]


def test_neural_artifacts_are_unwrapped_from_pipeline():
    class ArtifactEstimator(BaseEstimator):
        def get_training_history(self):
            return [{"epoch": 1}]

        def get_checkpoint_manifest(self):
            return {"checkpoint_path": "model.pt"}

        def get_model_card_info(self):
            return {"model_key": "fake"}

        def get_failure_diagnostics(self):
            return {}

        def get_artifact_metadata(self):
            return {"history": self.get_training_history()}

    class Config:
        enabled = False

    metadata = extract_metadata(
        Pipeline([("clf", ArtifactEstimator())]),
        spec=None,
        feature_selection_config=Config(),
    )
    assert metadata["artifacts"]["history"] == [{"epoch": 1}]
