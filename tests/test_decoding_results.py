import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentResult
from coco_pipe.decoding._cache import make_feature_cache_key
from coco_pipe.decoding._constants import RESULT_SCHEMA_VERSION
from coco_pipe.decoding.configs import (
    CVConfig,
    ExperimentConfig,
    LogisticRegressionConfig,
)


def _classification_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.8],
            [-1.0, -0.6],
            [-0.8, -0.4],
            [0.8, 0.4],
            [1.0, 0.6],
            [1.5, 0.8],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _config(output_dir=None):
    return ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200)},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=0),
        output_dir=output_dir,
        tag="result_schema_test",
        n_jobs=1,
        verbose=False,
    )


def test_run_result_payload_stores_config_provenance_sample_ids_and_groups():
    X, y = _classification_data()
    sample_ids = np.array([f"sample_{idx}" for idx in range(len(y))])
    groups = np.array(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"])
    sample_metadata = {
        "subject": ["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"],
        "session": ["visit1"] * len(y),
        "site": ["site1"] * len(y),
    }

    result = Experiment(_config()).run(
        X,
        y,
        groups=groups,
        sample_ids=sample_ids,
        sample_metadata=sample_metadata,
        observation_level="epoch",
        feature_names=["left", "right"],
    )

    payload = result.to_payload()
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["config"]["tag"] == "result_schema_test"
    assert payload["meta"]["task"] == "classification"
    assert payload["meta"]["n_samples"] == len(y)
    assert payload["meta"]["n_features"] == X.shape[1]
    assert payload["meta"]["observation_level"] == "epoch"
    assert payload["meta"]["inferential_unit"] == "subject"
    assert payload["meta"]["sample_metadata_columns"] == [
        "Subject",
        "Session",
        "Site",
    ]
    assert "versions" in payload["meta"]

    predictions = result.get_predictions()
    assert {
        "SampleIndex",
        "SampleID",
        "Group",
        "Subject",
        "Session",
        "Site",
    }.issubset(predictions.columns)
    assert set(predictions["SampleID"]) == set(sample_ids)
    assert set(predictions["Group"]) == set(groups)
    assert set(predictions["Subject"]) == {"s0", "s1", "s2", "s3"}
    assert set(predictions["Session"]) == {"visit1"}

    splits = result.get_splits()
    assert set(splits["Set"]) == {"train", "test"}
    assert set(splits["SampleID"]) == set(sample_ids)
    assert set(splits["Group"]) == set(groups)
    assert {"Subject", "Session", "Site"}.issubset(splits.columns)

    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=10,
        random_state=0,
    )
    assert set(ci["Unit"]) == {"subject"}


def test_duplicate_sample_ids_are_rejected():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(X, y, sample_ids=["s0"] * len(y))
    except ValueError as exc:
        assert "sample_ids must be unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate sample_ids to fail.")


def test_observation_level_is_explicitly_limited():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(X, y, observation_level="trial")
    except ValueError as exc:
        assert "observation_level must be 'sample' or 'epoch'" in str(exc)
    else:
        raise AssertionError("Expected invalid observation_level to fail.")


def test_sample_metadata_requires_subject_and_session():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(
            X,
            y,
            sample_metadata={"subject": [f"s{idx}" for idx in range(len(y))]},
        )
    except ValueError as exc:
        assert "sample_metadata must include Subject and Session" in str(exc)
    else:
        raise AssertionError("Expected incomplete sample_metadata to fail.")


def test_explicit_inferential_unit_overrides_epoch_default():
    X, y = _classification_data()
    metadata = {
        "subject": [f"s{idx // 2}" for idx in range(len(y))],
        "session": ["visit1"] * len(y),
    }

    result = Experiment(_config()).run(
        X,
        y,
        sample_metadata=metadata,
        observation_level="epoch",
        inferential_unit="epoch",
    )

    assert result.meta["inferential_unit"] == "epoch"
    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=10,
        random_state=0,
    )
    assert set(ci["Unit"]) == {"epoch"}


def test_save_load_roundtrip_preserves_decoding_payload(tmp_path):
    X, y = _classification_data()
    exp = Experiment(_config(output_dir=tmp_path))
    result = exp.run(X, y, sample_ids=[f"s{idx}" for idx in range(len(y))])

    path = result.save()
    loaded = ExperimentResult.load(path)

    assert loaded.schema_version == result.schema_version
    assert loaded.config["tag"] == result.config["tag"]
    assert loaded.meta["n_samples"] == result.meta["n_samples"]
    assert loaded.raw.keys() == result.raw.keys()
    assert loaded.to_payload()["schema_version"] == RESULT_SCHEMA_VERSION


def test_get_predictions_expands_temporal_arrays():
    raw = {
        "sliding": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.array([0, 1]),
                    "sample_id": np.array(["s0", "s1"]),
                    "group": np.array(["g0", "g1"]),
                    "y_true": np.array([0, 1]),
                    "y_pred": np.array([[0, 1], [1, 1]]),
                    "y_proba": np.array(
                        [
                            [[0.8, 0.4], [0.2, 0.6]],
                            [[0.3, 0.2], [0.7, 0.8]],
                        ]
                    ),
                }
            ],
        },
        "generalizing": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.array([0, 1]),
                    "sample_id": np.array(["s0", "s1"]),
                    "group": None,
                    "y_true": np.array([0, 1]),
                    "y_pred": np.array(
                        [
                            [[0, 1], [1, 0]],
                            [[1, 1], [0, 1]],
                        ]
                    ),
                    "y_proba": np.ones((2, 2, 2, 2)) * 0.5,
                }
            ],
        },
    }

    predictions = ExperimentResult(raw).get_predictions()

    sliding = predictions[predictions["Model"] == "sliding"]
    assert len(sliding) == 4
    assert set(sliding["Time"]) == {0, 1}
    assert {"y_proba_0", "y_proba_1"}.issubset(sliding.columns)

    generalizing = predictions[predictions["Model"] == "generalizing"]
    assert len(generalizing) == 8
    assert set(generalizing["TrainTime"]) == {0, 1}
    assert set(generalizing["TestTime"]) == {0, 1}
    assert generalizing["Group"].isna().all()


def test_get_detailed_scores_expands_temporal_scores():
    result = ExperimentResult(
        {
            "model": {
                "metrics": {
                    "accuracy": {
                        "mean": 0.5,
                        "std": 0.1,
                        "folds": [
                            0.75,
                            np.array([0.1, 0.2]),
                            np.array([[0.1, 0.2], [0.3, 0.4]]),
                        ],
                    }
                }
            }
        }
    )

    scores = result.get_detailed_scores()

    assert len(scores[scores["Fold"] == 0]) == 1
    assert set(scores[scores["Fold"] == 1]["Time"]) == {0, 1}
    matrix_scores = scores[scores["Fold"] == 2]
    assert len(matrix_scores) == 4
    assert set(matrix_scores["TrainTime"]) == {0, 1}
    assert set(matrix_scores["TestTime"]) == {0, 1}


def test_get_feature_importances_returns_named_aggregate_and_fold_tables():
    result = ExperimentResult(
        {
            "rf": {
                "metrics": {},
                "importances": {
                    "mean": np.array([0.25, 0.75]),
                    "std": np.array([0.05, 0.10]),
                    "raw": np.array([[0.2, 0.8], [0.3, 0.7]]),
                    "feature_names": ["alpha", "beta"],
                },
                "metadata": [{}],
            }
        }
    )

    aggregate = result.get_feature_importances()
    assert aggregate.columns.tolist() == [
        "Model",
        "Feature",
        "FeatureName",
        "Mean",
        "Std",
        "Rank",
    ]
    assert aggregate["FeatureName"].tolist() == ["alpha", "beta"]
    assert aggregate["Mean"].tolist() == [0.25, 0.75]

    fold_level = result.get_feature_importances(fold_level=True)
    assert fold_level.columns.tolist() == [
        "Model",
        "Fold",
        "Feature",
        "FeatureName",
        "Importance",
        "Rank",
    ]
    assert len(fold_level) == 4
    assert set(fold_level["Fold"]) == {0, 1}


def test_feature_cache_key_tracks_split_preprocessing_and_backbone_identity():
    base = make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )

    assert base == make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0"],
        test_sample_ids=["s1", "s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-b",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-b",
    )


def _diagnostic_result():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200, kind="classical")},
        metrics=["accuracy", "roc_auc", "brier_score"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=7),
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config).run(X, y)


def test_fit_diagnostics_are_recorded_per_fold():
    result = _diagnostic_result()
    diagnostics = result.get_fit_diagnostics()
    assert len(diagnostics) >= 2
    assert len(pd.unique(diagnostics["Fold"])) == 2
    assert {"FitTime", "PredictTime", "ScoreTime", "TotalTime"}.issubset(
        diagnostics.columns
    )
    assert (diagnostics["FitTime"] >= 0).all()


def test_fit_diagnostics_expands_warning_records():
    result = ExperimentResult(
        {
            "model": {
                "metrics": {},
                "predictions": [],
                "diagnostics": [
                    {
                        "fit_time": 0.1,
                        "predict_time": 0.2,
                        "score_time": 0.3,
                        "total_time": 0.6,
                        "warnings": [
                            {
                                "stage": "fit",
                                "category": "ConvergenceWarning",
                                "message": "did not converge",
                            }
                        ],
                    }
                ],
            }
        }
    )
    diagnostics = result.get_fit_diagnostics()
    assert diagnostics.loc[0, "Stage"] == "fit"
    assert diagnostics.loc[0, "WarningCategory"] == "ConvergenceWarning"
    assert "did not converge" in diagnostics.loc[0, "WarningMessage"]


def test_confusion_roc_pr_and_calibration_accessors():
    result = _diagnostic_result()
    confusion = result.get_confusion_matrices()
    counts = result.get_confusion_counts()
    pooled = result.get_pooled_confusion_matrix()
    roc = result.get_roc_curve()
    pr = result.get_pr_curve()
    calibration = result.get_calibration_curve(n_bins=3)
    proba = result.get_probability_diagnostics()

    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(confusion.columns)
    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(counts.columns)
    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(pooled.columns)
    assert {"FPR", "TPR", "Threshold"}.issubset(roc.columns)
    assert {"Precision", "Recall", "Threshold"}.issubset(pr.columns)
    assert {
        "MeanPredictedProbability",
        "FractionPositive",
    }.issubset(calibration.columns)
    assert {"Metric", "Class", "Value"}.issubset(proba.columns)
    assert not confusion.empty
    assert not counts.empty
    assert not pooled.empty
    assert not roc.empty
    assert not pr.empty
    assert not calibration.empty
    assert {"log_loss", "brier_score_macro"}.issubset(set(proba["Metric"]))


def test_multiclass_curves_use_one_vs_rest_rows():
    raw = {
        "m": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.arange(6),
                    "sample_id": np.arange(6),
                    "group": None,
                    "y_true": np.array([0, 1, 2, 0, 1, 2]),
                    "y_pred": np.array([0, 1, 2, 1, 1, 0]),
                    "y_proba": np.array(
                        [
                            [0.8, 0.1, 0.1],
                            [0.1, 0.7, 0.2],
                            [0.1, 0.2, 0.7],
                            [0.3, 0.5, 0.2],
                            [0.2, 0.6, 0.2],
                            [0.4, 0.2, 0.4],
                        ]
                    ),
                }
            ],
        }
    }
    result = ExperimentResult(raw)
    assert set(result.get_roc_curve()["Class"]) == {0, 1, 2}
    assert set(result.get_pr_curve()["Class"]) == {0, 1, 2}
    assert set(result.get_calibration_curve(n_bins=2)["Class"]) == {0, 1, 2}
    assert not result.get_probability_diagnostics().empty


def test_permutation_bootstrap_and_paired_comparison_helpers():
    X, y = make_classification(
        n_samples=36,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=9,
    )
    groups = np.repeat(np.arange(12), 3)
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": LogisticRegressionConfig(max_iter=200, kind="classical"),
            "dummy": {
                "kind": "classical",
                "method": "DummyClassifier",
                "strategy": "prior",
            },
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=9),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y, groups=groups)

    null = result.get_statistical_assessment(
        lightweight=True, n_permutations=20, random_state=1
    )
    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=20,
        unit="group",
        random_state=1,
    )
    paired = result.compare_models_paired(
        "lr",
        "dummy",
        n_permutations=20,
        unit="group",
        random_state=1,
    )

    assert {"Observed", "NullLower", "NullUpper", "PValue"}.issubset(null.columns)
    assert {"Estimate", "CILower", "CIUpper", "NUnits"}.issubset(ci.columns)
    assert paired.loc[0, "ModelA"] == "lr"
    assert paired.loc[0, "ModelB"] == "dummy"
    assert 0 <= paired.loc[0, "PValue"] <= 1


def test_make_serializable_all_types():
    from coco_pipe.decoding.result import make_serializable

    data = {
        "arr": np.array([1, 2, 3]),
        "int": np.int64(42),
        "float": np.float32(3.14),
        "bool": np.bool_(True),
        "nested": {
            "list": [np.int32(1), np.float64(2.0)],
            "tuple": (np.int16(3),),
        },
    }

    serialized = make_serializable(data)

    assert isinstance(serialized["arr"], list)
    assert isinstance(serialized["int"], int)
    assert isinstance(serialized["float"], float)
    assert isinstance(serialized["bool"], bool)
    assert isinstance(serialized["nested"]["list"][0], int)
    assert isinstance(serialized["nested"]["list"][1], float)
    assert isinstance(serialized["nested"]["tuple"][0], int)
    # Check JSON compatibility
    import json

    json.dumps(serialized)


def test_save_load_json_roundtrip(tmp_path):
    X, y = _classification_data()
    result = Experiment(_config(output_dir=tmp_path)).run(X, y)

    json_path = tmp_path / "result.json"
    saved_path = result.save(json_path)

    assert saved_path == json_path
    assert saved_path.exists()

    loaded = ExperimentResult.load(json_path)
    assert loaded.config["tag"] == result.config["tag"]
    assert loaded.raw.keys() == result.raw.keys()

    # Verify it was indeed JSON
    with open(json_path, "r") as f:
        import json

        data = json.load(f)
        assert data["schema_version"] == RESULT_SCHEMA_VERSION


def test_get_predictions_with_model_filter():
    raw = {
        "m1": {
            "metrics": {},
            "predictions": [
                {"sample_index": [0], "sample_id": ["s0"], "y_true": [0], "y_pred": [0]}
            ],
        },
        "m2": {
            "metrics": {},
            "predictions": [
                {"sample_index": [1], "sample_id": ["s1"], "y_true": [1], "y_pred": [1]}
            ],
        },
    }
    result = ExperimentResult(raw)

    preds_m1 = result.get_predictions(model="m1")
    assert (preds_m1["Model"] == "m1").all()
    assert len(preds_m1) == 1

    preds_all = result.get_predictions()
    assert set(preds_all["Model"]) == {"m1", "m2"}


def test_get_temporal_score_summary_1d_2d():
    raw = {
        "m1": {
            "metrics": {
                "acc": {
                    "folds": [np.array([0.5, 0.6]), np.array([0.7, 0.8])],
                }
            },
            "statistical_assessment": [
                {"Metric": "acc", "Time": 0.0, "PValue": 0.01, "Significant": True},
                {"Metric": "acc", "Time": 1.0, "PValue": 0.05, "Significant": False},
            ],
        },
        "m2": {
            "metrics": {
                "acc": {
                    "folds": [np.array([[0.5, 0.6], [0.7, 0.8]])],
                }
            }
        },
    }
    result = ExperimentResult(raw, time_axis=[0.0, 1.0])

    summary = result.get_temporal_score_summary()

    # 1D check
    m1_sum = summary[summary["Model"] == "m1"]
    assert len(m1_sum) == 2
    assert m1_sum.iloc[0]["Mean"] == 0.6  # (0.5 + 0.7) / 2
    assert m1_sum.iloc[0]["PValue"] == 0.01
    assert m1_sum.iloc[0]["Significant"]

    # 2D check
    m2_sum = summary[summary["Model"] == "m2"]
    assert len(m2_sum) == 4
    assert set(m2_sum["TrainTime"]) == {0.0, 1.0}
    assert set(m2_sum["TestTime"]) == {0.0, 1.0}


def test_get_splits_with_metadata_flattening():
    raw = {
        "m": {
            "splits": [
                {
                    "train_idx": [0],
                    "train_sample_id": ["s0"],
                    "train_metadata": {"sub": ["sub1"], "site": ["site1"]},
                    "test_idx": [1],
                    "test_sample_id": ["s1"],
                    "test_metadata": {"sub": ["sub2"], "site": ["site1"]},
                }
            ]
        }
    }
    result = ExperimentResult(raw)
    splits = result.get_splits()

    assert "sub" in splits.columns
    assert "site" in splits.columns
    assert splits.loc[splits["SampleID"] == "s0", "sub"].iloc[0] == "sub1"
    assert splits.loc[splits["SampleID"] == "s1", "sub"].iloc[0] == "sub2"


def test_summary_with_statistical_rows():
    raw = {
        "m": {
            "metrics": {"acc": {"mean": 0.8, "std": 0.05}},
            "statistical_assessment": [
                {
                    "Metric": "acc",
                    "Time": None,
                    "TrainTime": None,
                    "PValue": 0.001,
                    "Significant": True,
                }
            ],
        }
    }
    result = ExperimentResult(raw)
    summ = result.summary()

    assert summ.loc["m", "acc_mean"] == 0.8
    assert summ.loc["m", "acc_p_val"] == 0.001
    assert summ.loc["m", "acc_sig"] == "*"


def test_get_roc_pr_auc_summaries_multiclass():
    # Setup multiclass probas
    y_true = np.array([0, 0, 1, 1, 2, 2])
    # Model 1: perfect
    y_proba = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    )

    raw = {
        "m": {
            "predictions": [
                {
                    "sample_index": np.arange(6),
                    "sample_id": np.arange(6),
                    "y_true": y_true,
                    "y_pred": y_true,
                    "y_proba": y_proba,
                }
            ]
        }
    }
    result = ExperimentResult(raw)

    roc_auc = result.get_roc_auc_summary()
    pr_auc = result.get_pr_auc_summary()

    assert roc_auc.loc[0, "MacroROCAUC"] == 1.0
    assert pr_auc.loc[0, "MacroPRAUC"] == 1.0


def test_get_generalization_matrix_formats():
    raw = {
        "m": {"metrics": {"accuracy": {"folds": [np.array([[0.5, 0.6], [0.7, 0.8]])]}}}
    }
    result = ExperimentResult(raw, time_axis=["t1", "t2"])

    # Wide format (model specified)
    wide = result.get_generalization_matrix(model="m")
    assert wide.shape == (2, 2)
    assert wide.index.name == "TrainTime"
    assert list(wide.index) == ["t1", "t2"]

    # Long format (no model specified)
    long = result.get_generalization_matrix()
    assert len(long) == 4
    assert set(long["TrainTime"]) == {"t1", "t2"}
    assert "Value" in long.columns


def test_get_best_params_and_search_results():
    raw = {
        "m": {
            "metadata": [
                {
                    "best_params": {"C": 1.0, "penalty": "l2"},
                    "search_results": [
                        {
                            "candidate": 0,
                            "rank_test_score": 1,
                            "mean_test_score": 0.8,
                            "params": {"C": 1.0},
                        },
                        {
                            "candidate": 1,
                            "rank_test_score": 2,
                            "mean_test_score": 0.7,
                            "params": {"C": 0.1},
                        },
                    ],
                }
            ]
        }
    }
    result = ExperimentResult(raw)

    best = result.get_best_params()
    assert len(best) == 2
    assert set(best["Param"]) == {"C", "penalty"}

    search = result.get_search_results()
    assert len(search) == 2
    assert search.loc[0, "Rank"] == 1


def test_get_selected_features_with_order_and_stability():
    raw = {
        "m": {
            "metadata": [
                {
                    "selected_features": [True, False, True],
                    "selection_order": [2, 0],  # feature 2 first, then feature 0
                    "feature_names": ["f1", "f2", "f3"],
                },
                {
                    "selected_features": [True, False, False],
                    "feature_names": ["f1", "f2", "f3"],
                },
            ]
        }
    }
    result = ExperimentResult(raw)

    sel = result.get_selected_features()
    assert len(sel) == 6
    # Check order for first fold
    fold0 = sel[sel["Fold"] == 0]
    assert fold0.loc[fold0["FeatureName"] == "f3", "Order"].iloc[0] == 1
    assert fold0.loc[fold0["FeatureName"] == "f1", "Order"].iloc[0] == 2

    stability = result.get_feature_stability()
    assert (
        stability.loc[stability["FeatureName"] == "f1", "SelectionFrequency"].iloc[0]
        == 1.0
    )
    assert (
        stability.loc[stability["FeatureName"] == "f3", "SelectionFrequency"].iloc[0]
        == 0.5
    )


def test_result_error_paths(tmp_path):
    # load non-existent
    with pytest.raises(FileNotFoundError):
        ExperimentResult.load(tmp_path / "non_existent.pkl")

    # summary empty
    res = ExperimentResult({})
    assert res.summary().empty

    # get_detailed_scores with error in raw
    res_err = ExperimentResult({"m1": {"error": "failed"}})
    assert res_err.get_detailed_scores().empty
    assert res_err.get_temporal_score_summary().empty
    assert res_err.get_predictions().empty
    assert res_err.get_splits().empty
    assert res_err.get_fit_diagnostics().empty

    # empty curves
    assert res.get_roc_curve().empty
    assert res.get_pr_curve().empty
    assert res.get_roc_auc_summary().empty
    assert res.get_pr_auc_summary().empty


def test_result_save_default_path():
    res = ExperimentResult({}, config={"output_dir": "."})
    path = res.save()
    assert path.exists()
    path.unlink()


def test_compare_models_all_pairs():
    raw = {
        "m1": {
            "metrics": {"accuracy": {"mean": 0.8, "std": 0.1, "folds": [0.8, 0.8]}},
            "predictions": [
                {"sample_index": [0], "sample_id": ["s0"], "y_true": [0], "y_pred": [0]}
            ],
        },
        "m2": {
            "metrics": {"accuracy": {"mean": 0.7, "std": 0.1, "folds": [0.7, 0.7]}},
            "predictions": [
                {"sample_index": [0], "sample_id": ["s0"], "y_true": [0], "y_pred": [1]}
            ],
        },
        "m3": {
            "metrics": {"accuracy": {"mean": 0.6, "std": 0.1, "folds": [0.6, 0.6]}},
            "predictions": [
                {"sample_index": [0], "sample_id": ["s0"], "y_true": [0], "y_pred": [1]}
            ],
        },
    }
    res = ExperimentResult(raw)
    comp = res.compare_models(metric="accuracy", n_permutations=5)
    assert not comp.empty


def test_get_feature_scores_with_pvalues():
    raw = {
        "m": {
            "metadata": [
                {
                    "feature_scores": [10.0, 5.0],
                    "feature_pvalues": [0.001, 0.05],
                    "feature_names": ["a", "b"],
                    "feature_selection_method": "f_classif",
                }
            ]
        }
    }
    result = ExperimentResult(raw)
    scores = result.get_feature_scores()
    assert scores.loc[0, "Score"] == 10.0
    assert scores.loc[0, "PValue"] == 0.001


def test_get_statistical_nulls_and_model_artifacts():
    raw = {
        "m": {
            "statistical_nulls": {"acc": [0.4, 0.5, 0.6]},
            "metadata": [{"artifacts": {"coef": [1, 2], "intercept": 0.5}}],
        }
    }
    result = ExperimentResult(raw)

    nulls = result.get_statistical_nulls()
    assert "m" in nulls
    assert "acc" in nulls["m"]

    artifacts = result.get_model_artifacts()
    assert len(artifacts) == 2
    assert set(artifacts["Key"]) == {"coef", "intercept"}


def test_generalization_matrix_formatting():
    # Mock result with 2D temporal scores
    raw = {
        "tg_model": {
            "status": "success",
            "metrics": {
                "accuracy": {"folds": [np.ones((2, 2)) * 0.8, np.ones((2, 2)) * 0.9]}
            },
        }
    }
    result = ExperimentResult(raw, config={}, meta={"time_axis": [0.1, 0.2]})

    # 1. Long format
    long_df = result.get_generalization_matrix()
    assert len(long_df) == 4
    assert set(long_df.columns) == {"Model", "Metric", "TrainTime", "TestTime", "Value"}

    # 2. Wide format (matrix) for specific model
    wide_df = result.get_generalization_matrix(model="tg_model")
    assert wide_df.shape == (2, 2)
    assert wide_df.index.tolist() == [0.1, 0.2]
    assert wide_df.columns.tolist() == [0.1, 0.2]
    assert np.allclose(wide_df.values, 0.85)
