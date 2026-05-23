from __future__ import annotations

import numpy as np

from coco_pipe.decoding.result import ExperimentResult


def make_synthetic_feature_metadata(n_features: int = 8):
    """Build explicit feature metadata for sensor-wise visualization tests."""
    feature_names = [f"F{idx + 1}" for idx in range(n_features)]
    return [
        {
            "FeatureName": name,
            "Sensor": f"E{idx % 4 + 1}",
            "FeatureFamily": "spectral" if idx % 2 == 0 else "temporal",
            "x": float(np.cos(idx / n_features * 2 * np.pi) * 0.4 + (0.05 * (idx % 3))),
            "y": float(np.sin(idx / n_features * 2 * np.pi) * 0.4 - (0.03 * (idx % 2))),
        }
        for idx, name in enumerate(feature_names)
    ]


def make_synthetic_result(
    n_models: int = 2,
    n_folds: int = 3,
    n_samples: int = 30,
    n_classes: int = 2,
    n_features: int = 8,
    n_times: int = 5,
) -> ExperimentResult:
    """Build a deterministic decoding result without model fitting."""
    rng = np.random.default_rng(13)
    model_names = [f"model_{idx + 1}" for idx in range(n_models)]
    feature_names = [f"F{idx + 1}" for idx in range(n_features)]
    time_axis = np.linspace(-0.2, 0.6, n_times).round(3).tolist()
    raw = {}

    for model_idx, model_name in enumerate(model_names):
        scalar_acc = np.clip(
            0.62 + 0.04 * model_idx + rng.normal(0, 0.02, n_folds), 0, 1
        )
        scalar_auc = np.clip(
            0.70 + 0.03 * model_idx + rng.normal(0, 0.02, n_folds), 0, 1
        )
        temporal = [
            np.clip(
                0.55
                + 0.04 * model_idx
                + np.linspace(0, 0.12, n_times)
                + rng.normal(0, 0.01, n_times),
                0,
                1,
            )
            for _ in range(n_folds)
        ]
        generalization = [
            np.clip(
                0.52
                + 0.03 * model_idx
                + np.eye(n_times) * 0.16
                + rng.normal(0, 0.01, (n_times, n_times)),
                0,
                1,
            )
            for _ in range(n_folds)
        ]

        predictions = []
        splits = []
        fold_size = max(2, n_samples // n_folds)
        for fold in range(n_folds):
            start = fold * fold_size
            stop = min(n_samples, start + fold_size)
            if fold == n_folds - 1:
                stop = n_samples
            indices = np.arange(start, stop)
            if len(indices) == 0:
                continue
            train_indices = np.setdiff1d(np.arange(n_samples), indices)
            splits.append(
                {
                    "train_idx": train_indices,
                    "train_sample_id": np.array(
                        [f"S{idx:03d}" for idx in train_indices]
                    ),
                    "train_group": np.array([f"G{idx % 4}" for idx in train_indices]),
                    "train_metadata": {
                        "Subject": np.array(
                            [f"sub-{idx % 5:02d}" for idx in train_indices]
                        ),
                        "Session": np.array(["ses-01"] * len(train_indices)),
                        "Site": np.array([f"site-{idx % 2}" for idx in train_indices]),
                    },
                    "test_idx": indices,
                    "test_sample_id": np.array([f"S{idx:03d}" for idx in indices]),
                    "test_group": np.array([f"G{idx % 4}" for idx in indices]),
                    "test_metadata": {
                        "Subject": np.array([f"sub-{idx % 5:02d}" for idx in indices]),
                        "Session": np.array(["ses-01"] * len(indices)),
                        "Site": np.array([f"site-{idx % 2}" for idx in indices]),
                    },
                }
            )
            y_true = indices % n_classes
            flips = rng.random(len(indices)) < (0.20 - 0.03 * model_idx)
            y_pred = y_true.copy()
            y_pred[flips] = (y_pred[flips] + 1) % n_classes
            proba = np.full((len(indices), n_classes), 0.15 / max(1, n_classes - 1))
            for row_idx, pred in enumerate(y_pred):
                proba[row_idx, pred] = 0.85
            predictions.append(
                {
                    "sample_index": indices,
                    "sample_id": np.array([f"S{idx:03d}" for idx in indices]),
                    "group": np.array([f"G{idx % 4}" for idx in indices]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_proba": proba,
                    "sample_metadata": {
                        "Subject": np.array([f"sub-{idx % 5:02d}" for idx in indices]),
                        "Session": np.array(["ses-01"] * len(indices)),
                        "Site": np.array([f"site-{idx % 2}" for idx in indices]),
                    },
                }
            )

        importances_raw = np.abs(
            rng.normal(
                loc=0.3 + 0.05 * model_idx, scale=0.1, size=(n_folds, n_features)
            )
        )
        selected = importances_raw > np.quantile(importances_raw, 0.55, axis=1)[:, None]
        metadata = []
        for fold in range(n_folds):
            metadata.append(
                {
                    "feature_names": feature_names,
                    "selected_features": selected[fold].tolist(),
                    "selection_order": np.argsort(importances_raw[fold])[::-1].tolist(),
                    "feature_scores": (importances_raw[fold] * 10).tolist(),
                    "feature_pvalues": np.linspace(0.001, 0.08, n_features).tolist(),
                    "feature_selection_method": "f_classif",
                    "best_params": {"C": 1.0 + model_idx, "penalty": "l2"},
                    "search_results": [
                        {
                            "candidate": candidate,
                            "rank_test_score": candidate + 1,
                            "mean_test_score": float(
                                0.7 - 0.03 * candidate + 0.02 * model_idx
                            ),
                            "std_test_score": 0.02,
                            "params": {"C": 10 ** (-candidate)},
                        }
                        for candidate in range(3)
                    ],
                    "artifacts": {
                        "history": [
                            {"epoch": 1, "loss": 0.7, "val_loss": 0.75},
                            {"epoch": 2, "loss": 0.55, "val_loss": 0.62},
                        ]
                    },
                }
            )

        raw[model_name] = {
            "metrics": {
                "accuracy": {
                    "mean": float(np.mean(scalar_acc)),
                    "std": float(np.std(scalar_acc)),
                    "folds": scalar_acc.tolist(),
                },
                "roc_auc": {
                    "mean": float(np.mean(scalar_auc)),
                    "std": float(np.std(scalar_auc)),
                    "folds": scalar_auc.tolist(),
                },
                "temporal_accuracy": {
                    "mean": np.mean(np.stack(temporal), axis=0),
                    "std": np.std(np.stack(temporal), axis=0),
                    "folds": temporal,
                },
                "generalization_accuracy": {
                    "mean": np.mean(np.stack(generalization), axis=0),
                    "std": np.std(np.stack(generalization), axis=0),
                    "folds": generalization,
                },
            },
            "predictions": predictions,
            "splits": splits,
            "diagnostics": [
                {
                    "fit_time": 0.10 + 0.02 * fold + 0.01 * model_idx,
                    "predict_time": 0.03 + 0.01 * fold,
                    "score_time": 0.02,
                    "total_time": 0.15 + 0.03 * fold + 0.01 * model_idx,
                    "warnings": [],
                }
                for fold in range(n_folds)
            ],
            "importances": {
                "mean": importances_raw.mean(axis=0),
                "std": importances_raw.std(axis=0),
                "raw": importances_raw,
                "feature_names": feature_names,
            },
            "metadata": metadata,
            "statistical_assessment": [
                {
                    "Metric": "accuracy",
                    "Observed": float(np.mean(scalar_acc)),
                    "PValue": 0.01,
                    "CorrectedPValue": 0.02,
                    "Significant": True,
                    "NullMethod": "permutation",
                    "NPermutations": 100,
                    "InferentialUnit": "subject",
                    "ChanceThreshold": 1 / n_classes,
                    "NullMedian": 1 / n_classes,
                    "NEff": n_samples,
                    "Time": None,
                    "TrainTime": None,
                    "TestTime": None,
                    "NullLower": 0.40,
                    "NullUpper": 0.60,
                },
                *[
                    {
                        "Metric": "temporal_accuracy",
                        "Observed": float(np.mean([fold[t] for fold in temporal])),
                        "PValue": 0.02 if t in {2, 3} else 0.4,
                        "CorrectedPValue": 0.04 if t in {2, 3} else 0.5,
                        "Significant": t in {2, 3},
                        "NullMethod": "permutation",
                        "NPermutations": 100,
                        "InferentialUnit": "subject",
                        "ChanceThreshold": 1 / n_classes,
                        "NullMedian": 1 / n_classes,
                        "NEff": n_samples,
                        "Time": time_axis[t],
                        "TrainTime": None,
                        "TestTime": None,
                        "NullLower": 0.40,
                        "NullUpper": 0.60,
                    }
                    for t in range(n_times)
                ],
            ],
        }

    return ExperimentResult(
        raw,
        config={"tag": "synthetic", "task": "classification"},
        meta={
            "task": "classification",
            "n_samples": n_samples,
            "n_features": n_features,
            "observation_level": "epoch",
            "inferential_unit": "subject",
            "time_axis": time_axis,
        },
        time_axis=time_axis,
    )
