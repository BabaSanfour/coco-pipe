import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from coco_pipe.decoding import (
    ClassicalModelConfig,
    CVConfig,
    ExperimentConfig,
    ReducerConfig,
    TuningConfig,
    ensemble_predictions,
    evaluate_binary_scores,
    stack_oof_predictions,
)


def test_ensemble_predictions_rank_averages_partial_overlap():
    n = 40
    idx = pd.Index([f"c{i:02d}" for i in range(n)])
    y = np.arange(n) % 2
    scores = np.linspace(0.05, 0.95, n) + 0.4 * y

    fit_a = pd.DataFrame(
        {
            "group_id": idx,
            "y_true": y,
            "y_proba_0": 1.0 - scores / scores.max(),
            "y_proba_1": scores / scores.max(),
        }
    )
    fit_b = pd.DataFrame(
        {
            "group_id": idx[:20],
            "y_true": y[:20],
            "y_proba_0": 1.0 - scores[:20] / scores.max(),
            "y_proba_1": scores[:20] / scores.max(),
        }
    )

    ens, summary = ensemble_predictions(
        {"a": fit_a, "b": fit_b},
        evaluate=True,
        n_permutations=50,
        n_bootstrap=20,
        random_state=0,
    )
    assert len(ens) == n
    assert ens["n_fits"].max() == 2 and ens["n_fits"].min() == 1
    assert summary["n_units"] == n
    direct = evaluate_binary_scores(
        ens["y"], ens["score"], n_permutations=50, n_bootstrap=20, random_state=0
    )
    assert direct["auc"] == pytest.approx(summary["auc"])

    with pytest.raises(ValueError, match="must not be empty"):
        ensemble_predictions({})

    conflict = fit_b.copy()
    conflict.loc[0, "y_true"] = 1 - conflict.loc[0, "y_true"]
    with pytest.raises(ValueError, match="two labels across fits"):
        ensemble_predictions({"a": fit_a, "b": conflict})


def test_stack_oof_predictions_preserves_missing_blocks():
    rng = np.random.default_rng(1)
    n = 60
    idx = pd.Index([f"c{i:02d}" for i in range(n)])
    y = pd.Series(np.arange(n) % 2, index=idx)
    folds = pd.Series(np.arange(n) % 5, index=idx)

    X1 = pd.DataFrame(rng.normal(size=(n, 4)), index=idx)
    X1.iloc[:, 0] += 1.5 * y
    # Second source only available for the first 45 units
    X2 = pd.DataFrame(rng.normal(size=(45, 4)), index=idx[:45])
    X2.iloc[:, 0] += 1.5 * y.iloc[:45]

    base_cfg = ExperimentConfig(
        models={
            "lr": ClassicalModelConfig(
                estimator="LogisticRegression",
                params={"solver": "lbfgs", "max_iter": 2000},
            )
        },
        grids={"lr": {"alpha": [10.0, 1.0, 0.1]}},
        cv=CVConfig(strategy="predefined", fold_col="fold"),
        tuning=TuningConfig(
            enabled=True,
            scoring="roc_auc",
            n_jobs=1,
            cv=CVConfig(
                strategy="stratified", n_splits=3, shuffle=True, random_state=0
            ),
        ),
        reducer=ReducerConfig(enabled=True, n_components=3),
        metrics=["accuracy"],
        n_jobs=1,
        verbose=False,
    )
    stacked = stack_oof_predictions(
        {"s1": (X1, base_cfg), "s2": (X2, base_cfg)},
        y,
        folds,
        inner_splits=3,
        random_state=0,
    )
    assert stacked.notna().all()
    assert roc_auc_score(y, stacked) > 0.8
