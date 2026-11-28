import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from coco_pipe.decoding import (
    ClassicalModelConfig,
    CVConfig,
    Experiment,
    ExperimentConfig,
    ReducerConfig,
    TuningConfig,
    ensemble_predictions,
    evaluate_binary_scores,
    stack_oof_predictions,
)
from coco_pipe.io import DataContainer


def test_ensemble_predictions_rank_averages_partial_overlap():
    n = 40
    idx = pd.Index([f"c{i:02d}" for i in range(n)])
    y = np.arange(n) % 2
    scores = np.linspace(0.05, 0.95, n) + 0.4 * y

    fit_a = pd.DataFrame(
        {
            "Group": idx,
            "y_true": y,
            "y_proba_0": 1.0 - scores / scores.max(),
            "y_proba_1": scores / scores.max(),
        }
    )
    fit_b = pd.DataFrame(
        {
            "Group": idx[:20],
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

    # Empty fits and conflicting labels raise clearly
    with pytest.raises(ValueError, match="must not be empty"):
        ensemble_predictions({})

    conflict = fit_b.copy()
    conflict.loc[0, "y_true"] = 1 - conflict.loc[0, "y_true"]
    with pytest.raises(ValueError, match="two labels across fits"):
        ensemble_predictions({"a": fit_a, "b": conflict})

    within = pd.concat([fit_b.iloc[:2], fit_b.iloc[:1].assign(y_true=1)])
    with pytest.raises(ValueError, match="two labels within fit"):
        ensemble_predictions({"a": within})


def test_ensemble_predictions_uses_coco_schema_and_explicit_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = np.arange(30) % 2
    X[:, 0] += 2.0 * y
    ids = np.array([f"u{i:02d}" for i in range(30)])

    cfg = ExperimentConfig(
        models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
        cv=CVConfig(strategy="stratified", n_splits=3, random_state=0),
        metrics=["accuracy"],
        n_jobs=1,
        verbose=False,
    )
    # No groups -> Group column is all-None, so default falls back to SampleID
    res_ungrouped = Experiment(cfg).run(X, y, sample_ids=ids)
    # With groups -> default resolves Group directly from ExperimentResult
    res_grouped = Experiment(cfg).run(X, y, groups=ids, sample_ids=ids)
    ens = ensemble_predictions({"ungrouped": res_ungrouped, "grouped": res_grouped})
    assert len(ens) == 30
    assert (ens["n_fits"] == 2).all()

    # Custom DataFrames require explicit unit_col; score_col / y_score / unlabeled work
    custom_a = pd.DataFrame({"patient": ["p1", "p2", "p3"], "y_score": [0.1, 0.5, 0.9]})
    custom_b = pd.DataFrame(
        {"Patient": ["p1", "p2", "p3"], "custom_score": [0.9, 0.5, 0.1]}
    )
    with pytest.raises(ValueError, match="Pass unit_col="):
        ensemble_predictions({"a": custom_a})
    unlabeled = ensemble_predictions(
        {"a": custom_a, "b": custom_b.rename(columns={"custom_score": "y_score"})},
        unit_col="patient",
    )
    assert unlabeled["y"].isna().all()
    assert unlabeled["score"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3])

    explicit = ensemble_predictions(
        {"b": custom_b, "empty": pd.DataFrame()},
        unit_col="patient",
        score_col="Custom_Score",
    )
    assert explicit["unit"].tolist() == ["p1", "p2", "p3"]

    # Validation errors: missing score, ambiguous column, no valid predictions
    with pytest.raises(ValueError, match="no class probabilities"):
        ensemble_predictions({"bad": pd.DataFrame({"SampleID": ["s1"]})})
    with pytest.raises(ValueError, match="ambiguous"):
        ensemble_predictions(
            {"amb": pd.DataFrame({"id": ["a"], "ID": ["a"], "score": [0.5]})},
            unit_col="Id",
        )
    with pytest.raises(ValueError, match="No valid predictions"):
        ensemble_predictions(
            {"nan_only": pd.DataFrame({"SampleID": ["s1"], "score": [np.nan]})}
        )


def test_stack_oof_predictions_with_containers_and_missing_blocks():
    rng = np.random.default_rng(1)
    n = 60
    idx = pd.Index([f"c{i:02d}" for i in range(n)])
    y = pd.Series(np.arange(n) % 2, index=idx)
    folds = pd.Series(np.arange(n) % 5, index=idx)

    # First source: DataContainer with two feature blocks (one reduced, one kept)
    X1 = rng.normal(size=(n, 6))
    X1[:, 0] += 1.5 * y.to_numpy()
    dc1 = DataContainer(
        X=X1,
        dims=("obs", "feature"),
        coords={
            "feature": np.array([f"f{j}" for j in range(6)], dtype=object),
            "feature_block": np.array(["sig"] * 2 + ["noise"] * 4, dtype=object),
        },
        ids=idx.to_numpy(),
    )
    # Second source: DataFrame available only for the first 45 units (NaN for rest)
    X2 = pd.DataFrame(rng.normal(size=(45, 4)), index=idx[:45])
    X2.iloc[:, 0] += 1.5 * y.iloc[:45]

    cfg_blocks = ExperimentConfig(
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
        reducer=ReducerConfig(enabled=True, n_components={"sig": None, "noise": 2}),
        metrics=["accuracy"],
        n_jobs=1,
        verbose=False,
    )
    cfg_plain = cfg_blocks.model_copy(
        update={"reducer": ReducerConfig(enabled=True, n_components=3)}
    )

    inner_cv = CVConfig(strategy="stratified", n_splits=3, shuffle=True, random_state=0)
    stacked = stack_oof_predictions(
        {"s1": (dc1, cfg_blocks), "s2": (X2, cfg_plain)},
        y,
        folds,
        inner_cv=inner_cv,
        random_state=0,
    )
    assert stacked.notna().all()
    assert roc_auc_score(y, stacked) > 0.8

    # Custom combiner via coco-pipe ClassicalModelConfig and raw sklearn estimator
    cfg_combiner = ClassicalModelConfig(
        estimator="HistGradientBoostingClassifier",
        params={"max_depth": 2, "max_iter": 30, "random_state": 0},
    )
    stacked_cfg = stack_oof_predictions(
        {"s1": (dc1, cfg_blocks)},
        y,
        folds,
        inner_cv=inner_cv,
        combiner=cfg_combiner,
        random_state=0,
    )
    stacked_est = stack_oof_predictions(
        {"s1": (dc1, cfg_blocks)},
        y,
        folds,
        inner_cv=inner_cv,
        combiner=HistGradientBoostingClassifier(
            max_depth=2, max_iter=30, random_state=0
        ),
        random_state=0,
    )
    np.testing.assert_allclose(stacked_cfg.to_numpy(), stacked_est.to_numpy())

    # Validation errors
    with pytest.raises(ValueError, match="must not be empty"):
        stack_oof_predictions({}, y, folds)
    no_ids = DataContainer(X=X1, dims=("obs", "feature"))
    with pytest.raises(ValueError, match="must define ids"):
        stack_oof_predictions({"bad": (no_ids, cfg_plain)}, y, folds)
