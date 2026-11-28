from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
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
    register_estimator,
    register_estimator_spec,
    stack_oof_predictions,
)
from coco_pipe.decoding._specs import EstimatorSpec
from coco_pipe.dim_reduction import (
    FIT_METRIC_COLUMNS,
    build_fit_request,
    run_fit_group,
)
from coco_pipe.io import DataContainer, write_json
from coco_pipe.report import DimReductionReportContext, build_dataset_report
from coco_pipe.utils import get_environment_info


def _make_container(n=30, d=6, seed=0):
    rng = np.random.default_rng(seed)
    subjects = np.array([f"s{i % 6:02d}" for i in range(n)], dtype=object)
    conditions = np.array(
        ["EO" if i % 2 == 0 else "EC" for i in range(n)], dtype=object
    )
    return DataContainer(
        X=rng.normal(size=(n, d)),
        dims=("obs", "feature"),
        coords={
            "feature": np.array([f"f{j}" for j in range(d)], dtype=object),
            "subject": subjects,
            "condition": conditions,
        },
        ids=np.array([f"obs-{i:03d}" for i in range(n)], dtype=object),
        meta={},
    )


def test_dimred_skip_fit_metrics_and_clean_report(tmp_path):
    container = _make_container(n=24, d=6)
    unit_spec = {
        "unit_type": "flat",
        "unit_name": "global",
        "unit_key": "global",
    }
    input_sig = {
        "analysis_mode": "flat",
        "input_mode": "raw",
        "representation": "recording",
    }
    reqs = [
        build_fit_request(
            container=container,
            scope="condition",
            condition="EO",
            unit_spec=unit_spec,
            reducer="PCA",
            n_components=n,
            input_signature=input_sig,
            output_root=tmp_path,
            compute_fit_metrics=False,
        )
        for n in (2, 3)
    ]
    records = run_fit_group(reqs, errors="raise")
    assert len(records) == 2
    for rec in records:
        assert rec["status"] == "success"
        for col in FIT_METRIC_COLUMNS:
            assert np.isnan(rec[col])

    fit_runs_path = tmp_path / "runs" / "fit_runs.json"
    fit_runs_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(fit_runs_path, records, indent=2)

    ctx = DimReductionReportContext(
        analysis_mode="flat",
        selection_metric="",
        selection_eval_name="",
        reducers=["PCA"],
        conditions=["EO"],
        container_builder=lambda _: container,
        output_root=tmp_path,
        eval_specs=[],
        interactive=False,
        input_mode="raw",
        representation="recording",
        dataset_name="demo",
        report_title="Demo",
    )
    report = build_dataset_report(
        ctx,
        fit_runs_path=fit_runs_path,
        eval_runs_path=tmp_path / "runs" / "eval_runs.json",
        containers_by_scope={("condition", "EO"): container},
    )
    html = report.render()
    assert "Overview" in html
    assert "Condition Ranking" not in html
    assert "Evaluation Results" not in html
    assert "Peak Performance" not in html
    assert "Fit ranking" not in html


def test_subsample_and_balance_ratio():
    c = DataContainer(
        X=np.arange(30, dtype=float).reshape(15, 2),
        dims=("obs", "feature"),
        coords={
            "feature": np.array(["f0", "f1"]),
            "subject": np.array(["a"] * 10 + ["b"] * 5),
            "condition": np.array(["EO"] * 8 + ["EC"] * 2 + ["EO"] * 5),
        },
        ids=np.array([f"r{i}" for i in range(15)]),
    )
    capped = c.subsample({"EO": 3}, by=("subject", "condition"))
    a_eo = (capped.coords["subject"] == "a") & (capped.coords["condition"] == "EO")
    a_ec = (capped.coords["subject"] == "a") & (capped.coords["condition"] == "EC")
    assert a_eo.sum() == 3 and a_ec.sum() == 2
    # Deterministic per-cell draw independent of other groups
    alone = c.isel(obs=np.arange(10)).subsample({"EO": 3}, by=("subject", "condition"))
    assert list(alone.ids) == list(capped.ids[:5])
    # Scalar integer cap across single coordinate
    uniform = c.subsample(4, by="subject")
    assert (uniform.coords["subject"] == "a").sum() == 4
    assert (uniform.coords["subject"] == "b").sum() == 4

    # Balance with 2:1 ratio across covariate strata
    rng = np.random.default_rng(0)
    n_ref, n_pool = 12, 40
    role = np.array(["ref"] * n_ref + ["pool"] * n_pool)
    sex = np.array(["F"] * 6 + ["M"] * 6 + ["F"] * 20 + ["M"] * 20)
    bc = DataContainer(
        X=rng.normal(size=(n_ref + n_pool, 2)),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"], "role": role, "sex": sex},
        ids=np.array([f"c{i}" for i in range(n_ref + n_pool)]),
    )
    balanced = bc.balance(
        target="role",
        covariates=["sex"],
        ratio={"ref": 1, "pool": 2},
        random_state=0,
    )
    counts = pd.Series(balanced.coords["role"]).value_counts().to_dict()
    assert counts == {"ref": 12, "pool": 24}
    by_sex = (
        pd.DataFrame({"role": balanced.coords["role"], "sex": balanced.coords["sex"]})
        .groupby(["sex", "role"])
        .size()
        .unstack("role")
    )
    assert (by_sex["pool"] == 2 * by_sex["ref"]).all()


def test_get_environment_info_caches_and_isolates_mutations():
    get_environment_info.cache_clear()
    with patch(
        "coco_pipe.utils.get_git_revision_hash", return_value="abc123"
    ) as mock_git:
        first = get_environment_info()
        first["tag"] = "mutated"
        first["versions"]["numpy"] = "mutated"
        second = get_environment_info()
        assert mock_git.call_count == 1
        assert "tag" not in second
        assert second["versions"]["numpy"] != "mutated"
    get_environment_info.cache_clear()


def test_predefined_folds_per_block_pca_tuning_and_return_estimator():
    rng = np.random.default_rng(0)
    n = 50
    y = np.array([0, 1] * (n // 2))
    # Block A: 12 noisy features; Block B: 3 informative features
    X_a = rng.normal(scale=10.0, size=(n, 12))
    X_b = rng.normal(size=(n, 3))
    X_b[:, 0] += 2.0 * y
    X = np.hstack([X_a, X_b])
    ids = np.array([f"c{i:02d}" for i in range(n)])
    folds = np.arange(n) % 5
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": np.array([f"f{j}" for j in range(15)], dtype=object),
            "feature_block": np.array(["A"] * 12 + ["B"] * 3, dtype=object),
        },
        ids=ids,
    )

    # External unseen group shifted halfway between class 0 and class 1
    X_ext = np.hstack([rng.normal(scale=10.0, size=(10, 12)), rng.normal(size=(10, 3))])
    X_ext[:, 12] += 1.0

    cfg = ExperimentConfig(
        models={
            "lr": ClassicalModelConfig(
                estimator="LogisticRegression",
                params={
                    "solver": "lbfgs",
                    "class_weight": "balanced",
                    "max_iter": 5000,
                },
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
        reducer=ReducerConfig(enabled=True, n_components={"A": 4, "B": None}),
        return_estimator=True,
        metrics=["roc_auc", "accuracy"],
        n_jobs=1,
        verbose=False,
    )
    res = Experiment(cfg).run(
        container,
        y,
        groups=ids,
        sample_ids=ids,
        sample_metadata=pd.DataFrame({"Subject": ids, "Session": "01", "fold": folds}),
    )
    preds = res.get_predictions()
    assert len(preds) == n
    assert "y_score" in preds.columns and "y_proba_1" in preds.columns
    assert roc_auc_score(preds["y_true"], preds["y_proba_1"]) > 0.8

    scores = res.predict(X_ext, method="decision_function")
    assert scores.shape == (len(X_ext),)
    fold_scores = res.predict(X_ext, method="decision_function", aggregate=None)
    assert fold_scores.shape == (5, len(X_ext))
    assert np.allclose(fold_scores.mean(axis=0), scores)
    proba = res.predict(X_ext, method="predict_proba")
    assert proba.shape == (len(X_ext), 2)
    # JSON serialization cleanly omits retained estimators
    payload = res.to_payload(serializable=True)
    assert "estimators" not in payload["results"]["lr"]


def test_ensemble_predictions_and_missing_block_stacking():
    rng = np.random.default_rng(1)
    n = 60
    idx = pd.Index([f"c{i:02d}" for i in range(n)])
    y = pd.Series(np.arange(n) % 2, index=idx)
    folds = pd.Series(np.arange(n) % 5, index=idx)

    X1 = pd.DataFrame(rng.normal(size=(n, 4)), index=idx)
    X1.iloc[:, 0] += 1.5 * y
    # Second source only available for first 45 children
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

    # Ensemble of two fits with partial overlap
    f1 = pd.DataFrame(
        {
            "group_id": idx,
            "y_true": y.to_numpy(),
            "y_proba_0": 1 - stacked,
            "y_proba_1": stacked,
        }
    )
    f2 = pd.DataFrame(
        {
            "group_id": idx[:30],
            "y_true": y.iloc[:30].to_numpy(),
            "y_proba_0": 1 - stacked.iloc[:30],
            "y_proba_1": stacked.iloc[:30],
        }
    )
    ens, summary = ensemble_predictions(
        {"a": f1, "b": f2}, evaluate=True, n_permutations=50, n_bootstrap=20
    )
    assert len(ens) == n and ens["n_fits"].max() == 2 and ens["n_fits"].min() == 1
    assert summary["auc"] > 0.8 and summary["n_units"] == n
    direct = evaluate_binary_scores(
        ens["y"], ens["score"], n_permutations=50, n_bootstrap=20
    )
    assert direct["auc"] == pytest.approx(summary["auc"])


def test_custom_neural_estimator_instantiates_and_runs_via_experiment():
    class _ToyBagMIL(ClassifierMixin, BaseEstimator):
        def __init__(self, scale: float = 1.0):
            self.scale = scale

        def get_metadata_routing(self):
            requests = super().get_metadata_routing()
            requests.fit.add_auto_request("groups")
            requests.predict.add_auto_request("groups")
            requests.predict_proba.add_auto_request("groups")
            return requests

        def fit(self, X, y, groups=None):
            self.classes_ = np.unique(y)
            self.mean_pos_ = np.asarray(X)[y == 1].mean(axis=0)
            self.mean_neg_ = np.asarray(X)[y == 0].mean(axis=0)
            self._last_weights = None
            return self

        def predict_proba(self, X, groups=None):
            X = np.asarray(X)
            diff = self.scale * (self.mean_pos_ - self.mean_neg_)
            logits = X @ diff
            if groups is not None:
                # Pool logits per group (bag)
                s = (
                    pd.Series(logits)
                    .groupby(np.asarray(groups))
                    .transform("mean")
                    .to_numpy()
                )
                self._last_weights = {"n_rows": len(X)}
            else:
                s = logits
            p1 = 1.0 / (1.0 + np.exp(-np.clip(s, -20, 20)))
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X, groups=None):
            return (self.predict_proba(X, groups=groups)[:, 1] >= 0.5).astype(int)

        def get_artifact_metadata(self):
            return self._last_weights

    name = "_TestToyBagMIL"
    register_estimator(name)(_ToyBagMIL)
    register_estimator_spec(
        EstimatorSpec(
            name=name,
            import_path="tests.test_unification_primitives:_ToyBagMIL",
            family="neural",
            task=("classification",),
            supports_groups=True,
            supports_proba=True,
        )
    )
    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(12)], 3)
    y = np.repeat([0, 1] * 6, 3)
    X = rng.normal(size=(36, 4))
    X[:, 0] += 2.0 * y

    cfg = ExperimentConfig(
        models={"mil": ClassicalModelConfig(estimator=name, params={"scale": 1.0})},
        cv=CVConfig(strategy="stratified_group_kfold", n_splits=3),
        use_scaler=False,
        metrics=["accuracy", "roc_auc"],
        n_jobs=1,
        verbose=False,
    )
    res = Experiment(cfg).run(X, y, groups=groups)
    preds = res.get_predictions()
    assert roc_auc_score(preds["y_true"], preds["y_proba_1"]) > 0.85
    assert res.raw["mil"]["metadata"][0]["artifacts"]["n_rows"] > 0
