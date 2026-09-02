"""
Decoding Engine
===============
Functions for fitting, scoring, and metadata extraction.

This module provides the core execution logic for cross-validation folds.
It is designed for high-performance, parallel execution.
"""

import inspect
import logging
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from ._constants import GROUP_CV_STRATEGIES
from ._metrics import get_metric_spec
from ._splitters import _CVWithGroups, cv_uses_groups, get_cv_splitter
from .interfaces import NeuralTrainable
from .scalers import SubjectStandardScaler

logger = logging.getLogger(__name__)


def _wrapped_pipeline(estimator: Any) -> Pipeline | None:
    """Find the preprocessing pipeline inside search/calibration wrappers."""
    seen: set[int] = set()
    stack = [estimator]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, Pipeline):
            return candidate
        for attribute in ("best_estimator_", "estimator", "base_estimator"):
            stack.append(getattr(candidate, attribute, None))
    return None


def _release_foundation_memory(estimator: Any) -> None:
    """Release a fitted foundation/neural estimator's torch modules after a fold.

    Walks the (possibly pipeline/search/calibration-wrapped) *estimator*, calls
    ``release_memory`` on any component that exposes it, then forces a GC pass
    and clears the CUDA cache. This keeps peak memory bounded to roughly one
    fold's model instead of accumulating loaded backbones across the CV loop.
    Best-effort: teardown must never fail a fold that already produced its
    scores, so component errors are swallowed. No-op for estimators without a
    ``release_memory`` hook (i.e. all classical models).
    """
    seen: set[int] = set()
    stack: list[Any] = [estimator]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        release = getattr(obj, "release_memory", None)
        if callable(release):
            try:
                release()
            except Exception:
                logger.debug("release_memory failed for %r", type(obj), exc_info=True)
        # Unwrap the common sklearn containers to reach the wrapped model.
        for attr in ("estimator", "best_estimator_", "base_estimator"):
            stack.append(getattr(obj, attr, None))
        steps = getattr(obj, "named_steps", None)
        if steps:
            stack.extend(steps.values())
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class GroupedSequentialFeatureSelector(SequentialFeatureSelector):
    """
    SequentialFeatureSelector that accepts groups via Pipeline fit parameters.

    In sklearn 1.8, SFS can route metadata to its internal CV splitter when
    called directly, but Pipeline calls intermediate transformers through
    ``fit_transform`` and does not forward top-level ``groups`` to SFS. This
    adapter keeps the public SFS behavior but accepts a sliced ``groups`` fit
    parameter, binds it to the SFS CV for this fit call, and then restores the
    original ``cv`` object.
    """

    def fit(self, X: Any, y: Any = None, groups: Any = None, **params: Any):
        """Fit SFS, using ``groups`` for its internal grouped CV if supplied."""
        if groups is None:
            return super().fit(X, y, **params)

        groups_arr = (
            groups
            if isinstance(groups, (np.ndarray, pd.Series))
            else np.asarray(groups)
        )
        if len(groups_arr) != len(X):
            raise ValueError(
                "SequentialFeatureSelector groups length does not match X: "
                f"{len(groups_arr)} != {len(X)}."
            )

        original_cv = self.cv
        self.cv = _CVWithGroups(original_cv, groups_arr)
        try:
            return super().fit(X, y, **params)
        finally:
            self.cv = original_cv

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        **params: Any,
    ):
        """Fit to data, then transform it."""
        return self.fit(X, y, groups=groups, **params).transform(X)


class _SafeSelectKBest(BaseEstimator):
    """Clamp ``k`` to the fold-local feature width after reduction."""

    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        effective_k = (
            min(self.k, int(np.asarray(X).shape[1]))
            if isinstance(self.k, int)
            else self.k
        )
        self.selector_ = SelectKBest(
            score_func=self.score_func,
            k=effective_k,
        ).fit(X, y)
        self.n_features_in_ = self.selector_.n_features_in_
        self.scores_ = self.selector_.scores_
        self.pvalues_ = getattr(self.selector_, "pvalues_", None)
        self.effective_k_ = effective_k
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices=indices)


def fit_and_score_fold(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    sample_ids: np.ndarray,
    sample_metadata: dict[str, np.ndarray] | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metrics: Sequence[str],
    feature_selection_config: Any,
    calibration_config: Any,
    spec: Any,
    temporal_alignment_config: Any = None,
    tuning_config: Any = None,
    feature_names: list[str] | None = None,
    search_enabled: bool = False,
    force_serial: bool = False,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Execute a single Cross-Validation fold: Fit, Predict, and Score.

    This function is designed to be pure and standalone, making it safe for
    parallel execution via joblib. It handles the entire lifecycle of a fold,
    including feature selection, calibration, and metadata extraction.

    Parameters
    ----------
    estimator : BaseEstimator
        The un-fitted estimator instance (or pipeline) for this fold.
    X : np.ndarray
        The full 2-D feature matrix or 3-D temporal array.
    y : np.ndarray
        The full target vector of shape (n_samples,).
    groups : np.ndarray, optional
        Group labels (e.g., Subject IDs) for group-aware splitting.
    sample_ids : np.ndarray
        Unique IDs for each sample, used for tracking predictions.
    sample_metadata : dict, optional
        Pre-converted metadata dictionary (column: numpy array) for the split.
    train_idx : np.ndarray
        The indices of X/y to use for training.
    test_idx : np.ndarray
        The indices of X/y to use for testing.
    metrics : Sequence[str]
        List of metric names to compute (e.g., ['accuracy', 'roc_auc']).
    feature_selection_config : Any
        Configuration for the feature selection step (from CVConfig).
    calibration_config : Any
        Configuration for probability calibration (from CVConfig).
    spec : EstimatorSpec
        Hardened registry specification for the model.
    temporal_alignment_config : Any, optional
        Fold-level temporal alignment settings. When enabled, the training
        template is fit on ``train_idx`` before the estimator is fitted.
    tuning_config : Any, optional
        Hyperparameter tuning settings.
    feature_names : list of str, optional
        Original names of the features, used for importance labeling.
    force_serial : bool, default=False
        If True, forces the internal estimator fit to be serial.
    sample_weight : np.ndarray, optional
        Per-sample weights for the full dataset. Only the training-fold
        slice is forwarded to the classifier; test samples are never weighted.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing test indices, predictions, scores,
        feature importances, and diagnostic timing information.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sw_train = sample_weight[train_idx] if sample_weight is not None else None
    groups_train = groups[train_idx] if groups is not None else None
    test_groups = groups[test_idx] if groups is not None else None
    alignment_time = 0.0
    if getattr(temporal_alignment_config, "enabled", False):
        if groups_train is None or test_groups is None:
            raise ValueError("Temporal alignment requires participant groups.")
        if getattr(temporal_alignment_config, "method", None) != "procrustes":
            raise ValueError(
                "Unknown temporal alignment method: "
                f"{getattr(temporal_alignment_config, 'method', None)!r}."
            )
        from coco_pipe.transforms import TemporalProcrustesAlignment

        alignment_start = time.perf_counter()
        aligner = TemporalProcrustesAlignment(
            n_components=temporal_alignment_config.n_components,
            adaptation=temporal_alignment_config.adaptation,
            rotate=getattr(temporal_alignment_config, "rotate", True),
            use_shared_basis=getattr(
                temporal_alignment_config, "use_shared_basis", False
            ),
            augment=getattr(temporal_alignment_config, "augment", ()),
            augment_only=getattr(temporal_alignment_config, "augment_only", False),
            trajectory_metrics=getattr(
                temporal_alignment_config, "trajectory_metrics", None
            ),
            random_state=temporal_alignment_config.random_state,
        )
        X_train = aligner.fit_transform(X_train, groups=groups_train)
        X_test = aligner.transform(X_test, groups=test_groups)
        alignment_time = time.perf_counter() - alignment_start
    routing_pipeline = _wrapped_pipeline(estimator)
    _needs_group_routing = (
        test_groups is not None
        and routing_pipeline is not None
        and (
            isinstance(
                routing_pipeline.named_steps.get("scaler"), SubjectStandardScaler
            )
            or "erasure" in routing_pipeline.named_steps
        )
    )
    captured_warnings = []
    fit_time = np.nan
    predict_time = np.nan
    score_time = np.nan

    # 1. Fit
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        backend = (
            joblib.parallel_backend("sequential") if force_serial else nullcontext()
        )
        with backend:
            fit_estimator(
                estimator,
                X_train,
                y_train,
                groups_train,
                feature_selection_config=feature_selection_config,
                calibration_config=calibration_config,
                tuning_config=tuning_config,
                sample_weight=sw_train,
            )
    fit_time = time.perf_counter() - fit_start
    captured_warnings.extend(warning_records_to_dict("fit", warning_records))
    prediction_estimator = getattr(estimator, "best_estimator_", estimator)

    # 2. Predict (Standard or Temporal)
    predict_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        if _needs_group_routing:
            with config_context(enable_metadata_routing=True):
                y_pred = prediction_estimator.predict(X_test, groups=test_groups)
        else:
            y_pred = prediction_estimator.predict(X_test)
    predict_time = time.perf_counter() - predict_start
    captured_warnings.extend(warning_records_to_dict("predict", warning_records))

    fold_data = {
        "sample_index": test_idx,
        "sample_id": sample_ids[test_idx],
        "group": test_groups,
        "sample_metadata": metadata_slice(sample_metadata, test_idx),
        "y_true": y_test,
        "y_pred": y_pred,
    }

    # 3. Predict probabilities
    if spec.supports_proba:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            if _needs_group_routing:
                with config_context(enable_metadata_routing=True):
                    fold_data["y_proba"] = prediction_estimator.predict_proba(
                        X_test, groups=test_groups
                    )
            else:
                fold_data["y_proba"] = prediction_estimator.predict_proba(X_test)
        captured_warnings.extend(
            warning_records_to_dict("predict_proba", warning_records)
        )

    if "y_proba" not in fold_data and spec.supports_decision_function:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            if _needs_group_routing:
                with config_context(enable_metadata_routing=True):
                    fold_data["y_score"] = prediction_estimator.decision_function(
                        X_test, groups=test_groups
                    )
            else:
                fold_data["y_score"] = prediction_estimator.decision_function(X_test)
        captured_warnings.extend(
            warning_records_to_dict("decision_function", warning_records)
        )

    # 4. Extract Feature Importances (Zero Guesswork)
    imp = (
        extract_feature_importances(
            estimator,
            spec,
            fs_enabled=feature_selection_config.enabled,
            search_enabled=search_enabled,
            calibration_enabled=calibration_config.enabled,
        )
        if spec.importance[0] != "unavailable"
        else None
    )

    # 5. Compute Metrics (Pre-fetched Specs)
    scores = {}
    is_multiclass = type_of_target(y_test) == "multiclass"
    metric_specs = {m: get_metric_spec(m) for m in metrics}

    score_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        for name, m_spec in metric_specs.items():
            if m_spec.response_method == "predict":
                y_est, is_p = y_pred, False
            elif m_spec.response_method == "proba":
                y_est, is_p = fold_data.get("y_proba"), True
            else:  # proba_or_score
                y_est = fold_data.get("y_proba")
                is_p = True
                if y_est is None:
                    y_est, is_p = fold_data.get("y_score"), False

            scores[name] = compute_metric_safe(
                m_spec.scorer,
                y_test,
                y_est,
                is_multiclass,
                is_proba=is_p,
                name=name,
            )
    captured_warnings.extend(warning_records_to_dict("score", warning_records))
    score_time = time.perf_counter() - score_start

    # 6. Extract Metadata
    meta = extract_metadata(
        estimator,
        spec,
        feature_selection_config=feature_selection_config,
        search_enabled=search_enabled,
        feature_names=feature_names,
    )

    # 7. Release the fitted backbone. Foundation fits hold large torch modules;
    # tearing them down here keeps peak host/GPU memory bounded to one fold
    # instead of accumulating loaded backbones across the CV loop. Gated to the
    # foundation family so classical folds skip the gc/torch overhead entirely.
    if spec.family == "foundation":
        _release_foundation_memory(estimator)

    return {
        "test_idx": test_idx,
        "preds": fold_data,
        "scores": scores,
        "importance": imp,
        "metadata": meta,
        "split": {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_sample_id": sample_ids[train_idx],
            "test_sample_id": sample_ids[test_idx],
            "train_group": groups[train_idx] if groups is not None else None,
            "test_group": groups[test_idx] if groups is not None else None,
            "train_metadata": metadata_slice(sample_metadata, train_idx),
            "test_metadata": metadata_slice(sample_metadata, test_idx),
        },
        "diagnostics": {
            "fit_time": fit_time,
            "predict_time": predict_time,
            "score_time": score_time,
            "alignment_time": alignment_time,
            "total_time": alignment_time + fit_time + predict_time + score_time,
            "warnings": captured_warnings,
        },
    }


def fit_estimator(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    feature_selection_config: Any,
    calibration_config: Any,
    tuning_config: Any = None,
    sample_weight: np.ndarray | None = None,
) -> None:
    """
    Fit an estimator with intelligent metadata and group routing.

    Handles specialized logic for group-aware internal CV using standard
    scikit-learn fit-parameter slicing. SearchCV receives ``groups`` for its
    splitter, Pipeline receives ``fs__groups`` for SFS, and calibration CV gets
    a fold-local group binding because CalibratedClassifierCV does not pass
    ``groups`` to its splitter unless global metadata routing is enabled.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator or pipeline to fit.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    groups_train : np.ndarray, optional
        Training group labels.
    feature_selection_config : Any
        Feature selection settings.
    calibration_config : Any
        Probability calibration settings.
    tuning_config : Any
        Hyperparameter tuning settings.
    sample_weight : np.ndarray, optional
        Per-sample weights for the training fold. Forwarded as
        ``clf__sample_weight`` when the pipeline's ``clf`` step exposes
        a ``sample_weight`` parameter in its ``fit`` signature.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    calibrated = isinstance(estimator, CalibratedClassifierCV)
    fitted_estimator = estimator.estimator if calibrated else estimator
    search_cv = isinstance(fitted_estimator, (GridSearchCV, RandomizedSearchCV))

    pipeline = fitted_estimator.estimator if search_cv else fitted_estimator
    sfs = None
    if isinstance(pipeline, Pipeline) and "fs" in pipeline.named_steps:
        sfs = pipeline.named_steps["fs"]
        if (
            getattr(feature_selection_config, "enabled", False)
            and getattr(feature_selection_config, "method", None) == "sfs"
            and _config_uses_group_cv(getattr(feature_selection_config, "cv", None))
            and hasattr(sfs, "cv")
            and not cv_uses_groups(sfs.cv)
        ):
            sfs.cv = get_cv_splitter(feature_selection_config.cv, require_groups=False)

    fit_params: dict[str, Any] = {}
    if groups_train is not None:
        if calibrated and _config_uses_group_cv(
            getattr(calibration_config, "cv", None)
        ):
            cal_cv = get_cv_splitter(calibration_config.cv, require_groups=False)
            estimator.cv = _CVWithGroups(cal_cv, groups_train)
        if search_cv and _config_uses_group_cv(getattr(tuning_config, "cv", None)):
            fit_params["groups"] = groups_train

        if sfs is not None and _config_uses_group_cv(
            getattr(feature_selection_config, "cv", None)
        ):
            fit_params["fs__groups"] = groups_train

        if isinstance(pipeline, Pipeline) and "scaler" in pipeline.named_steps:
            scaler_step = pipeline.named_steps["scaler"]
            if "groups" in inspect.signature(scaler_step.fit).parameters:
                fit_params["scaler__groups"] = groups_train
        if isinstance(pipeline, Pipeline) and "erasure" in pipeline.named_steps:
            erasure_step = pipeline.named_steps["erasure"]
            if "groups" in inspect.signature(erasure_step.fit).parameters:
                fit_params["erasure__groups"] = groups_train
        if isinstance(pipeline, Pipeline) and "clf" in pipeline.named_steps:
            classifier_step = pipeline.named_steps["clf"]
            if "groups" in inspect.signature(classifier_step.fit).parameters:
                fit_params["clf__groups"] = groups_train

    if (
        sample_weight is not None
        and isinstance(pipeline, Pipeline)
        and "clf" in pipeline.named_steps
    ):
        clf_step = pipeline.named_steps["clf"]
        if "sample_weight" in inspect.signature(clf_step.fit).parameters:
            fit_params["clf__sample_weight"] = sample_weight

    estimator.fit(X_train, y_train, **fit_params)


def _config_uses_group_cv(cv_config: Any) -> bool:
    """Return True when a decoding CV config needs subject groups."""
    strategy = getattr(cv_config, "strategy", None)
    return isinstance(strategy, str) and strategy.lower() in GROUP_CV_STRATEGIES


def extract_feature_importances(
    estimator: BaseEstimator,
    spec: Any,
    fs_enabled: bool = False,
    search_enabled: bool = False,
    calibration_enabled: bool = False,
) -> np.ndarray | None:
    """
    Extract and aggregate feature importances or coefficients from a fitted model.

    This function drills down through potential pipeline wrappers (calibration,
    tuning, and feature selection) to reach the base estimator. It then
    delegates to `_get_raw_importance` and handles masking if feature
    selection was applied.

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted estimator, potentially wrapped in several layers.
    spec : EstimatorSpec
        The model capability registry entry defining how to extract weights.
    fs_enabled : bool, optional
        Whether feature selection (and thus a projection mask) was used.
    search_enabled : bool, optional
        Whether hyperparameter search (and thus `best_estimator_`) was used.
    calibration_enabled : bool, optional
        Whether calibration (and thus `.estimator`) was used.

    Returns
    -------
    np.ndarray, optional
        A vector of importances aligned with the input features, or None
        if the model does not support weight extraction (e.g., k-NN).
    """
    # 1. Drill down through wrappers
    if calibration_enabled:
        calibrated = getattr(estimator, "calibrated_classifiers_", None)
        if calibrated:
            fold_imps = []
            for calibrated_classifier in calibrated:
                base = getattr(calibrated_classifier, "estimator", None)
                imp = extract_feature_importances(
                    base,
                    spec,
                    fs_enabled=fs_enabled,
                    search_enabled=search_enabled,
                    calibration_enabled=False,
                )
                if imp is not None:
                    fold_imps.append(imp)
            if fold_imps and all(imp.shape == fold_imps[0].shape for imp in fold_imps):
                return np.mean(np.vstack(fold_imps), axis=0)
            return None

        estimator = getattr(estimator, "estimator", estimator)

    if search_enabled:
        estimator = getattr(estimator, "best_estimator_", estimator)

    # 2. Handle Pipeline (Scaler + [FS] + Clf)
    if hasattr(estimator, "named_steps"):
        clf = estimator.named_steps["clf"]
        if fs_enabled:
            fs = estimator.named_steps["fs"]
            raw_imp = _get_raw_importance(clf, spec)
            if raw_imp is not None:
                mask = fs.get_support()
                full_imp = np.zeros_like(mask, dtype=float)
                full_imp[mask] = raw_imp
                return full_imp
            return None
        estimator = clf

    # 3. Direct extraction from base estimator
    return _get_raw_importance(estimator, spec)


def _get_raw_importance(estimator: Any, spec: Any) -> np.ndarray | None:
    """
    Internal helper to extract importance from the base estimator.

    Handles sparse coefficients and multiclass magnitude aggregation.
    """
    imp_type = spec.importance[0]

    if imp_type == "coefficients":
        if not hasattr(estimator, "coef_"):
            return None
        vals = estimator.coef_
        # Zero-guesswork sparse handling
        if spec.is_sparse_capable and hasattr(vals, "toarray"):
            vals = vals.toarray()

        if vals.ndim > 1:
            # Aggregate across classes (multiclass). Note: binary LDA or
            # LogisticRegression often return shape (1, n_features).
            return np.mean(np.abs(vals), axis=0)
        return np.abs(vals)

    if imp_type == "feature_importances":
        if not hasattr(estimator, "feature_importances_"):
            return None
        return estimator.feature_importances_

    return None


def compute_metric_safe(
    scorer: Callable,
    y_true: np.ndarray,
    y_est: np.ndarray,
    is_multiclass: bool,
    is_proba: bool = False,
    name: str | None = None,
) -> float | np.ndarray:
    """
    Handles Standard, Sliding, and Generalizing decoding results efficiently.

    Parameters
    ----------
    scorer : Callable
        Scikit-learn compatible scoring function.
    y_true : np.ndarray
        Ground truth labels.
    y_est : np.ndarray
        Predictions or probabilities (can be 2D, 3D, or 4D).
    is_multiclass : bool
        Whether the task is multiclass.
    is_proba : bool
        Whether y_est contains probabilities.

    Returns
    -------
    float or np.ndarray
        Single score or temporal score matrix/vector.
    """
    # 1. Configuration Setup
    kwargs = {"multi_class": "ovr"} if (is_proba and is_multiclass) else {}

    # 2. Handle Binary Probability Slicing (Positive class only)
    if (
        is_proba
        and not is_multiclass
        and y_est is not None
        and y_est.ndim >= 2
        and y_est.shape[1] == 2
    ):
        y_est = y_est[:, 1, ...]

    # 3. Shape Analysis
    if y_est is None:
        return np.nan

    slice_ndim = 2 if (is_proba and is_multiclass) else 1
    n_temporal_dims = y_est.ndim - slice_ndim

    # 4. Temporal Dispatch
    if n_temporal_dims == 0:  # Standard decoding
        return float(scorer(y_true, y_est, **kwargs))

    if n_temporal_dims == 1:  # Sliding decoding (n_times,)
        if name == "accuracy" and not is_proba:
            return np.mean(y_true[:, None] == y_est, axis=0)

        return np.array(
            [
                float(scorer(y_true, y_est[..., t], **kwargs))
                for t in range(y_est.shape[-1])
            ]
        )

    if n_temporal_dims == 2:  # Generalizing decoding
        if name == "accuracy" and not is_proba:
            return np.mean(y_true[:, None, None] == y_est, axis=0)

        n_tr, n_te = y_est.shape[-2], y_est.shape[-1]
        results = np.zeros((n_tr, n_te))
        for tr in range(n_tr):
            for te in range(n_te):
                results[tr, te] = float(scorer(y_true, y_est[..., tr, te], **kwargs))
        return results

    raise ValueError(f"Unsupported y_est shape for scoring: {y_est.shape}")


def extract_metadata(
    estimator: BaseEstimator,
    spec: Any,
    feature_selection_config: Any,
    search_enabled: bool = False,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Metadata extraction.

    Parameters
    ----------
    estimator : BaseEstimator
        The fitted estimator or pipeline.
    spec : EstimatorSpec
        Model registry entry.
    feature_selection_config : Any
        FS configuration.
    search_enabled : bool
        Whether search wrapper is active.
    feature_names : list of str, optional
        Original feature names.

    Returns
    -------
    Dict[str, Any]
        Aggregated fold metadata.
    """
    meta = {}

    # 1. Search Diagnostics
    if search_enabled:
        meta["best_params"] = getattr(estimator, "best_params_", {})
        meta["best_score"] = getattr(estimator, "best_score_", np.nan)
        meta["best_index"] = getattr(estimator, "best_index_", -1)
        if hasattr(estimator, "cv_results_"):
            meta["search_results"] = compact_search_results(estimator)
        estimator = getattr(estimator, "best_estimator_", estimator)

    # 2. Pipeline / Feature Selection Diagnostics
    if feature_selection_config.enabled and hasattr(estimator, "named_steps"):
        fs_step = estimator.named_steps.get("fs")
        clf_step = estimator.named_steps.get("clf")

        if fs_step is not None:
            mask = fs_step.get_support()
            indices = np.flatnonzero(mask)
            n_feat = len(mask)

            actual_names = (
                feature_names
                if (feature_names and len(feature_names) == n_feat)
                else [f"feature_{i}" for i in range(n_feat)]
            )

            meta.update(
                {
                    "feature_selection_method": feature_selection_config.method,
                    "selected_features": mask,
                    "selected_feature_indices": indices,
                    "selected_feature_names": [actual_names[i] for i in indices],
                    "feature_names": actual_names,
                }
            )

            if feature_selection_config.method == "sfs" and hasattr(
                fs_step, "ranking_"
            ):
                meta["selection_order"] = fs_step.ranking_
            elif feature_selection_config.method == "k_best" and hasattr(
                fs_step, "scores_"
            ):
                meta["feature_scores"] = fs_step.scores_

        elif clf_step is not None and hasattr(clf_step, "estimators_"):
            # Temporal (sliding/generalizing) wrapper: feature selection was
            # pushed one level down into each per-timepoint base-classifier
            # clone (see Experiment._prepare_estimator's "temporal" branch),
            # so there is no single top-level "fs" step - there are up to
            # n_times of them, one per `clf_step.estimators_[t]`. Aggregate
            # into a per-feature selection frequency across every timepoint
            # this fold fit, reusing the "selected_features" key so
            # ExperimentResult.get_feature_stability()'s existing cross-fold
            # averaging picks it up unchanged - the result is then a
            # frequency across both folds and timepoints.
            per_timepoint_masks = [
                sub.named_steps["fs"].get_support()
                for sub in clf_step.estimators_
                if hasattr(sub, "named_steps") and "fs" in sub.named_steps
            ]
            if per_timepoint_masks:
                per_timepoint_stack = np.vstack(per_timepoint_masks)
                frequency = np.mean(per_timepoint_stack, axis=0)
                n_feat = len(frequency)
                actual_names = (
                    feature_names
                    if (feature_names and len(feature_names) == n_feat)
                    else [f"feature_{i}" for i in range(n_feat)]
                )
                meta.update(
                    {
                        "feature_selection_method": feature_selection_config.method,
                        "selected_features": frequency,
                        # Raw (n_times, n_features) boolean mask, one row per
                        # per-timepoint base-classifier clone in this fold -
                        # preserved (not just the time-averaged "frequency"
                        # above) so callers can recover exactly which
                        # features were selected at one specific timepoint
                        # (e.g. a peak-accuracy latency), not just the
                        # across-time selection rate. See
                        # ExperimentResult.get_selected_features_per_timepoint.
                        "selected_features_per_timepoint": per_timepoint_stack,
                        "feature_names": actual_names,
                    }
                )

        if clf_step is not None:
            estimator = clf_step

    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("clf", estimator)

    # 3. Custom Artifacts (Structural Type Check via Protocol)
    if isinstance(estimator, NeuralTrainable):
        meta["artifacts"] = estimator.get_artifact_metadata()

    return meta


def compact_search_results(estimator: BaseEstimator) -> list[dict[str, Any]]:
    """
    Return compact search diagnostics with pre-standardized arrays.

    Converts Scikit-learn's cv_results_ to a serializable list of records.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator (e.g., GridSearchCV) containing cv_results_.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, where each dictionary represents a single
        hyperparameter combination and its scores.
    """
    cv_res = estimator.cv_results_
    params = cv_res["params"]

    ranks = (
        np.asarray(cv_res["rank_test_score"]) if "rank_test_score" in cv_res else None
    )
    means = (
        np.asarray(cv_res["mean_test_score"], dtype=float)
        if "mean_test_score" in cv_res
        else None
    )
    stds = (
        np.asarray(cv_res["std_test_score"], dtype=float)
        if "std_test_score" in cv_res
        else None
    )

    results = []
    for idx, p in enumerate(params):
        row = {"candidate": idx, "params": dict(p)}
        if ranks is not None:
            row["rank"] = int(ranks[idx])
        if means is not None:
            row["mean"] = float(means[idx])
        if stds is not None:
            row["std"] = float(stds[idx])
        results.append(row)

    return results


def metadata_slice(
    metadata_dict: dict[str, np.ndarray] | None,
    indices: np.ndarray,
) -> dict[str, list[Any]] | None:
    """
    Slicing of pre-converted metadata.

    Parameters
    ----------
    metadata_dict : dict, optional
        Dictionary of numpy arrays.
    indices : np.ndarray
        Indices to select.

    Returns
    -------
    dict, optional
        Slicing result as a serializable dictionary of lists.
    """
    if metadata_dict is None:
        return None
    return {k: v[indices].tolist() for k, v in metadata_dict.items()}


def warning_records_to_dict(
    stage: str, warning_records: Sequence[Any]
) -> list[dict[str, Any]]:
    """
    Return serializable warning records captured in one fold stage.
    """
    return [
        {
            "stage": stage,
            "category": record.category.__name__,
            "message": str(record.message),
        }
        for record in warning_records
    ]
