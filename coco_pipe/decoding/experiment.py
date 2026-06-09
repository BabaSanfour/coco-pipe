"""
Decoding Experiment
===================
Main executor for decoding experiments.
"""

import atexit
import logging
import time
from collections import defaultdict
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

from ..report.provenance import get_environment_info
from ._constants import CLASSICAL_FAMILIES, GROUP_CV_STRATEGIES, RESULT_SCHEMA_VERSION
from ._engine import GroupedSequentialFeatureSelector, fit_and_score_fold
from ._metrics import get_metric_spec
from ._splitters import get_cv_splitter
from .configs import ExperimentConfig
from .registry import (
    _get_val,
    get_estimator_cls,
    get_selector_capabilities,
    resolve_estimator_spec,
)
from .result import ExperimentResult

logger = logging.getLogger(__name__)


class Experiment:
    """
    Main executor for decoding experiments.

    Parameters
    ----------
    config : ExperimentConfig
        The complete configuration for the experiment.

    Examples
    --------
    >>> from coco_pipe.decoding import Experiment, ExperimentConfig
    >>> config = ExperimentConfig(
    ...     task="classification", models={"lr": {"kind": "classical"}}
    ... )
    >>> exp = Experiment(config)

    See Also
    --------
    ExperimentResult : Container for experiment outputs.
    get_cv_splitter : Factory for cross-validation splitters.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.result_: Optional[ExperimentResult] = None
        self._model_specs: Dict[str, Any] = {}
        self._model_capabilities: Dict[str, Any] = {}
        self._propagate_random_state()
        self._validate_config()

    def _validate_config(self):
        """Perform comprehensive runtime validation of the configuration."""
        task = self.config.task
        if self.config.tuning.enabled and not self.config.grids:
            logger.warning(
                "Hyperparameter tuning is enabled but no 'grids' are defined "
                "in the config."
            )

        if self.config.calibration.enabled and task != "classification":
            raise ValueError("calibration is only available for classification.")

        if task == "regression" and "stratified" in self.config.cv.strategy:
            raise ValueError(
                f"CV strategy '{self.config.cv.strategy}' is invalid "
                "for regression tasks."
            )

        # 1. Inner CV Grouping Validation (Leakage Guard)
        if self.config.cv.strategy in GROUP_CV_STRATEGIES:
            targets = []
            if self.config.tuning.enabled:
                targets.append(
                    (
                        "tuning.cv",
                        self.config.tuning.cv,
                        self.config.tuning.allow_nongroup_inner_cv,
                    )
                )
            fs = self.config.feature_selection
            if fs.enabled and fs.method == "sfs":
                targets.append(
                    ("feature_selection.cv", fs.cv, fs.allow_nongroup_inner_cv)
                )
            cal = self.config.calibration
            if cal.enabled:
                targets.append(("calibration.cv", cal.cv, cal.allow_nongroup_inner_cv))

            for name, cv_cfg, allowed in targets:
                if (
                    cv_cfg
                    and cv_cfg.strategy not in GROUP_CV_STRATEGIES
                    and not allowed
                ):
                    raise ValueError(
                        f"Outer CV strategy is group-based, but {name} strategy "
                        f"'{cv_cfg.strategy}' is not. This leads to data leakage. "
                        "Set allow_nongroup_inner_cv=True if this is intentional."
                    )

        # Validate FS scoring if explicitly set
        fs_scoring = self.config.feature_selection.scoring
        if fs_scoring and get_metric_spec(fs_scoring).task != task:
            raise ValueError(
                f"Feature selection scoring '{fs_scoring}' is incompatible with "
                f"task '{task}'."
            )

        for name, model_cfg in self.config.models.items():
            spec = resolve_estimator_spec(model_cfg)
            caps = spec.to_capabilities()
            self._model_specs[name] = spec
            self._model_capabilities[name] = caps
            if not caps.supports_task(task):
                raise ValueError(f"Model '{name}' does not support task '{task}'.")

            # 2. Metric Compatibility Check
            has_proba = (
                caps.has_response("predict_proba") or self.config.calibration.enabled
            )
            has_score = caps.has_response("decision_function")
            for metric in self.config.get_all_evaluation_metrics():
                m_spec = get_metric_spec(metric)
                if m_spec.task != task:
                    raise ValueError(
                        f"Metric '{metric}' is for {m_spec.task} but experiment task "
                        f"is {task}."
                    )
                if m_spec.response_method == "proba" and not has_proba:
                    raise ValueError(
                        f"Metric '{metric}' requires probabilities, but model "
                        f"'{name}' doesn't provide them (and calibration is disabled)."
                    )
                if m_spec.response_method == "proba_or_score" and not (
                    has_proba or has_score
                ):
                    raise ValueError(
                        f"Metric '{metric}' requires probabilities or decision scores, "
                        f"but model '{name}' provides neither."
                    )

    def _prepare_estimator(self, model_name: str, model_config: Any) -> BaseEstimator:
        """Orchestrate the creation of the full Estimator Pipeline."""

        full_est = self._instantiate_model(model_name, model_config)
        spec = self._model_specs.get(model_name) or resolve_estimator_spec(model_config)
        steps = []

        # Classical models on tabular/embedding data support standard preprocessing
        allow_prep = spec.family in CLASSICAL_FAMILIES and any(
            k in {"tabular_2d", "embedding_2d", "tabular", "embeddings"}
            for k in spec.input_kinds
        )

        if self.config.use_scaler and allow_prep:
            steps.append(("scaler", StandardScaler()))

        if self.config.feature_selection.enabled and allow_prep:
            fs_step = self._create_fs_step(full_est)
            if fs_step:
                steps.append(fs_step)
        elif self.config.feature_selection.enabled and not allow_prep:
            raise ValueError(
                f"Feature selection is only valid for classical 2D tabular "
                f"inputs. Model '{model_name}' uses {spec.input_kinds} data."
            )

        steps.append(("clf", full_est))

        if (
            self.config.feature_selection.enabled
            and self.config.tuning.enabled
            and self.config.grids
        ):
            cachedir = mkdtemp()
            atexit.register(lambda: rmtree(cachedir, ignore_errors=True))
            est = Pipeline(steps, memory=cachedir)
        else:
            est = Pipeline(steps)

        if (
            self.config.tuning.enabled
            and self.config.grids
            and model_name in self.config.grids
        ):
            est = self._wrap_with_tuning(model_name, est)

        if self.config.calibration.enabled:
            from sklearn.calibration import CalibratedClassifierCV

            cal_cv = get_cv_splitter(self.config.calibration.cv, require_groups=False)
            est = CalibratedClassifierCV(
                estimator=est,
                method=self.config.calibration.method,
                cv=cal_cv,
                n_jobs=self.config.calibration.n_jobs or self.config.n_jobs,
            )
        return est

    def _propagate_random_state(self):
        """Ensure the global random state is distributed to all config sub-objects."""
        seed = self.config.random_state
        if seed is None:
            return

        self._inject_seed(self.config.cv, seed)
        self._inject_seed(self.config.feature_selection, seed + 1)
        self._inject_seed(self.config.tuning, seed + 2)
        self._inject_seed(self.config.calibration, seed + 3)

        # 2. Model seeds
        model_names = sorted(self.config.models.keys())
        from numpy.random import SeedSequence

        ss = SeedSequence(seed + 4)
        model_seeds = ss.spawn(len(model_names))
        for name, m_ss in zip(model_names, model_seeds):
            self._inject_seed(self.config.models[name], int(m_ss.generate_state(1)[0]))

    def _inject_seed(self, cfg: Any, seed: int):
        """Safely inject a random seed into a config object if it supports it."""
        if hasattr(cfg, "random_state"):
            cfg.random_state = seed
        if hasattr(cfg, "cv") and cfg.cv and hasattr(cfg.cv, "random_state"):
            cfg.cv.random_state = seed

        # Classical parameters dictionary
        if getattr(cfg, "kind", None) == "classical" and hasattr(cfg, "params"):
            if resolve_estimator_spec(cfg).supports_random_state:
                cfg.params["random_state"] = seed

        # Recursion into sub-components (backbone, head, base)
        for attr in ("backbone", "head", "base"):
            if hasattr(cfg, attr):
                self._inject_seed(getattr(cfg, attr), seed)

    def _instantiate_model(self, model_name: str, config: Any) -> BaseEstimator:
        """Create a concrete scikit-learn estimator instance from a model config."""
        # 1. Use the pre-resolved spec for explicit dispatch
        spec = self._model_specs.get(model_name) or resolve_estimator_spec(config)

        if spec.family in CLASSICAL_FAMILIES:
            est_cls = get_estimator_cls(spec.name)
            if hasattr(config, "params"):
                params = config.params
            elif isinstance(config, dict):
                params = config.get("params", {})
            else:
                params = config.model_dump(exclude={"method", "kind"})
            try:
                return est_cls(**params)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate model '{model_name}': {e}"
                ) from e

        if spec.family == "foundation":
            from .fm_hub import build_foundation_model

            return build_foundation_model(config)

        if spec.family == "temporal":
            # wrapper is 'sliding' or 'generalizing'
            wrapper = _get_val(config, "wrapper")
            method = (
                "SlidingEstimator" if wrapper == "sliding" else "GeneralizingEstimator"
            )
            est_cls = get_estimator_cls(method)

            if hasattr(config, "model_dump"):
                params = config.model_dump(exclude={"kind", "wrapper", "base"})
            elif isinstance(config, dict):
                params = {
                    k: v
                    for k, v in config.items()
                    if k not in {"kind", "wrapper", "base"}
                }
            else:
                params = {}

            params["base_estimator"] = self._prepare_estimator(
                f"{model_name}_base", _get_val(config, "base")
            )
            try:
                return est_cls(**params)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate model '{model_name}': {e}"
                ) from e

        # Fallback for other registry-based estimators
        method = _get_val(config, "method")
        est_cls = get_estimator_cls(method)
        if hasattr(config, "model_dump"):
            params = config.model_dump(exclude={"method", "kind"})
        elif isinstance(config, dict):
            params = {k: v for k, v in config.items() if k not in {"method", "kind"}}
        else:
            params = {}

        if "base_estimator" in params:
            params["base_estimator"] = self._prepare_estimator(
                f"{model_name}_base", _get_val(config, "base")
            )
        return est_cls(**params)

    def _create_fs_step(self, estimator: BaseEstimator) -> Optional[tuple]:
        """Create a feature selection step compatible with the chosen model."""
        fs_conf = self.config.feature_selection
        if fs_conf.method == "k_best":
            score_func = (
                f_classif if self.config.task == "classification" else f_regression
            )
            return (
                "fs",
                SelectKBest(score_func=score_func, k=fs_conf.n_features or "all"),
            )
        if fs_conf.method == "sfs":
            cv = get_cv_splitter(fs_conf.cv, require_groups=False)
            scoring = (
                fs_conf.scoring or self.config.tuning.scoring or self.config.metrics[0]
            )
            sfs = GroupedSequentialFeatureSelector(
                estimator=clone(estimator),
                n_features_to_select=fs_conf.n_features,
                direction=fs_conf.direction,
                cv=cv,
                scoring=scoring,
                n_jobs=self.config.n_jobs,
            )
            return ("fs", sfs)
        return None

    def _wrap_with_tuning(self, name: str, estimator: BaseEstimator) -> BaseEstimator:
        """Wrap an estimator with hyperparameter search (GridSearch/RandomSearch)."""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        grid = self.config.grids[name]
        mapped = {k if "__" in k else f"clf__{k}": v for k, v in grid.items()}
        valid = estimator.get_params(deep=True)
        invals = [k for k in mapped if k not in valid]
        if invals:
            raise ValueError(f"Invalid tuning keys for '{name}': {invals}")
        cv = get_cv_splitter(self.config.tuning.cv, require_groups=False)
        kwargs = {
            "estimator": estimator,
            "cv": cv,
            "scoring": self.config.tuning.scoring or self.config.metrics[0],
            "n_jobs": self.config.tuning.n_jobs,
            "refit": True,
        }
        if self.config.tuning.search_type == "grid":
            return GridSearchCV(param_grid=mapped, **kwargs)
        return RandomizedSearchCV(
            param_distributions=mapped,
            n_iter=self.config.tuning.n_iter,
            random_state=self.config.tuning.random_state,
            **kwargs,
        )

    def run(
        self,
        X: np.ndarray,
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Sequence[Any]]]] = None,
        observation_level: str = "sample",
        inferential_unit: Optional[str] = None,
        time_axis: Optional[Sequence[Any]] = None,
    ) -> ExperimentResult:
        """
        Execute the complete decoding experiment pipeline.

        This method orchestrates the full scientific workflow:
        1. Resolves and validates data hierarchy (metadata, groups, sample IDs).
        2. Aligns temporal dimensions and feature names.
        3. Performs model-by-model evaluation using stratified/grouped cross-validation.
        4. Aggregates results into a unified ExperimentResult object.
        5. Performs statistical assessment (permutations/bootstrapping) if enabled.

        Parameters
        ----------
        X : np.ndarray
            The input data. Can be 2D (samples x features) or 3D temporal
            (samples x sensors x time).
        y : Union[pd.Series, np.ndarray]
            The target labels or values.
        groups : Union[pd.Series, np.ndarray], optional
            Grouping labels for grouped cross-validation (e.g., subject IDs).
        feature_names : Sequence[str], optional
            Human-readable names for features. Auto-generated if None.
        sample_ids : Sequence[Any], optional
            Unique identifiers for each sample. Auto-generated if None.
        sample_metadata : Union[pd.DataFrame, Dict], optional
            Additional scientific context for each sample (BIDS-like).
        observation_level : str, default='sample'
            Level of the input rows ('sample' or 'epoch').
        inferential_unit : str, optional
            The level of statistical independence ('sample' or 'subject').
        time_axis : Sequence[Any], optional
            Scientific time points for 3D temporal data.

        Returns
        -------
        ExperimentResult
            A container holding all raw results, metrics, and diagnostics.

        Raises
        ------
        ValueError
            If input lengths mismatch, data is empty, or configuration
            is scientifically invalid (e.g. regression with stratification).

        Examples
        --------
        >>> from coco_pipe.decoding import Experiment, ExperimentConfig
        >>> config = ExperimentConfig(
        ...     task="classification", models={"lr": {"kind": "classical"}}
        ... )
        >>> result = Experiment(config).run(X, y)

        See Also
        --------
        ExperimentResult.get_predictions : Tidy prediction accessor.
        """
        start_time = time.time()
        logger.info(f"Starting Experiment: Task={self.config.task}")
        X, y = np.asarray(X), np.asarray(y)
        if len(X) == 0:
            raise ValueError("X is empty.")
        if len(y) != len(X):
            raise ValueError("Length mismatch between X and y.")

        # 1. Scientific Guard: Double-Normalization Warning
        if self.config.use_scaler and X.ndim == 2:
            # Simple heuristic: if means are near 0 and stds are near 1, warn.
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
            if np.all(np.abs(means) < 1e-3) and np.all(np.abs(stds - 1.0) < 1e-2):
                logger.warning(
                    "Input data X appears to be already normalized "
                    "(means ~0, stds ~1). Enabling 'use_scaler' will "
                    "result in redundant double-normalization."
                )

        self._feature_names = self._resolve_feature_names(X, feature_names)
        self._sample_ids = self._resolve_sample_ids(len(X), sample_ids)
        if observation_level not in {"sample", "epoch"}:
            raise ValueError("observation_level must be 'sample' or 'epoch'.")
        self._observation_level = observation_level
        self._sample_metadata, groups = self._resolve_metadata_and_groups(
            len(X), sample_metadata, groups
        )

        # 3. Resolve Inferential Unit (Level of statistical independence)
        if inferential_unit is not None:
            self._inferential_unit = inferential_unit
        elif (
            observation_level == "epoch" and "Subject" in self._sample_metadata.columns
        ):
            self._inferential_unit = "subject"
        else:
            self._inferential_unit = "sample"

        # 4. Resolve Time Axis
        if X.ndim == 3:
            if time_axis is None:
                self._time_axis = np.arange(X.shape[-1])
            else:
                self._time_axis = np.asarray(time_axis)
                if len(self._time_axis) != X.shape[-1]:
                    raise ValueError(
                        f"time_axis length mismatch: {len(self._time_axis)} vs "
                        f"{X.shape[-1]}"
                    )
        else:
            self._time_axis = np.asarray(time_axis) if time_axis is not None else None

        # 5. Input Rank Capability Guard
        rank = "3d_temporal" if X.ndim == 3 else "2d"

        # 6. Group Validation: Early check before entering model loop
        if groups is None:
            from ._constants import GROUP_CV_STRATEGIES

            cv_configs = [self.config.cv]
            if self.config.tuning.enabled and self.config.tuning.cv:
                cv_configs.append(self.config.tuning.cv)
            if (
                self.config.feature_selection.enabled
                and self.config.feature_selection.cv
            ):
                cv_configs.append(self.config.feature_selection.cv)
            if self.config.calibration.enabled and self.config.calibration.cv:
                cv_configs.append(self.config.calibration.cv)

            for cv_conf in cv_configs:
                if cv_conf.strategy in GROUP_CV_STRATEGIES:
                    raise ValueError(
                        f"Strategy '{cv_conf.strategy}' requires groups, but none "
                        "were provided."
                    )

        for name, caps in self._model_capabilities.items():
            if rank not in caps.input_ranks:
                raise ValueError(f"Model '{name}' doesn't support rank '{rank}'.")
        if self.config.feature_selection.enabled:
            sel = get_selector_capabilities(self.config.feature_selection.method)
            if rank not in sel.input_ranks:
                raise ValueError(
                    f"FS method '{sel.method}' doesn't support rank '{rank}'."
                )
        if self.config.task == "classification" and type_of_target(y) == "continuous":
            raise ValueError("Task is 'classification' but target is 'continuous'.")

        for name, cfg in self.config.models.items():
            label = getattr(cfg, "method", getattr(cfg, "kind", "Unknown"))
            logger.info(f"Evaluating Model: {name} ({label})")
            try:
                # 1. Resolve Spec & Capabilities
                from .registry import resolve_estimator_spec

                spec = resolve_estimator_spec(cfg)

                # 2. Parallelism Safety
                is_fm = spec.family in {"foundation", "neural"}
                model_n_jobs = 1 if is_fm else self.config.n_jobs

                est = self._prepare_estimator(name, cfg)
                self.results[name] = self._cross_validate(
                    est,
                    X,
                    y,
                    groups,
                    self._sample_ids,
                    self._sample_metadata,
                    n_jobs=model_n_jobs,
                    spec=spec,
                    model_name=name,
                )
            except Exception as e:
                logger.error(f"Failed model '{name}': {e}", exc_info=True)
                self.results[name] = {"error": str(e), "status": "failed"}

        logger.info(f"Experiment Completed in {time.time() - start_time:.2f}s")
        res_obj = ExperimentResult(
            self.results,
            config=self.config.model_dump(),
            meta=self._build_result_meta(X, self._time_axis),
        )

        if self.config.evaluation.enabled:
            from .stats import run_statistical_assessment

            assessment = run_statistical_assessment(
                res_obj,
                self.config,
                X,
                y,
                groups,
                self._sample_ids,
                self._sample_metadata,
                self._feature_names,
                self._time_axis,
                self._observation_level,
                self._inferential_unit,
            )
            res_obj.meta["statistical_assessment"] = assessment["meta"]
            for m_name, m_res in res_obj.raw.items():
                if "error" in m_res:
                    continue
                m_res["statistical_assessment"] = [
                    r for r in assessment["rows"] if r.get("Model") == m_name
                ]
                if m_name in assessment["nulls"]:
                    m_res["statistical_nulls"] = assessment["nulls"][m_name]
        return res_obj

    def _cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        sample_ids: np.ndarray,
        sample_metadata: pd.DataFrame,
        n_jobs: int = 1,
        spec: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform parallel cross-validation for a single estimator."""
        cv = get_cv_splitter(self.config.cv, groups=groups, y=y)
        splits = list(cv.split(X, y, groups))

        for train_idx, test_idx in splits:
            _validate_fold_integrity(y[train_idx], y[test_idx], spec.task)

        est = clone(estimator)

        meta_dict = None
        if sample_metadata is not None:
            meta_dict = {
                col: sample_metadata[col].values for col in sample_metadata.columns
            }

        # 4. Execute Parallel CV
        parallel = joblib.Parallel(n_jobs=n_jobs, verbose=self.config.verbose)
        results = parallel(
            joblib.delayed(fit_and_score_fold)(
                clone(est),
                X,
                y,
                groups,
                sample_ids,
                meta_dict,
                train_idx,
                test_idx,
                metrics=self.config.get_all_evaluation_metrics(),
                feature_selection_config=self.config.feature_selection,
                calibration_config=self.config.calibration,
                spec=spec,
                tuning_config=self.config.tuning,
                feature_names=self._feature_names,
                search_enabled=(
                    self.config.tuning.enabled
                    and self.config.grids is not None
                    and model_name in self.config.grids
                ),
                force_serial=(n_jobs == 1),
            )
            for train_idx, test_idx in splits
        )

        fold_scores = defaultdict(list)
        f_idx, f_preds, f_imps, f_meta, f_splits, f_diags = [], [], [], [], [], []
        for res in results:
            f_idx.append(res["test_idx"])
            f_preds.append(res["preds"])
            f_imps.append(res["importance"])
            f_meta.append(res.get("metadata", {}))
            f_splits.append(res["split"])
            f_diags.append(res.get("diagnostics", {}))
            for m, s in res["scores"].items():
                fold_scores[m].append(s)

        metrics = {}
        for m, s in fold_scores.items():
            if np.isnan(s).any():
                logger.warning(
                    f"NaN score detected in one or more folds for metric '{m}'. "
                    "This usually indicates a model failure or degenerate test fold."
                )
            metrics[m] = {"mean": np.nanmean(s), "std": np.nanstd(s), "folds": s}
        valid_imps = [f for f in f_imps if f is not None]
        agg_imp = None
        if valid_imps and all(imp.shape == valid_imps[0].shape for imp in valid_imps):
            stack = np.vstack(valid_imps)
            agg_imp = {
                "mean": np.mean(stack, axis=0),
                "std": np.std(stack, axis=0),
                "raw": stack,
                "feature_names": self._feature_names
                if len(self._feature_names) == stack.shape[1]
                else [f"feature_{idx}" for idx in range(stack.shape[1])],
            }

        return {
            "status": "success",
            "metrics": metrics,
            "predictions": f_preds,
            "indices": f_idx,
            "importances": agg_imp,
            "metadata": f_meta,
            "splits": f_splits,
            "diagnostics": f_diags,
        }

    @staticmethod
    def _resolve_sample_ids(n: int, ids: Optional[Sequence[Any]]) -> np.ndarray:
        """Ensure sample IDs are provided and have correct length."""
        if ids is None:
            return np.arange(n)
        ids = np.asarray(ids)
        if len(ids) != n:
            raise ValueError(f"sample_ids length mismatch: {len(ids)} vs {n}")
        if len(pd.unique(ids)) != n:
            raise ValueError("sample_ids must be unique.")
        return ids

    def _resolve_metadata_and_groups(
        self,
        n: int,
        meta_in: Optional[Union[pd.DataFrame, Dict[str, Sequence[Any]]]],
        groups_in: Optional[np.ndarray],
    ) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Validate metadata and extract cross-validation groups if required."""
        # 1. Standardize Metadata to DataFrame
        if meta_in is None:
            meta = pd.DataFrame(index=range(n))
        else:
            meta = pd.DataFrame(meta_in).reset_index(drop=True)
            meta.columns = [str(c).capitalize() for c in meta.columns]
            if len(meta) != n:
                raise ValueError(f"sample_metadata length mismatch: {len(meta)} vs {n}")

        # 2. Scientific Guard: Metadata Requirements
        # We must track subject and session to ensure independent validation
        # and prevent pseudoreplication, especially for epoch-level data.
        if meta_in is not None:
            missing = [c for c in ["Subject", "Session"] if c not in meta.columns]
            if missing:
                raise ValueError(
                    f"sample_metadata must include Subject and Session for "
                    f"proper independence tracking. Missing: {missing}"
                )

        # 2. Resolve Groups
        gv = None
        key = self.config.cv.group_key
        has_grouped_cv = (
            self.config.cv.strategy in GROUP_CV_STRATEGIES
            or (
                self.config.tuning.enabled
                and self.config.tuning.cv.strategy in GROUP_CV_STRATEGIES
            )
            or (
                self.config.calibration.enabled
                and self.config.calibration.cv.strategy in GROUP_CV_STRATEGIES
            )
        )

        # Case A: Explicit groups array provided
        if groups_in is not None:
            gv = np.asarray(groups_in)
            if len(gv) != n:
                raise ValueError(f"groups length mismatch: {len(gv)} vs {n}")
            if key is not None:
                meta[key] = gv

        # Case B: Extract from metadata using group_key
        elif key is not None:
            if key not in meta.columns:
                raise ValueError(f"group_key '{key}' not found in sample_metadata.")
            gv = meta[key].to_numpy()

        # Validation: Grouped strategies with only 1 group will fail in sklearn
        if has_grouped_cv:
            if gv is None:
                raise ValueError(
                    f"CV strategy '{self.config.cv.strategy}' requires groups "
                    "via 'groups' parameter or 'group_key' in config."
                )
            unique_groups = len(pd.unique(gv))
            if unique_groups < 2:
                raise ValueError(
                    f"Grouped CV requires at least 2 unique groups, but found "
                    f"only {unique_groups}. Check your sample_metadata or "
                    "groups array."
                )
        return meta, gv

    def _build_result_meta(
        self, X: np.ndarray, t_axis: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        meta = get_environment_info()
        meta.update(
            {
                "tag": self.config.tag,
                "task": self.config.task,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
                "observation_level": self._observation_level,
                "inferential_unit": self._inferential_unit,
                "sample_metadata_columns": self._sample_metadata.columns.tolist(),
                "run_manifest": {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "model_names": list(self.config.models),
                    "cv_strategy": self.config.cv.strategy,
                    "metrics": self.config.get_all_evaluation_metrics(),
                },
                "hardware_provenance": {"n_jobs": self.config.n_jobs},
                "capabilities": self._capability_payload(),
            }
        )
        if t_axis is not None:
            meta["time_axis"] = t_axis.tolist()
        return meta

    def _capability_payload(self) -> Dict[str, Any]:
        sels = {}
        if self.config.feature_selection.enabled:
            sels[self.config.feature_selection.method] = get_selector_capabilities(
                self.config.feature_selection.method
            ).to_dict()
        return {
            "models": {n: c.to_dict() for n, c in self._model_capabilities.items()},
            "estimator_specs": {n: s.to_dict() for n, s in self._model_specs.items()},
            "feature_selectors": sels,
            "metrics": {
                m: {
                    "task": get_metric_spec(m).task,
                    "response_method": get_metric_spec(m).response_method,
                    "family": get_metric_spec(m).family,
                }
                for m in self.config.get_all_evaluation_metrics()
            },
        }

    def _resolve_feature_names(
        self, X: np.ndarray, names: Optional[Sequence[str]]
    ) -> list[str]:
        exp = 1 if X.ndim < 2 else X.shape[1]
        if names is not None:
            if len(names) != exp:
                raise ValueError(
                    f"feature_names length mismatch: {len(names)} vs {exp}"
                )
            return [str(n) for n in names]
        return (
            ["feature_0"]
            if X.ndim < 2
            else [f"feature_{idx}" for idx in range(X.shape[1])]
        )


def _validate_fold_integrity(
    y_train: np.ndarray, y_test: np.ndarray, tasks: tuple
) -> None:
    """Check if CV folds are degenerate before fitting."""
    if y_train.size == 0 or y_test.size == 0:
        raise ValueError("Empty fold detected.")

    if np.min(y_train) == np.max(y_train):
        raise ValueError(
            f"Degenerate Train Fold: Only one value found ({y_train[0]}). "
            "Scoring metrics are undefined for constant targets."
        )

    if "classification" in tasks and np.min(y_test) == np.max(y_test):
        raise ValueError(
            f"Degenerate Test Fold: Only one class found ({y_test[0]}). "
            "Metrics like ROC-AUC are undefined for single-class test sets."
        )
