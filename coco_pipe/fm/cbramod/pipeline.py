"""
Pipeline wrapper for running regression with the CBRAMod foundation model.

This mirrors the public surface of the classical regression pipeline but swaps
the estimator for a foundation-model-backed regressor. The embedder is expected
to turn raw inputs into embeddings; a light downstream regressor is trained on
top of those embeddings.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from coco_pipe.ml.base import BasePipeline
from coco_pipe.ml.config import DEFAULT_CV, REGRESSION_METRICS


class FoundationRegressor(BaseEstimator, RegressorMixin):
    """
    Thin sklearn-compatible wrapper around a foundation model embedder.

    Parameters
    ----------
    embed_fn : callable
        Callable that maps ``X`` to a 2D numpy array of embeddings. Either a
        ``__call__`` or ``transform`` method will be used.
    base_regressor : BaseEstimator, optional
        Downstream regressor trained on the embeddings. Defaults to ``Ridge``.
    multioutput : bool, optional
        If True, wraps the base regressor in ``MultiOutputRegressor`` for
        multi-target regression.
    """

    def __init__(
        self,
        embed_fn: Callable[[Any], np.ndarray],
        base_regressor: Optional[BaseEstimator] = None,
        multioutput: bool = False,
    ):
        self.embed_fn = embed_fn
        self.base_regressor = base_regressor or Ridge(random_state=42)
        self.multioutput = multioutput

        if self.multioutput:
            self.base_regressor = MultiOutputRegressor(self.base_regressor)

    def _embed(self, X: Any) -> np.ndarray:
        if hasattr(self.embed_fn, "transform"):
            emb = self.embed_fn.transform(X)
        else:
            emb = self.embed_fn(X)
        emb = np.asarray(emb)
        if emb.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got shape {emb.shape} from embed_fn"
            )
        return emb

    def fit(self, X: Any, y: Any):
        X_emb = self._embed(X)
        self.base_regressor.fit(X_emb, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_emb = self._embed(X)
        return self.base_regressor.predict(X_emb)


class CBRAModRegressionPipeline(BasePipeline):
    """
    Regression pipeline that routes features through the CBRAMod foundation
    model before fitting a lightweight regressor.

    The interface mirrors ``RegressionPipeline`` for analysis_type dispatch but
    exposes a required ``embed_fn`` to obtain embeddings from the foundation
    model.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        embed_fn: Callable[[Any], np.ndarray],
        metrics: Union[str, Sequence[str], None] = None,
        base_regressor: Optional[BaseEstimator] = None,
        hp_search_params: Optional[Dict[str, Sequence[Any]]] = None,
        use_scaler: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = False,
    ):
        self.embed_fn = embed_fn
        self.verbose = verbose

        # Determine if we need multi-output support for the downstream regressor
        is_multi = hasattr(y, "ndim") and getattr(y, "ndim", 1) == 2

        metric_funcs = REGRESSION_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["r2"])

        foundation_estimator = FoundationRegressor(
            embed_fn=embed_fn,
            base_regressor=base_regressor,
            multioutput=is_multi,
        )

        model_configs = {
            "CBRAMod": {
                "estimator": foundation_estimator,
                "default_params": {},
                "hp_search_params": hp_search_params
                or ({} if is_multi else {"base_regressor__alpha": [0.1, 1.0, 10.0]}),
            }
        }

        cv = dict(DEFAULT_CV)
        # Regression should not stratify continuous targets; default to kfold
        cv["cv_strategy"] = "kfold"
        if cv_kwargs:
            cv.update(cv_kwargs)

        super().__init__(
            X=X,
            y=y,
            metric_funcs=metric_funcs,
            model_configs=model_configs,
            use_scaler=use_scaler,
            default_metrics=default_metrics,
            cv_kwargs=cv,
            groups=groups,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.model_name = "CBRAMod"

    def run(
        self,
        analysis_type: str = "baseline",
        n_features: Optional[int] = None,
        direction: str = "forward",
        search_type: str = "grid",
        n_iter: int = 50,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        analysis_type = analysis_type.lower()
        if analysis_type not in {"baseline", "feature_selection", "hp_search", "hp_search_fs"}:
            raise ValueError(f"Invalid analysis type: {analysis_type}")

        if analysis_type == "baseline":
            return self.baseline_evaluation(self.model_name)

        if analysis_type == "feature_selection":
            return self.feature_selection(
                self.model_name,
                n_features=n_features,
                direction=direction,
                scoring=scoring,
            )

        if analysis_type == "hp_search":
            return self.hp_search(
                self.model_name,
                param_grid=None,
                search_type=search_type,
                n_iter=n_iter,
                scoring=scoring,
            )

        return self.hp_search_fs(
            self.model_name,
            param_grid=None,
            search_type=search_type,
            n_features=n_features,
            direction=direction,
            n_iter=n_iter,
            scoring=scoring,
        )
