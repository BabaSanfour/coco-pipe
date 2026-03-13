"""
Dimensionality Reduction Core
=============================

Execution manager for one dimensionality reduction method.

`DimReduction` is intentionally narrow. It owns reducer instantiation,
input-shape validation for execution, fit/transform operations, and cached
evaluation/interpretation state for one reducer instance. Plotting, trajectory
reshaping, reporting, and multi-method comparison live in dedicated modules.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .analysis import interpret_features
from .config import BaseReducerConfig, get_reducer_class
from .evaluation.core import evaluate_embedding
from .reducers.base import BaseReducer

__all__ = ["DimReduction"]


class DimReduction:
    """
    Manage one dimensionality reduction workflow.

    Parameters
    ----------
    method : str or BaseReducerConfig
        Canonical public reducer name or a typed configuration object.
        Method names are exact and must match the registry, for example
        ``"PCA"``, ``"Isomap"``, ``"Pacmap"``, or ``"TopologicalAE"``.
    n_components : int, default=2
        Target dimensionality when ``method`` is a string.
    params : dict, optional
        Additional reducer keyword arguments merged into the constructor
        arguments when ``method`` is a string.
    **kwargs : dict
        Runtime reducer keyword overrides. These are merged after ``params``.

    Attributes
    ----------
    method : str
        Canonical reducer name.
    n_components : int
        Target dimensionality used for the reducer instance.
    reducer : BaseReducer
        Instantiated reducer backend.
    metrics_ : dict
        Cached scalar evaluation summaries from the latest ``score()`` call.
    quality_metadata_ : dict
        Cached scalar reducer metadata exposed through the reducer contract.
    diagnostics_ : dict
        Cached non-scalar diagnostic artifacts exposed through the reducer
        contract or the evaluation layer.
    metric_records_ : list of dict
        Cached tidy metric observations produced by the evaluator.
    interpretation_ : dict
        Cached feature interpretation payloads from the latest
        ``interpret()`` call.
    interpretation_records_ : list of dict
        Cached tidy feature-interpretation observations.

    See Also
    --------
    coco_pipe.dim_reduction.analysis.interpret_features
        Pure interpretation backend used by ``interpret()``.
    coco_pipe.dim_reduction.evaluation.core.evaluate_embedding
        Pure evaluator used by ``score()``.
    coco_pipe.dim_reduction.evaluation.core.MethodSelector
        Post-hoc comparison and ranking over already-scored reducers.
    coco_pipe.viz.dim_reduction
        Plotting utilities for embeddings, metrics, and diagnostics.

    Examples
    --------
    >>> reducer = DimReduction("UMAP", n_components=2, n_neighbors=15)
    >>> embedding = reducer.fit_transform(X)
    >>> scores = reducer.score(embedding, X=X)
    >>> "trustworthiness" in scores["metrics"]
    True
    >>> interpretation = reducer.interpret(
    ...     X,
    ...     X_emb=embedding,
    ...     analyses=["correlation"],
    ...     feature_names=feature_names,
    ... )
    >>> "correlation" in interpretation["analysis"]
    True
    """

    def __init__(
        self,
        method: Union[str, "BaseReducerConfig"],
        n_components: int = 2,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a reduction manager for one canonical reducer.

        Parameters
        ----------
        method : str or BaseReducerConfig
            Exact canonical reducer name or a typed dim-reduction
            configuration.
        n_components : int, default=2
            Target dimensionality when ``method`` is a string.
        params : dict, optional
            Reducer keyword arguments merged before ``**kwargs`` when
            ``method`` is a string.
        **kwargs : dict
            Runtime reducer keyword overrides.
        """
        self.reducer_kwargs = params.copy() if params else {}

        if isinstance(method, BaseReducerConfig):
            self.method = method.method
            self.n_components = method.n_components
            self.reducer_kwargs.update(method.to_reducer_kwargs())
        else:
            self.method = method
            self.n_components = n_components

        self.reducer_kwargs.update(kwargs)

        ReducerCls = get_reducer_class(self.method)
        self.reducer: BaseReducer = ReducerCls(
            n_components=self.n_components, **self.reducer_kwargs
        )

        self.metrics_: Dict[str, Any] = {}
        self.quality_metadata_: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.metric_records_: List[Dict[str, Any]] = []
        self.interpretation_: Dict[str, Any] = {}
        self.interpretation_records_: List[Dict[str, Any]] = []

    @property
    def random_state(self) -> Optional[int]:
        """Return the random seed from parameters if any."""
        return self.reducer_kwargs.get("random_state")

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Return reducer capability metadata through the manager interface."""
        return self.reducer.capabilities

    def _reset_cached_outputs(self) -> None:
        """Clear cached evaluation outputs."""
        self.metrics_ = {}
        self.quality_metadata_ = {}
        self.diagnostics_ = {}
        self.metric_records_ = []
        self.interpretation_ = {}
        self.interpretation_records_ = []

    def _validate_input(self, X: Any) -> np.ndarray:
        """
        Validate reducer input shape and coerce to a NumPy array.

        Parameters
        ----------
        X : array-like or MNE object
            Input data accepted by the reducer. Objects exposing ``get_data()``
            are unwrapped before validation.

        Returns
        -------
        X : np.ndarray
            Validated reducer input.

        Raises
        ------
        ValueError
            If the input dimensionality does not match the reducer contract.
        """
        if hasattr(X, "get_data"):  # Handle MNE objects
            X = X.get_data()

        X = np.asarray(X)

        caps = self.reducer.capabilities
        expected_ndim = caps.get("input_ndim", 2)

        if X.ndim != expected_ndim:
            raise ValueError(
                f"Method '{self.method}' requires {expected_ndim}D input; "
                f"got shape {X.shape}."
            )

        return X

    def fit(self, X: Any, y: Optional[Any] = None) -> "DimReduction":
        """
        Fit the reducer on the provided data.

        Parameters
        ----------
        X : array-like or MNE object
            Input data in the reducer's native layout.
        y : array-like, optional
            Optional supervision forwarded to the reducer.

        Returns
        -------
        self : DimReduction
            The fitted reducer.
        """
        X_arr = self._validate_input(X)
        self._reset_cached_outputs()
        self.reducer.fit(X_arr, y=y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Transform new data with a fitted reducer.

        Parameters
        ----------
        X : array-like or MNE object
            Input data in the reducer's native layout.

        Returns
        -------
        X_emb : np.ndarray
            Reduced representation returned by the reducer.
        """
        X = self._validate_input(X)
        return self.reducer.transform(X)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """
        Fit the reducer and return the reduced representation.

        Parameters
        ----------
        X : array-like or MNE object
            Input data in the reducer's native layout.
        y : array-like, optional
            Optional supervision forwarded to the reducer.

        Returns
        -------
        X_emb : np.ndarray
            Reduced representation returned by the reducer.
        """
        X = self._validate_input(X)
        self._reset_cached_outputs()
        return self.reducer.fit_transform(X, y=y)

    def get_components(self) -> np.ndarray:
        """
        Return reducer-defined component-like outputs.

        Returns
        -------
        components : np.ndarray
            Component-like array exposed by the reducer.

        Raises
        ------
        ValueError
            If the reducer does not expose public components.
        """
        return self.reducer.get_components()

    def score(
        self,
        X_emb: np.ndarray,
        X: Any = None,
        n_neighbors: int = 5,
        metrics: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        labels: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        separation_method: str = "centroid",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate an explicit embedding against the original data.

        Parameters
        ----------
        X_emb : array-like
            Embedded data to evaluate.
        X : array-like, optional
            Original high-dimensional data in evaluation-ready layout. This is
            required for standard 2D metrics and optional for native 3D
            trajectory metrics.
        n_neighbors : int, default=5
            K-nearest neighbors size for metric computation.
        metrics : list of str, optional
            Metric selectors to compute. ``None`` evaluates all metric families
            available for the embedding shape.
        k_values : list of int, optional
            Neighborhood sizes used for multi-scale standard metric evaluation.
        labels : np.ndarray, optional
            Optional labels aligned with the embedding. Used for trajectory
            separation when ``X_emb`` is 3D.
        times : np.ndarray, optional
            Optional trajectory time coordinates aligned with the trajectory
            length axis.
        separation_method : str, default="centroid"
            Separation definition passed to trajectory evaluation when labels
            are available for native 3D trajectory embeddings.

        Returns
        -------
        scores : dict
            Dictionary with keys ``"metrics"``, ``"metadata"``, and
            ``"diagnostics"``.

        Notes
        -----
        ``score()`` does not infer or cache embeddings. Callers must pass
        ``X_emb`` explicitly. ``X`` is only required when the requested
        evaluation path needs the original high-dimensional samples.
        """
        payload = evaluate_embedding(
            X_emb=X_emb,
            X=X,
            method_name=self.method,
            metrics=metrics,
            labels=labels,
            times=times,
            quality_metadata=self.get_quality_metadata(),
            diagnostics=self.get_diagnostics(),
            random_state=getattr(self, "random_state", None),
            n_neighbors=n_neighbors,
            k_values=k_values,
            separation_method=separation_method,
        )

        metrics_payload = payload["metrics"]
        metadata_payload = payload["metadata"]
        diagnostics_payload = payload["diagnostics"]

        if not metrics_payload:
            metrics_payload["note"] = (
                "Metrics unavailable for the current embedding layout."
            )

        self.metrics_ = metrics_payload
        self.quality_metadata_ = dict(metadata_payload)
        self.diagnostics_ = dict(diagnostics_payload)
        self.metric_records_ = list(payload["records"])

        return {
            "metrics": self.metrics_,
            "metadata": self.quality_metadata_,
            "diagnostics": self.diagnostics_,
        }

    def interpret(
        self,
        X: np.ndarray,
        *,
        X_emb: np.ndarray,
        analyses: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run feature interpretation analyses for an explicit embedding.

        Parameters
        ----------
        X : np.ndarray
            Original input data.
        X_emb : np.ndarray
            Explicit embedding aligned with ``X``.
        analyses : list of {"correlation", "perturbation", "gradient"}, optional
            Interpretation analyses to compute. ``None`` defaults to
            ``["correlation"]``.
        feature_names : list of str, optional
            Feature names aligned with the columns of ``X`` when the requested
            interpretation returns feature-keyed outputs.
        n_repeats : int, default=5
            Number of shuffles per feature for perturbation importance.
        random_state : int, optional
            Random seed for perturbation importance.

        Returns
        -------
        dict
            Dictionary with keys ``"analysis"`` and ``"records"``.

        Notes
        -----
        ``interpret()`` does not fit the reducer or compute embeddings.
        Callers must pass both ``X`` and ``X_emb`` explicitly.

        See Also
        --------
        coco_pipe.dim_reduction.analysis.interpret_features
            Pure interpretation backend used by this manager method.
        score
            Evaluate structure-preservation metrics for an explicit embedding.

        Examples
        --------
        >>> reducer = DimReduction("PCA", n_components=2)
        >>> embedding = reducer.fit_transform(X)
        >>> result = reducer.interpret(
        ...     X,
        ...     X_emb=embedding,
        ...     analyses=["correlation"],
        ...     feature_names=feature_names,
        ... )
        >>> sorted(result)
        ['analysis', 'records']
        """
        payload = interpret_features(
            X,
            X_emb=X_emb,
            model=self.reducer,
            analyses=analyses,
            feature_names=feature_names,
            method_name=self.method,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        self.interpretation_ = dict(payload["analysis"])
        self.interpretation_records_ = list(payload["records"])
        return {
            "analysis": self.interpretation_,
            "records": list(self.interpretation_records_),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return cached diagnostics merged with reducer diagnostics.

        Returns
        -------
        diagnostics : dict
            Diagnostic artifacts declared by the reducer contract and the
            evaluation layer.
        """
        self.diagnostics_.update(self.reducer.get_diagnostics())
        return self.diagnostics_.copy()

    def get_quality_metadata(self) -> Dict[str, Any]:
        """
        Return cached scalar metadata merged with reducer metadata.

        Returns
        -------
        metadata : dict
            Scalar metadata declared by the reducer contract and the evaluation
            layer.
        """
        self.quality_metadata_.update(self.reducer.get_quality_metadata())
        return self.quality_metadata_.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Return cached scalar metrics from the latest ``score()`` call."""
        return self.metrics_.copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a normalized summary payload for report and export paths.

        Returns
        -------
        dict
            Plain dictionary containing method identity, cached scalar
            summaries, reducer metadata, diagnostics, tidy metric records, and
            capability flags, plus cached feature interpretation payloads.

        Notes
        -----
        The summary does not include an embedding payload. Embeddings are
        handled explicitly outside the manager and must be passed directly to
        plotting or reporting utilities that need them.
        """
        return {
            "method": self.method,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "metrics": self.get_metrics(),
            "metric_records": list(self.metric_records_),
            "quality_metadata": self.get_quality_metadata(),
            "diagnostics": self.get_diagnostics(),
            "interpretation": dict(self.interpretation_),
            "interpretation_records": list(self.interpretation_records_),
            "capabilities": self.capabilities,
        }

    def save(self, path: Union[str, Path]):
        """
        Save the underlying reducer to disk.

        Parameters
        ----------
        path : str or Path
            Output path for reducer persistence.

        Notes
        -----
        Only the reducer model is persisted. Cached manager state such as
        metrics and diagnostics is not included.
        """
        self.reducer.save(path)

    @classmethod
    def load(cls, path: Union[str, Path], method: str) -> "DimReduction":
        """
        Load a persisted reducer and wrap it in a fresh manager.

        Parameters
        ----------
        path : str or Path
            Path to a serialized reducer saved with ``save()``.
        method : str
            Canonical public reducer name used to reconstruct the manager.

        Returns
        -------
        DimReduction
            Fresh manager wrapping the loaded reducer model.

        Notes
        -----
        This restores the reducer model only. Cached manager state such as
        scores, diagnostics, and metric records is not persisted.
        """
        reducer_instance = BaseReducer.load(path)
        manager = cls(method=method, n_components=reducer_instance.n_components)
        manager.reducer = reducer_instance
        return manager
