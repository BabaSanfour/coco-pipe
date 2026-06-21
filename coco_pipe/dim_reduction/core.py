"""
Dimensionality Reduction Core
=============================

Execution manager for one dimensionality reduction method.

`~coco_pipe.dim_reduction.DimReduction` is intentionally narrow. It owns reducer instantiation,
input-shape validation for execution, fit/transform operations, and cached
evaluation/interpretation state for one reducer instance. Plotting, trajectory
reshaping, reporting, and multi-method comparison live in dedicated modules.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from pathlib import Path
from typing import Any, Union

import numpy as np

from coco_pipe.io.structures import DataContainer

from .analysis import interpret_features
from .config import BaseReducerConfig, EvaluationConfig, get_reducer_class
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
        params: dict[str, Any] | None = None,
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

        self.metrics_: dict[str, Any] = {}
        self.quality_metadata_: dict[str, Any] = {}
        self.diagnostics_: dict[str, Any] = {}
        self.metric_records_: list[dict[str, Any]] = []
        self.interpretation_: dict[str, Any] = {}
        self.interpretation_records_: list[dict[str, Any]] = []

    @property
    def random_state(self) -> int | None:
        """Return the random seed from parameters if any."""
        return self.reducer_kwargs.get("random_state")

    @property
    def capabilities(self) -> dict[str, Any]:
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

    def _prepare_input(self, X: Any) -> tuple[np.ndarray, DataContainer | None]:
        """
        Validate reducer input shape and unwrap it to a NumPy array.

        Parameters
        ----------
        X : DataContainer, array-like, or MNE object
            Input data accepted by the reducer. A ``~coco_pipe.io.DataContainer`` is unwrapped
            to its ``X`` array and remembered so the embedding can be re-wrapped
            as a container; objects exposing ``get_data()`` (e.g. MNE) are
            unwrapped to arrays.

        Returns
        -------
        data : np.ndarray
            Validated reducer input.
        source : DataContainer or None
            The source container when ``X`` was a ``~coco_pipe.io.DataContainer``, else
            ``None``.

        Raises
        ------
        ValueError
            If the input dimensionality does not match the reducer contract.
        """
        source = X if isinstance(X, DataContainer) else None
        if source is not None:
            data = source.X
        elif hasattr(X, "get_data"):  # Handle MNE objects
            data = X.get_data()
        else:
            data = X

        data = np.asarray(data)

        caps = self.reducer.capabilities
        expected_ndim = caps.get("input_ndim", 2)

        if data.ndim != expected_ndim:
            raise ValueError(
                f"Method '{self.method}' requires {expected_ndim}D input; "
                f"got shape {data.shape}."
            )

        return data, source

    def _wrap_embedding(
        self, embedding: np.ndarray, source: DataContainer | None
    ) -> Any:
        """
        Re-attach an embedding to its source container when possible.

        Parameters
        ----------
        embedding : np.ndarray
            Reduced representation returned by the reducer.
        source : DataContainer or None
            Source container from :meth:`_prepare_input`, or ``None`` when the
            call was made with a raw array.

        Returns
        -------
        DataContainer or np.ndarray
            An embedding ``~coco_pipe.io.DataContainer`` with a ``component`` axis when
            ``source`` is a container and the embedding shares its observation
            layout (the standard 2-D case); otherwise the raw embedding array.
            Native 3-D trajectory embeddings stay as arrays.
        """
        embedding = np.asarray(embedding)
        if (
            source is None
            or embedding.ndim != source.X.ndim
            or embedding.shape[:-1] != source.X.shape[:-1]
        ):
            return embedding
        names = [f"component_{i + 1}" for i in range(embedding.shape[-1])]
        return source.with_features(embedding, names=names, new_dim_name="component")

    def fit(self, X: Any, y: Any | None = None) -> "DimReduction":
        """
        Fit the reducer on the provided data.

        Parameters
        ----------
        X : DataContainer, array-like, or MNE object
            Input data in the reducer's native layout.
        y : array-like, optional
            Optional supervision forwarded to the reducer.

        Returns
        -------
        self : DimReduction
            The fitted reducer.
        """
        X_arr, _ = self._prepare_input(X)
        self._reset_cached_outputs()
        self.reducer.fit(X_arr, y=y)
        return self

    def transform(self, X: Any) -> Any:
        """
        Transform new data with a fitted reducer.

        Parameters
        ----------
        X : DataContainer, array-like, or MNE object
            Input data in the reducer's native layout.

        Returns
        -------
        X_emb : DataContainer or np.ndarray
            Reduced representation. A ``~coco_pipe.io.DataContainer`` input yields an embedding
            ``~coco_pipe.io.DataContainer`` (``component`` axis, ids/coords/meta preserved) for
            standard 2-D embeddings; array input yields an array.
        """
        X_arr, source = self._prepare_input(X)
        embedding = self.reducer.transform(X_arr)
        return self._wrap_embedding(embedding, source)

    def fit_transform(self, X: Any, y: Any | None = None) -> Any:
        """
        Fit the reducer and return the reduced representation.

        Parameters
        ----------
        X : DataContainer, array-like, or MNE object
            Input data in the reducer's native layout.
        y : array-like, optional
            Optional supervision forwarded to the reducer.

        Returns
        -------
        X_emb : DataContainer or np.ndarray
            Reduced representation. A ``~coco_pipe.io.DataContainer`` input yields an embedding
            ``~coco_pipe.io.DataContainer`` (``component`` axis, ids/coords/meta preserved) for
            standard 2-D embeddings; array input yields an array.
        """
        X_arr, source = self._prepare_input(X)
        self._reset_cached_outputs()
        embedding = self.reducer.fit_transform(X_arr, y=y)
        return self._wrap_embedding(embedding, source)

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
        X_emb: Any,
        X: Any = None,
        n_neighbors: int = 5,
        metrics: list[str] | None = None,
        k_values: list[int] | None = None,
        labels: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        times: np.ndarray | None = None,
        separation_method: str | None = None,
        config: EvaluationConfig | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate an explicit embedding against the original data.

        Parameters
        ----------
        X_emb : DataContainer or array-like
            Embedded data to evaluate. A ``~coco_pipe.io.DataContainer`` is unwrapped to its
            ``X`` array.
        X : DataContainer or array-like, optional
            Original high-dimensional data in evaluation-ready layout. This is
            required for standard 2D metrics and optional for native 3D
            trajectory metrics. A ``~coco_pipe.io.DataContainer`` is unwrapped to its ``X``
            array.
        n_neighbors : int, default=5
            K-nearest neighbors size for metric computation.
        metrics : list of str, optional
            Metric selectors to compute. ``None`` evaluates all metric families
            available for the embedding shape. Explicit values take precedence
            over ``config``.
        k_values : list of int, optional
            Neighborhood sizes used for multi-scale standard metric evaluation.
            Explicit values take precedence over ``config``.
        labels : np.ndarray, optional
            Optional labels aligned with the embedding. Used for trajectory
            separation when ``X_emb`` is 3D and for explicit supervised 2D
            metrics when requested.
        groups : np.ndarray, optional
            Optional grouping variable aligned with the embedding. Required by
            grouped supervised evaluation metrics such as
            ``separation_logreg_balanced_accuracy``.
        times : np.ndarray, optional
            Optional trajectory time coordinates aligned with the trajectory
            length axis.
        separation_method : str, optional
            Separation definition passed to trajectory evaluation when labels
            are available for native 3D trajectory embeddings. ``None`` defers to
            ``config`` and otherwise falls back to ``"centroid"``.
        config : EvaluationConfig, optional
            Typed evaluation configuration. Supplies ``metrics``, ``k_values``
            (from ``config.k_range``), and ``separation_method`` when those are
            not passed explicitly. Mirrors how ``DimReduction.__init__`` accepts
            a ``BaseReducerConfig``.

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
        if isinstance(X_emb, DataContainer):
            X_emb = X_emb.X
        if isinstance(X, DataContainer):
            X = X.X

        from .evaluation.core import evaluate_embedding

        payload = evaluate_embedding(
            X_emb=X_emb,
            X=X,
            method_name=self.method,
            metrics=metrics,
            labels=labels,
            groups=groups,
            times=times,
            quality_metadata=self.get_quality_metadata(),
            diagnostics=self.get_diagnostics(),
            random_state=getattr(self, "random_state", None),
            n_neighbors=n_neighbors,
            k_values=k_values,
            separation_method=separation_method,
            config=config,
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
        analyses: list[str] | None = None,
        feature_names: list[str] | None = None,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> dict[str, Any]:
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

    def get_diagnostics(self) -> dict[str, Any]:
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

    def get_quality_metadata(self) -> dict[str, Any]:
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

    def get_metrics(self) -> dict[str, Any]:
        """Return cached scalar metrics from the latest ``score()`` call."""
        return self.metrics_.copy()

    def get_summary(self) -> dict[str, Any]:
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

    def save(self, path: str | Path):
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
    def load(cls, path: str | Path, method: str) -> "DimReduction":
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
