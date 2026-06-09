"""
Evaluation Core
===============

Pure evaluation orchestration for dimensionality-reduction workflows.

This module contains the two public evaluation interfaces used by the
dim-reduction stack:

- ``evaluate_embedding(...)`` evaluates an explicit embedding and returns
  scalar metrics, scalar metadata, diagnostics, and tidy metric records.
- ``MethodSelector`` compares and ranks multiple already-scored
  ``DimReduction`` objects without refitting or recomputing embeddings.

The module is intentionally evaluation-only. It does not fit reducers,
transform data, reconstruct 3D trajectory tensors from flat embeddings, or
provide plotting methods. Reduction execution belongs to
``coco_pipe.dim_reduction.core.DimReduction`` and plotting belongs to
``coco_pipe.viz.dim_reduction``.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from ..core import DimReduction

from ._supervised import _cross_validate_score
from .geometry import (
    trajectory_acceleration,
    trajectory_curvature,
    trajectory_dispersion,
    trajectory_displacement,
    trajectory_path_length,
    trajectory_separation,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
)
from .metrics import (
    compute_coranking_matrix,
    compute_mrre,
    continuity,
    lcmc,
    shepard_diagram_data,
    trustworthiness,
)

__all__ = ["evaluate_embedding", "MethodSelector"]

METRIC_COLUMNS = ("method", "metric", "value", "scope", "scope_value")
SEPARATION_LOGREG_BALANCED_ACCURACY = "separation_logreg_balanced_accuracy"
SWEEP_METRICS = (
    "trustworthiness",
    "continuity",
    "lcmc",
    "mrre_intrusion",
    "mrre_extrusion",
    "mrre_total",
)

DEFAULT_SCORE_METRICS = (
    *SWEEP_METRICS,
    "shepard_correlation",
    "trajectory_speed",
    "trajectory_acceleration",
    "trajectory_curvature",
    "trajectory_turning_angle",
    "trajectory_dispersion",
    "trajectory_path_length",
    "trajectory_displacement",
    "trajectory_tortuosity",
    "trajectory_separation",
)
RANKING_DIRECTIONS = {
    "trustworthiness": "desc",
    "continuity": "desc",
    "lcmc": "desc",
    "shepard_correlation": "desc",
    SEPARATION_LOGREG_BALANCED_ACCURACY: "desc",
    "mrre_intrusion": "asc",
    "mrre_extrusion": "asc",
    "mrre_total": "asc",
}


def _summarize_trajectory_metric(
    prefix: str,
    values: np.ndarray,
    *,
    summary_type: str,
    use_last_axis: bool = False,
) -> Dict[str, float]:
    """Return scalar summaries for one trajectory metric payload."""
    arr = np.asarray(values, dtype=float)
    summary_: Dict[str, float] = {}
    if summary_type == "peak":
        summary_[f"{prefix}_mean"] = float(np.nanmean(arr))
        summary_[f"{prefix}_peak"] = float(np.nanmax(arr))
    elif summary_type == "final":
        final_values = arr[..., -1] if use_last_axis else arr
        summary_[f"{prefix}_final"] = float(np.nanmean(final_values))
    else:
        raise ValueError(f"Unsupported trajectory summary type '{summary_type}'.")
    return summary_


def _evaluate_trajectory_metrics(
    method_name: str,
    X_emb: np.ndarray,
    metric_selection: Optional[set],
    labels: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    separation_method: str = "centroid",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute trajectory summaries and diagnostics for native 3D embeddings.

    Parameters
    ----------
    method_name : str
        Display name attached to the tidy metric records.
    X_emb : np.ndarray
        Embedded trajectories with shape ``(n_trajectories, n_times, n_dims)``.
    metric_selection : set of str or None
        Requested trajectory metric families. ``None`` computes all supported
        trajectory metrics.
    labels : np.ndarray, optional
        One label per trajectory. Labels are currently only used by
        ``trajectory_separation``.
    times : np.ndarray, optional
        One time value per trajectory step. When provided and aligned, it is
        used for separation AUC integration and stored as a diagnostic.
    separation_method : str, default="centroid"
        Separation definition passed to ``trajectory_separation``.

    Returns
    -------
    metrics : dict
        Scalar summary metrics for the requested trajectory families.
    metadata : dict
        Scalar metadata describing the trajectory tensor.
    diagnostics : dict
        Array-like or structured trajectory diagnostics.
    records : list of dict
        Tidy long-form metric records.

    Notes
    -----
    This evaluator only operates on embeddings that are already shaped as
    ``(n_trajectories, n_times, n_dims)``. Trajectory reconstruction from flat
    2D embeddings is intentionally out of scope and must happen upstream.

    The evaluator-level ``trajectory_dispersion`` metric always uses the global,
    unlabeled dispersion definition by calling
    ``trajectory_dispersion(traj, labels=None)``. Even when trajectory labels are
    available, those labels are currently only used by
    ``trajectory_separation``, using the caller-provided ``separation_method``.
    This keeps the non-separation trajectory metric loop uniform, but it means
    evaluator-level dispersion summarizes overall spread rather than per-label
    spread.
    """
    traj = np.asarray(X_emb)
    if traj.ndim != 3:
        return {}, {}, {}, []

    if times is not None:
        candidate = np.asarray(times).reshape(-1)
        if len(candidate) == traj.shape[1]:
            times = candidate

    if labels is not None:
        candidate = np.asarray(labels).reshape(-1)
        if len(candidate) == traj.shape[0]:
            labels = candidate

    metrics_payload: Dict[str, Any] = {}
    metadata_payload: Dict[str, Any] = {
        "trajectory_count": int(traj.shape[0]),
        "trajectory_length": int(traj.shape[1]),
    }
    diagnostics_payload: Dict[str, Any] = {
        "trajectory_times_": times,
    }
    records: List[Dict[str, Any]] = []

    metrics = (
        ("trajectory_speed", trajectory_speed, "peak", False, 2),
        ("trajectory_acceleration", trajectory_acceleration, "peak", False, 3),
        ("trajectory_curvature", trajectory_curvature, "peak", False, 2),
        ("trajectory_turning_angle", trajectory_turning_angle, "peak", False, 3),
        ("trajectory_dispersion", trajectory_dispersion, "peak", False, 1),
        (
            "trajectory_path_length",
            lambda values: trajectory_path_length(values, cumulative=True),
            "final",
            True,
            2,
        ),
        ("trajectory_displacement", trajectory_displacement, "final", True, 1),
        ("trajectory_tortuosity", trajectory_tortuosity, "final", False, 2),
    )
    for (
        metric_prefix,
        metric_func,
        summary_type,
        use_last_axis,
        min_timepoints,
    ) in metrics:
        if metric_selection is not None and metric_prefix not in metric_selection:
            continue
        if traj.shape[1] < min_timepoints:
            continue

        values = metric_func(traj)
        diagnostics_payload[f"{metric_prefix}_"] = values
        summary = _summarize_trajectory_metric(
            metric_prefix,
            values,
            summary_type=summary_type,
            use_last_axis=use_last_axis,
        )
        metrics_payload.update(summary)
        for metric_name, value in summary.items():
            is_num = isinstance(value, (int, float, np.number))
            if is_num and not isinstance(value, bool):
                records.append(
                    {
                        "method": method_name,
                        "metric": metric_name,
                        "value": float(value),
                        "scope": "global",
                        "scope_value": "global",
                    }
                )
    if (
        (metric_selection is None or "trajectory_separation" in metric_selection)
        and labels is not None
        and len(np.unique(labels)) > 1
    ):
        separation = trajectory_separation(
            traj,
            labels,
            method=separation_method,
        )
        diagnostics_payload["trajectory_separation_"] = separation
        for pair, values in separation.items():
            pair_suffix = f"{pair[0]}::{pair[1]}"
            values_arr = np.asarray(values)
            integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if values_arr.size == 0:
                auc_value = float("nan")
                peak_value = float("nan")
            elif times is None:
                auc_value = float(integrate(values_arr))
                peak_value = float(np.nanmax(values_arr))
            else:
                time_arr = np.asarray(times)
                auc_value = (
                    float(integrate(values_arr, x=time_arr))
                    if len(time_arr) == len(values_arr)
                    else float(integrate(values_arr))
                )
                peak_value = float(np.nanmax(values_arr))
            pair_metrics = {
                f"trajectory_separation_auc::{pair_suffix}": auc_value,
                f"trajectory_separation_peak::{pair_suffix}": peak_value,
            }
            metrics_payload.update(pair_metrics)
            for metric_name, value in pair_metrics.items():
                is_num = isinstance(value, (int, float, np.number))
                if is_num and not isinstance(value, bool):
                    records.append(
                        {
                            "method": method_name,
                            "metric": metric_name,
                            "value": float(value),
                            "scope": "global",
                            "scope_value": "global",
                            "pair": f"{pair[0]} vs {pair[1]}",
                        }
                    )

    return metrics_payload, metadata_payload, diagnostics_payload, records


def _evaluate_standard_metrics(
    method_name: str,
    X_eval: np.ndarray,
    X_emb_eval: np.ndarray,
    metric_selection: Optional[set],
    n_neighbors: int,
    k_values: Optional[Sequence[int]],
    random_state: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute standard co-ranking and Shepard-based metrics for a 2D embedding.

    Parameters
    ----------
    method_name : str
        Display name attached to the tidy metric records.
    X_eval : np.ndarray
        Original data with shape ``(n_samples, n_features)``.
    X_emb_eval : np.ndarray
        Embedded data with shape ``(n_samples, n_components)``.
    metric_selection : set of str or None
        Requested standard metrics. ``None`` computes all standard metrics
        supported by this evaluator.
    n_neighbors : int
        Neighborhood size used when no explicit ``k_values`` sweep is
        requested.
    k_values : sequence of int, optional
        Explicit neighborhood sizes for sweep-style evaluation.
    random_state : int, optional
        Random state used for sampled Shepard distances.

    Returns
    -------
    metrics : dict
        Scalar standard metrics.
    diagnostics : dict
        Standard evaluation diagnostics such as the co-ranking matrix or
        Shepard sampled distances.
    records : list of dict
        Tidy long-form metric records.
    """
    metrics_payload: Dict[str, Any] = {}
    diagnostics_payload: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []

    requested_k_metrics = (
        set(SWEEP_METRICS)
        if metric_selection is None
        else set(SWEEP_METRICS).intersection(metric_selection)
    )
    needs_shepard = (
        metric_selection is None or "shepard_correlation" in metric_selection
    )

    if not requested_k_metrics and not needs_shepard:
        return metrics_payload, diagnostics_payload, []

    n_samples = X_eval.shape[0]
    if requested_k_metrics:
        Q = compute_coranking_matrix(X_eval, X_emb_eval)
        diagnostics_payload["coranking_matrix_"] = Q
        valid_k: List[int] = []
        needs_positive_normalizer = bool(
            {"trustworthiness", "continuity"} & requested_k_metrics
        )
        for k in [n_neighbors] if k_values is None else list(k_values):
            if k <= 0 or k >= (n_samples - 1):
                continue
            if needs_positive_normalizer and (2 * n_samples - 3 * k - 1) <= 0:
                continue
            valid_k.append(k)

        for k in valid_k:
            row_values: Dict[str, float] = {}
            for metric_name, metric_func in (
                ("trustworthiness", trustworthiness),
                ("continuity", continuity),
                ("lcmc", lcmc),
            ):
                if metric_name in requested_k_metrics:
                    row_values[metric_name] = metric_func(Q, k)

            if requested_k_metrics & {"mrre_intrusion", "mrre_extrusion", "mrre_total"}:
                mrre_int, mrre_ext = compute_mrre(Q, k)
                if "mrre_intrusion" in requested_k_metrics:
                    row_values["mrre_intrusion"] = mrre_int
                if "mrre_extrusion" in requested_k_metrics:
                    row_values["mrre_extrusion"] = mrre_ext
                if "mrre_total" in requested_k_metrics:
                    row_values["mrre_total"] = mrre_int + mrre_ext

            if k_values is None:
                metrics_payload.update(row_values)
            for metric_name, value in row_values.items():
                is_num = isinstance(value, (int, float, np.number))
                if is_num and not isinstance(value, bool):
                    records.append(
                        {
                            "method": method_name,
                            "metric": metric_name,
                            "value": float(value),
                            "scope": "global" if k_values is None else "k",
                            "scope_value": "global" if k_values is None else k,
                        }
                    )

    if needs_shepard:
        d_orig, d_emb = shepard_diagram_data(
            X_eval,
            X_emb_eval,
            sample_size=1000,
            random_state=random_state,
        )
        shepard_metrics = {
            "shepard_correlation": float(np.corrcoef(d_orig, d_emb)[0, 1])
            if len(d_orig) > 1
            else np.nan
        }
        metrics_payload.update(shepard_metrics)
        diagnostics_payload["shepard_distances_"] = {
            "original": d_orig,
            "embedded": d_emb,
        }
        for metric_name, value in shepard_metrics.items():
            is_num = isinstance(value, (int, float, np.number))
            if is_num and not isinstance(value, bool):
                records.append(
                    {
                        "method": method_name,
                        "metric": metric_name,
                        "value": float(value),
                        "scope": "global",
                        "scope_value": "global",
                    }
                )

    return metrics_payload, diagnostics_payload, records


def evaluate_embedding(
    X_emb: np.ndarray,
    X: Optional[np.ndarray] = None,
    method_name: str = "embedding",
    metrics: Optional[Sequence[str]] = None,
    labels: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    quality_metadata: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    n_neighbors: int = 5,
    k_values: Optional[Sequence[int]] = None,
    separation_method: str = "centroid",
) -> Dict[str, Any]:
    """
    Evaluate an already computed embedding.

    Parameters
    ----------
    X_emb : np.ndarray
        Embedded data to evaluate.

        - ``(n_samples, n_components)`` triggers standard co-ranking and
          Shepard-style metrics.
        - ``(n_trajectories, n_times, n_dims)`` triggers trajectory metrics.
    X : np.ndarray, optional
        Original data with shape ``(n_samples, n_features)``. Required when
        standard 2D metrics are requested.
    method_name : str, default="embedding"
        Display name attached to tidy metric records.
    metrics : sequence of str, optional
        Metric selectors to compute. ``None`` computes all metrics available for
        the provided inputs.
    labels : np.ndarray, optional
        Optional labels aligned with the embedding. Used by
        ``trajectory_separation`` for native 3D embeddings and by explicit
        supervised 2D metrics such as
        ``separation_logreg_balanced_accuracy`` when requested.
    groups : np.ndarray, optional
        Optional grouping variable aligned with ``X_emb``. Required by
        ``separation_logreg_balanced_accuracy``.
    times : np.ndarray, optional
        Optional trajectory time coordinates used for separation AUC
        integration when trajectory metrics are evaluated.
    quality_metadata : dict, optional
        Scalar quality metadata to attach to the evaluation payload.
    diagnostics : dict, optional
        Precomputed diagnostics to carry through the evaluation payload.
    random_state : int, optional
        Random state used for sampled Shepard distances.
    n_neighbors : int, default=5
        Neighborhood size for single-score standard metrics.
    k_values : sequence of int, optional
        Neighborhood sizes for benchmark sweeps.
    separation_method : str, default="centroid"
        Separation definition passed to ``trajectory_separation`` when
        trajectory labels are available.

    Returns
    -------
    dict
        Dictionary with these keys:

        - ``embedding`` : the evaluated embedding
        - ``metrics`` : scalar metric summaries
        - ``metadata`` : scalar descriptive metadata
        - ``diagnostics`` : array-like or structured diagnostics
        - ``records`` : tidy long-form metric records as ``list[dict]``
        - ``artifacts`` : copy of the diagnostics payload

    Raises
    ------
    TypeError
        If ``quality_metadata`` or ``diagnostics`` is not a dictionary.
    ValueError
        If ``X_emb`` is not 2D or 3D, or if standard 2D evaluation is
        requested without a compatible ``X``.

    Notes
    -----
    This function is intentionally pure. It does not fit reducers, transform
    data, or inspect reducer internals. Callers are responsible for preparing
    ``X_emb`` and any optional metadata such as trajectory labels or times.

    See Also
    --------
    coco_pipe.dim_reduction.core.DimReduction.score
        Manager-level wrapper that prepares inputs and stores the returned
        evaluation payload on a fitted ``DimReduction`` object.
    MethodSelector
        Post-hoc comparison and ranking across multiple scored reductions.

    Examples
    --------
    Evaluate a standard 2D embedding:

    >>> import numpy as np
    >>> X = np.random.RandomState(0).randn(20, 5)
    >>> X_emb = X[:, :2]
    >>> result = evaluate_embedding(X_emb, X=X, method_name="demo")
    >>> "metrics" in result and "records" in result
    True

    Evaluate a native trajectory embedding:

    >>> traj = np.random.RandomState(0).randn(4, 10, 2)
    >>> labels = np.array(["A", "A", "B", "B"])
    >>> result = evaluate_embedding(
    ...     traj,
    ...     method_name="traj",
    ...     metrics=["trajectory_speed", "trajectory_separation"],
    ...     labels=labels,
    ... )
    >>> "trajectory_speed_mean" in result["metrics"]
    True
    """
    X_emb = np.asarray(X_emb)
    if X is not None:
        X = np.asarray(X)
    metric_selection = None if metrics is None else set(metrics)

    standard_metric_names = set(SWEEP_METRICS) | {"shepard_correlation"}
    supervised_metric_names = {SEPARATION_LOGREG_BALANCED_ACCURACY}
    trajectory_metric_names = set(DEFAULT_SCORE_METRICS) - standard_metric_names

    metrics_payload: Dict[str, Any] = {}
    if quality_metadata is None:
        metadata_payload = {}
    elif not isinstance(quality_metadata, dict):
        raise TypeError("Evaluation quality metadata must be a dictionary.")
    else:
        metadata_payload = dict(quality_metadata)

    if diagnostics is None:
        diagnostics_payload = {}
    elif not isinstance(diagnostics, dict):
        raise TypeError("Evaluation diagnostics must be a dictionary.")
    else:
        diagnostics_payload = dict(diagnostics)
    records: List[Dict[str, Any]] = []

    if X_emb.ndim == 2:
        if metric_selection is None:
            standard_selection = standard_metric_names
            supervised_selection = set()
        else:
            standard_selection = metric_selection & standard_metric_names
            supervised_selection = metric_selection & supervised_metric_names

        if standard_selection:
            if X is None:
                raise ValueError(
                    "Original data `X` is required to evaluate standard metrics "
                    "for 2D embeddings."
                )
            if X.ndim != 2 or X.shape[0] != X_emb.shape[0]:
                raise ValueError(
                    "Standard evaluation requires 2D `X` and `X_emb` with matching "
                    "sample counts."
                )

            std_metrics, std_diagnostics, std_records = _evaluate_standard_metrics(
                method_name=method_name,
                X_eval=X,
                X_emb_eval=X_emb,
                metric_selection=standard_selection,
                n_neighbors=n_neighbors,
                k_values=k_values,
                random_state=random_state,
            )
            metrics_payload.update(std_metrics)
            diagnostics_payload.update(std_diagnostics)
            records.extend(std_records)
        if SEPARATION_LOGREG_BALANCED_ACCURACY in supervised_selection:
            if labels is None or groups is None:
                raise ValueError(
                    f"`labels` and `groups` are required for "
                    f"'{SEPARATION_LOGREG_BALANCED_ACCURACY}'."
                )
            separation_score = _cross_validate_score(
                LogisticRegression(max_iter=1000, class_weight="balanced"),
                X_emb,
                labels,
                groups=groups,
                cv_strategy="stratified_group_kfold",
                n_splits=5,
                shuffle=True,
                random_state=42,
                metric="balanced_accuracy",
                use_scaler=True,
            )
            metrics_payload[SEPARATION_LOGREG_BALANCED_ACCURACY] = separation_score
            records.append(
                {
                    "method": method_name,
                    "metric": SEPARATION_LOGREG_BALANCED_ACCURACY,
                    "value": separation_score,
                    "scope": "global",
                    "scope_value": "global",
                }
            )
    elif X_emb.ndim == 3:
        if metric_selection is None:
            metric_selection = trajectory_metric_names
        else:
            metric_selection = metric_selection & trajectory_metric_names

        (traj_metrics, traj_metadata, traj_diagnostics, traj_records) = (
            _evaluate_trajectory_metrics(
                method_name=method_name,
                X_emb=X_emb,
                metric_selection=metric_selection,
                labels=labels,
                times=times,
                separation_method=separation_method,
            )
        )
        metrics_payload.update(traj_metrics)
        metadata_payload.update(traj_metadata)
        diagnostics_payload.update(traj_diagnostics)
        records.extend(traj_records)
    else:
        raise ValueError("`X_emb` must be either 2D or 3D for evaluation.")

    return {
        "embedding": X_emb,
        "metrics": metrics_payload,
        "metadata": metadata_payload,
        "diagnostics": diagnostics_payload,
        "records": list(records),
        "artifacts": diagnostics_payload.copy(),
    }


class MethodSelector:
    """
    Compare and rank already-scored dimensionality reduction methods.

    ``MethodSelector`` is intentionally post-hoc. It does not fit reducers or
    compute embeddings. Each reducer must already be a scored ``DimReduction``
    instance with cached ``metric_records_``.

    Parameters
    ----------
    reducers : dict or list of DimReduction
        Scored ``DimReduction`` objects to compare. Lists are converted to a
        method-keyed mapping using ``reducer.method``.

    Attributes
    ----------
    reducers : dict of str to DimReduction
        Compared reductions keyed by method name.
    metric_records_ : list of dict
        Cached long-form metric records populated by ``collect()``.

    See Also
    --------
    evaluate_embedding
        Pure evaluator used upstream by ``DimReduction.score``.
    coco_pipe.dim_reduction.core.DimReduction.score
        Scores a fitted reduction and populates the records consumed here.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import DimReduction
    >>> X = np.random.RandomState(0).randn(30, 4)
    >>> reducers = [
    ...     DimReduction("PCA", n_components=2),
    ...     DimReduction("Isomap", n_components=2, n_neighbors=5),
    ... ]
    >>> for reducer in reducers:
    ...     embedding = reducer.fit_transform(X)
    ...     reducer.score(embedding, X=X, k_values=[5])
    >>> selector = MethodSelector(reducers).collect()
    >>> frame = selector.to_frame()
    >>> not frame.empty
    True
    """

    def __init__(
        self, reducers: Union[Dict[str, "DimReduction"], List["DimReduction"]]
    ):
        """
        Create a post-hoc comparison layer over scored reductions.

        Parameters
        ----------
        reducers : dict or list of DimReduction
            Already-scored reductions to compare. When a list is provided,
            reducers are keyed by ``reducer.method``.

        Raises
        ------
        TypeError
            If any provided object is not a ``DimReduction`` instance.
        """
        from ..core import DimReduction

        if isinstance(reducers, list):
            validated: Dict[str, DimReduction] = {}
            for reducer in reducers:
                if not isinstance(reducer, DimReduction):
                    raise TypeError(
                        "MethodSelector only accepts scored DimReduction objects. "
                        f"Got {type(reducer).__name__}."
                    )
                validated[reducer.method] = reducer
            self.reducers = validated
        else:
            self.reducers = dict(reducers)
            for name, reducer in self.reducers.items():
                if not isinstance(reducer, DimReduction):
                    raise TypeError(
                        "MethodSelector only accepts scored DimReduction objects. "
                        f"Reducer '{name}' has type {type(reducer).__name__}."
                    )

        self.metric_records_ = []

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> "MethodSelector":
        """Create a selector directly from long-form metric records."""
        selector = cls({})
        selector.metric_records_ = [dict(record) for record in records]
        return selector

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MethodSelector":
        """Create a selector directly from a metric-record DataFrame."""
        return cls.from_records(frame.to_dict(orient="records"))

    def collect(self) -> "MethodSelector":
        """
        Collect cached metric records from already-scored reducers.

        Returns
        -------
        MethodSelector
            The selector populated with comparison-ready metric records.

        Raises
        ------
        ValueError
            If a reducer has not been scored yet.

        See Also
        --------
        coco_pipe.dim_reduction.core.DimReduction.score
            Populates the ``metric_records_`` consumed by this method.
        to_frame
            Materialize the collected long-form records as a DataFrame.

        Notes
        -----
        ``collect()`` does not fit reducers or recompute evaluation metrics.
        It only gathers cached metric observations from reducers that were
        already scored explicitly.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DimReduction
        >>> X = np.random.RandomState(0).randn(20, 4)
        >>> reducer = DimReduction("PCA", n_components=2)
        >>> embedding = reducer.fit_transform(X)
        >>> reducer.score(embedding, X=X, k_values=[5])
        >>> selector = MethodSelector([reducer]).collect()
        >>> len(selector.metric_records_) > 0
        True
        """
        self.metric_records_ = []
        records: List[Dict[str, Any]] = []
        for name, reducer in self.reducers.items():
            if not reducer.metric_records_:
                raise ValueError(
                    f"Reducer '{name}' has no metric records. Call score() first."
                )
            for record in reducer.metric_records_:
                updated = dict(record)
                updated["method"] = name
                records.append(updated)

        self.metric_records_ = records
        return self

    def to_frame(self) -> pd.DataFrame:
        """
        Return the cached long-form metric table.

        Returns
        -------
        pandas.DataFrame
            Tidy metric table with columns ``method``, ``metric``, ``value``,
            ``scope``, and ``scope_value``.

        Notes
        -----
        This method only materializes a DataFrame at the public export
        boundary. Internally, ``MethodSelector`` stores metric records as plain
        Python dictionaries.

        See Also
        --------
        collect
            Gather cached metric records from scored reducers.
        rank_methods
            Rank reducers from the collected metric table.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DimReduction
        >>> X = np.random.RandomState(0).randn(20, 4)
        >>> reducer = DimReduction("PCA", n_components=2)
        >>> embedding = reducer.fit_transform(X)
        >>> reducer.score(embedding, X=X, k_values=[5])
        >>> frame = MethodSelector([reducer]).collect().to_frame()
        >>> set(["method", "metric", "value"]).issubset(frame.columns)
        True
        """
        if not self.metric_records_:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        return pd.DataFrame.from_records(self.metric_records_)

    def rank_methods(
        self,
        selection_metric: str,
        *,
        selection_k: Optional[int] = None,
        tie_breakers: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Rank methods using one primary metric and optional tie-breakers.

        Parameters
        ----------
        selection_metric : str
            Metric to optimize.
        selection_k : int, optional
            Neighborhood size to compare for k-scoped metrics.
        tie_breakers : sequence of str, optional
            Additional metrics used in order when primary values tie.

        Returns
        -------
        pandas.DataFrame
            Ranked comparison table. The first row is the best-scoring method
            under the requested ranking policy.

        Raises
        ------
        ValueError
            If the requested metrics are unsupported, unavailable in the cached
            records, or missing the requested ``selection_k`` observations.

        Notes
        -----
        Ranking is based on mean metric values per method. For k-scoped metrics,
        ``selection_k`` restricts comparison to a single neighborhood size when
        requested.

        See Also
        --------
        collect
            Gather cached metric observations before ranking.
        to_frame
            Inspect the underlying long-form metric observations directly.
        coco_pipe.dim_reduction.core.DimReduction.score
            Produces the metric records that feed into ranking.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DimReduction
        >>> X = np.random.RandomState(0).randn(20, 4)
        >>> reducers = [DimReduction("PCA", n_components=2)]
        >>> reducer = reducers[0]
        >>> embedding = reducer.fit_transform(X)
        >>> reducer.score(embedding, X=X, k_values=[5])
        >>> ranked = MethodSelector(reducers).collect().rank_methods(
        ...     "trustworthiness",
        ...     selection_k=5,
        ... )
        >>> ranked.iloc[0]["method"] == reducer.method
        True
        """
        if selection_metric not in RANKING_DIRECTIONS:
            raise ValueError(
                f"Unsupported selection metric '{selection_metric}'. "
                f"Supported metrics: {sorted(RANKING_DIRECTIONS)}"
            )

        tie_metrics = list(tie_breakers) if tie_breakers is not None else []
        for tie_metric in tie_metrics:
            if tie_metric not in RANKING_DIRECTIONS:
                raise ValueError(
                    f"Unsupported tie-breaker metric '{tie_metric}'. "
                    f"Supported metrics: {sorted(RANKING_DIRECTIONS)}"
                )

        records = self.to_frame()
        if records.empty:
            raise ValueError(
                "No evaluation metrics available. "
                "Score reducers and call collect() first."
            )

        summary = pd.DataFrame(index=sorted(records["method"].unique()))
        comparison_metrics = [selection_metric, *tie_metrics]
        for metric in comparison_metrics:
            metric_df = records[records["metric"] == metric].copy()
            if metric_df.empty:
                raise ValueError(
                    f"Metric '{metric}' is not available in the current results."
                )
            if selection_k is not None and (metric_df["scope"] == "k").any():
                k_numeric = pd.to_numeric(metric_df["scope_value"], errors="coerce")
                metric_df = metric_df[k_numeric == float(selection_k)]
                if metric_df.empty:
                    raise ValueError(
                        f"Metric '{metric}' has no observations at k={selection_k}."
                    )
            summary[metric] = metric_df.groupby("method", dropna=False)["value"].mean()

        summary = summary.reset_index().rename(columns={"index": "method"})
        sort_by = []
        ascending = []
        for metric in comparison_metrics:
            sort_by.append(metric)
            ascending.append(RANKING_DIRECTIONS[metric] == "asc")
        sort_by.append("method")
        ascending.append(True)

        ranked = summary.sort_values(
            sort_by, ascending=ascending, na_position="last"
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        return ranked
