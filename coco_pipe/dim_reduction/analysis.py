"""
Feature Attribution and Analysis
================================

Pure attribution and interpretability utilities for dimensionality reduction.

This module is intentionally separate from the preservation-focused evaluation
stack. The functions here answer a different question:

- ``evaluate_embedding(...)`` in :mod:`coco_pipe.dim_reduction.evaluation`
  asks whether an embedding preserves structure well.
- ``analysis.py`` asks which input features appear to drive an embedding.

The public surface is explicit and array-first:

- ``correlate_features(...)`` computes feature-to-dimension correlations.
- ``perturbation_importance(...)`` measures embedding sensitivity to shuffled
  features.
- ``gradient_importance(...)`` computes encoder saliency for supported
  torch-based reducers.
- ``interpret_features(...)`` is a pure backend that combines one or more of
  these analyses and returns normalized payloads plus tidy records for future
  manager/report integration.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from ..utils import import_optional_dependency

__all__ = [
    "correlate_features",
    "perturbation_importance",
    "gradient_importance",
    "interpret_features",
]


def _analysis_records_from_correlations(
    correlations: Dict[str, Dict[str, float]],
    *,
    method_name: str,
) -> list[Dict[str, Any]]:
    """Flatten nested correlation output into tidy analysis records."""
    records: list[Dict[str, Any]] = []
    for component, feature_scores in correlations.items():
        for feature, value in feature_scores.items():
            records.append(
                {
                    "method": method_name,
                    "analysis": "correlation",
                    "component": component,
                    "feature": feature,
                    "value": float(value),
                }
            )
    return records


def _analysis_records_from_importance(
    scores: Dict[str, float],
    *,
    analysis_name: str,
    method_name: str,
) -> list[Dict[str, Any]]:
    """Flatten feature-importance scores into tidy analysis records."""
    records: list[Dict[str, Any]] = []
    for feature, value in scores.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            records.append(
                {
                    "method": method_name,
                    "analysis": analysis_name,
                    "feature": feature,
                    "value": float(value),
                }
            )
    return records


def correlate_features(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Spearman correlations between original features and embedding axes.

    Parameters
    ----------
    X_orig : np.ndarray
        Original data with shape ``(n_samples, n_features)``.
    X_emb : np.ndarray
        Embedded data with shape ``(n_samples, n_dimensions)``.
    feature_names : sequence of str
        Feature names aligned with the columns of ``X_orig``.

    Returns
    -------
    dict
        Nested mapping of dimension names to feature-correlation mappings,
        sorted by descending absolute correlation magnitude within each
        dimension.

    Raises
    ------
    ValueError
        If ``X_orig`` or ``X_emb`` is not 2D, if sample counts do not match,
        or if ``feature_names`` has the wrong length.

    Notes
    -----
    Constant features or constant embedding dimensions can yield undefined
    Spearman coefficients. These are reported as ``0.0`` to keep the output
    stable and sortable.

    See Also
    --------
    perturbation_importance
        Model-agnostic feature importance by embedding perturbation.
    gradient_importance
        Encoder saliency for supported torch-based reducers.
    interpret_features
        Higher-level backend that packages correlation and importance outputs.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    >>> X_emb = np.array([[0.0, 0.5], [1.0, 0.0], [2.0, 0.5]])
    >>> result = correlate_features(X, X_emb, feature_names=["f1", "f2"])
    >>> sorted(result)
    ['Dimension 1', 'Dimension 2']
    """
    X_orig = np.asarray(X_orig)
    X_emb = np.asarray(X_emb)
    if X_orig.ndim != 2:
        raise ValueError("`X_orig` must be a 2D array.")
    if X_emb.ndim != 2:
        raise ValueError("`X_emb` must be a 2D array.")
    if X_orig.shape[0] != X_emb.shape[0]:
        raise ValueError("`X_orig` and `X_emb` must have matching sample counts.")

    names = list(feature_names)
    if len(names) != X_orig.shape[1]:
        raise ValueError(
            "Length of `feature_names` must match the number of input features."
        )
    results: Dict[str, Dict[str, float]] = {}

    for component_index in range(X_emb.shape[1]):
        component_scores: Dict[str, float] = {}
        for feature_index, feature_name in enumerate(names):
            rho, _ = spearmanr(X_orig[:, feature_index], X_emb[:, component_index])
            if not np.isfinite(rho):
                rho = 0.0
            component_scores[feature_name] = float(rho)

        results[f"Dimension {component_index + 1}"] = dict(
            sorted(
                component_scores.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        )

    return results


def perturbation_importance(
    model: Any,
    X: np.ndarray,
    feature_names: Sequence[str],
    X_emb: np.ndarray,
    n_repeats: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute model-agnostic feature importance by feature shuffling.

    Parameters
    ----------
    model : Any
        Fitted reducer or estimator exposing ``transform(X)``.
    X : np.ndarray
        Input data with shape ``(n_samples, n_features)``.
    feature_names : sequence of str
        Feature names aligned with the columns of ``X``.
    X_emb : np.ndarray
        Explicit embedding of ``X`` used as the perturbation reference.
    n_repeats : int, default=5
        Number of independent shuffles per feature.
    random_state : int, optional
        Random seed for reproducible shuffling.

    Returns
    -------
    dict
        Mapping of feature name to normalized importance score. Scores sum to 1
        when the perturbation signal is nonzero; otherwise all scores are 0.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, if ``X_emb`` does not align with ``X`` along the
        sample axis, or if ``feature_names`` has the wrong length.

    See Also
    --------
    correlate_features
        Cheap feature-to-dimension interpretation based on correlations.
    gradient_importance
        Encoder saliency for supported torch-based reducers.
    interpret_features
        Higher-level backend that packages correlation and importance outputs.

    Examples
    --------
    >>> import numpy as np
    >>> class MockReducer:
    ...     def transform(self, X):
    ...         return X[:, :2]
    >>> X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    >>> X_emb = X[:, :2]
    >>> scores = perturbation_importance(
    ...     MockReducer(),
    ...     X,
    ...     feature_names=["f1", "f2"],
    ...     X_emb=X_emb,
    ...     n_repeats=1,
    ...     random_state=0,
    ... )
    >>> sorted(scores)
    ['f1', 'f2']
    """
    X = np.asarray(X)
    X_emb = np.asarray(X_emb)
    if X.ndim != 2:
        raise ValueError("`X` must be a 2D array.")
    if X_emb.shape[0] != X.shape[0]:
        raise ValueError("`X_emb` must have the same number of samples as `X`.")
    names = list(feature_names)
    if len(names) != X.shape[1]:
        raise ValueError(
            "Length of `feature_names` must match the number of input features."
        )
    rng = np.random.default_rng(random_state)

    original_emb = X_emb

    scores = np.zeros(X.shape[1], dtype=float)
    for feature_index in range(X.shape[1]):
        feature_score = 0.0
        for _ in range(n_repeats):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, feature_index])
            emb_permuted = np.asarray(model.transform(X_permuted))
            feature_score += float(np.mean((original_emb - emb_permuted) ** 2))
        scores[feature_index] = feature_score / float(n_repeats)

    total = float(np.sum(scores))
    if total <= 0 or not np.isfinite(total):
        scores = np.zeros_like(scores, dtype=float)
    else:
        scores = scores / total
    return {name: float(score) for name, score in zip(names, scores)}


def gradient_importance(
    wrapper: Any,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute encoder saliency by differentiating embedding magnitude w.r.t. input.

    Parameters
    ----------
    wrapper : Any
        Fitted encoder-based reducer wrapper exposing
        ``get_pytorch_module()``.
    X : np.ndarray
        Input array. The sample axis is assumed to be axis 0. Remaining axes are
        treated as feature dimensions.
    feature_names : sequence of str, optional
        Feature names for 2D inputs. Named outputs are only supported when the
        reduced saliency is one-dimensional.

    Returns
    -------
    dict
        For one-dimensional reduced saliency with names, returns a mapping of
        feature name to normalized importance score. For higher-dimensional
        saliency, returns ``{"importance_matrix": scores}``.

    Raises
    ------
    ValueError
        If ``X`` has fewer than 2 dimensions, or if ``feature_names`` is
        incompatible with the reduced saliency shape.

    Notes
    -----
    This function assumes an encoder-based torch wrapper that exposes
    ``get_pytorch_module()`` and an ``encoder`` submodule.

    See Also
    --------
    perturbation_importance
        Model-agnostic importance that only requires ``transform``.
    correlate_features
        Cheap feature-to-dimension interpretation from explicit embeddings.
    interpret_features
        Higher-level backend that packages gradient and perturbation outputs.

    Examples
    --------
    >>> import numpy as np
    >>> class Encoder:
    ...     def __call__(self, X):
    ...         return X
    >>> class MockModule:
    ...     def __init__(self):
    ...         self.encoder = Encoder()
    ...     def eval(self):
    ...         return None
    ...     def parameters(self):
    ...         return iter(())
    >>> class MockWrapper:
    ...     def get_pytorch_module(self):
    ...         return MockModule()
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> result = gradient_importance(MockWrapper(), X)
    >>> isinstance(result, dict)
    True
    """
    torch = import_optional_dependency(
        lambda: __import__("torch"),
        feature="gradient_importance",
        dependency="torch",
        install_hint="pip install coco-pipe[topology]",
    )

    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("`X` must have at least 2 dimensions.")

    model = wrapper.get_pytorch_module()
    model.eval()

    parameters = iter(model.parameters())
    try:
        device = next(parameters).device
    except StopIteration:
        device = torch.device("cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    Z = model.encoder(X_tensor)
    Z.sum().backward()

    grads = X_tensor.grad
    scores = torch.mean(torch.abs(grads), dim=0).detach().cpu().numpy()
    total = float(np.sum(scores))
    if total <= 0 or not np.isfinite(total):
        scores = np.zeros_like(scores, dtype=float)
    else:
        scores = scores / total

    if feature_names is None:
        return {"importance_matrix": scores}

    if scores.ndim != 1:
        raise ValueError(
            "Named gradient importance is only supported when the reduced "
            "saliency is one-dimensional."
        )

    names = list(feature_names)
    if len(names) != scores.shape[0]:
        raise ValueError(
            "Length of `feature_names` must match the number of reduced features."
        )
    return {name: float(score) for name, score in zip(names, scores)}


def interpret_features(
    X: np.ndarray,
    *,
    X_emb: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    analyses: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    method_name: str = "embedding",
    n_repeats: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one or more feature interpretation analyses.

    Parameters
    ----------
    X : np.ndarray
        Original input data.
    X_emb : np.ndarray, optional
        Explicit embedding used by correlation-based analysis.
    model : Any, optional
        Fitted reducer or model used by importance analyses.
    analyses : sequence of {"correlation", "perturbation", "gradient"}, optional
        Analyses to compute. ``None`` defaults to ``("correlation",)``.
    feature_names : sequence of str, optional
        Feature names aligned with ``X`` when the requested analysis returns
        feature-keyed outputs.
    method_name : str, default="embedding"
        Display name written into the returned analysis records.
    n_repeats : int, default=5
        Number of permutations per feature for perturbation importance.
    random_state : int, optional
        Random seed for perturbation importance.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``analysis``: nested analysis payloads
        - ``records``: tidy analysis records as ``list[dict]``

    Raises
    ------
    ValueError
        If a requested analysis is unsupported, missing required inputs, or
        lacks required feature names.

    Notes
    -----
    This function is a pure interpretation backend for manager, report, or
    visualization workflows. It does not fit models, compute embeddings, or
    mutate reducer state.

    See Also
    --------
    correlate_features
        Feature-to-dimension interpretation from explicit embeddings.
    perturbation_importance
        Model-agnostic importance based on shuffled features.
    gradient_importance
        Encoder saliency for supported torch-based reducers.

    Examples
    --------
    >>> import numpy as np
    >>> class MockReducer:
    ...     def transform(self, X):
    ...         return X[:, :2]
    >>> X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    >>> X_emb = X[:, :2]
    >>> result = interpret_features(
    ...     X,
    ...     X_emb=X_emb,
    ...     model=MockReducer(),
    ...     analyses=["correlation", "perturbation"],
    ...     feature_names=["f1", "f2"],
    ...     n_repeats=1,
    ...     random_state=0,
    ... )
    >>> sorted(result)
    ['analysis', 'records']
    """
    requested = list(analyses) if analyses is not None else ["correlation"]

    analysis_payload: Dict[str, Any] = {}
    records: list[Dict[str, Any]] = []

    for analysis_name in requested:
        if analysis_name == "correlation":
            if X_emb is None:
                raise ValueError("`X_emb` is required for correlation analysis.")
            if feature_names is None:
                raise ValueError(
                    "`feature_names` is required for correlation analysis."
                )
            result = correlate_features(X, X_emb, feature_names=feature_names)
            analysis_payload["correlation"] = result
            records.extend(
                _analysis_records_from_correlations(
                    result,
                    method_name=method_name,
                )
            )
            continue

        if analysis_name == "perturbation":
            if model is None:
                raise ValueError("`model` is required for perturbation importance.")
            if X_emb is None:
                raise ValueError("`X_emb` is required for perturbation importance.")
            if feature_names is None:
                raise ValueError(
                    "`feature_names` is required for perturbation importance."
                )
            result = perturbation_importance(
                model,
                X,
                feature_names=feature_names,
                X_emb=X_emb,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            analysis_payload["perturbation"] = result
            records.extend(
                _analysis_records_from_importance(
                    result,
                    analysis_name="perturbation",
                    method_name=method_name,
                )
            )
            continue

        if analysis_name == "gradient":
            if model is None:
                raise ValueError("`model` is required for gradient importance.")
            result = gradient_importance(
                model,
                X,
                feature_names=feature_names,
            )
            analysis_payload["gradient"] = result
            if "importance_matrix" not in result and all(
                isinstance(value, (int, float, np.number)) for value in result.values()
            ):
                records.extend(
                    _analysis_records_from_importance(
                        result,
                        analysis_name="gradient",
                        method_name=method_name,
                    )
                )
            continue

        raise ValueError(f"Unknown analysis selector: {analysis_name!r}")

    return {
        "analysis": analysis_payload,
        "records": records,
    }
