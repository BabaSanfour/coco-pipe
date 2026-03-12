"""
Dim-Reduction Visualization Utilities
=====================================

Shared normalization helpers for the explicit, data-first dim-reduction
plotting API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "filter_metric_frame",
    "filter_metrics",
    "infer_metric_plot_type",
    "is_categorical",
    "prepare_embedding_frame",
    "prepare_feature_scores",
    "prepare_interpretation_frame",
    "prepare_metrics_frame",
    "extract_interpretation_matrix",
]


def is_categorical(
    labels: Sequence[Any] | np.ndarray, max_unique_numeric: int = 20
) -> bool:
    """
    Return whether a label array should be treated as categorical.

    Parameters
    ----------
    labels : array-like
        Labels or values to inspect.
    max_unique_numeric : int, default=20
        Maximum number of unique numeric values still treated as categorical.

    Returns
    -------
    bool
        True when the values should be visualized as categories.
    """
    if labels is None:
        return False

    arr = np.asarray(labels)
    if arr.dtype.kind in {"U", "S", "O", "b"}:
        return True

    try:
        valid_mask = ~pd.isna(arr)
        if hasattr(valid_mask, "to_numpy"):
            valid_mask = valid_mask.to_numpy()
        return len(np.unique(arr[valid_mask])) < max_unique_numeric
    except Exception:
        return False


def filter_metrics(scores: Mapping[str, Any]) -> dict[str, float]:
    """
    Keep scalar numeric metrics suitable for plotting.

    Parameters
    ----------
    scores : mapping
        Metric mapping.

    Returns
    -------
    dict
        Numeric scalar metrics only.
    """
    if not scores:
        return {}

    exclude_keys = {"n_iter_", "n_components", "n_components_"}
    filtered: dict[str, float] = {}
    for key, value in scores.items():
        if key in exclude_keys:
            continue
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            filtered[key] = float(value)
    return filtered


def prepare_embedding_frame(
    embedding: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    dimensions: int = 2,
) -> pd.DataFrame:
    """
    Convert embedding coordinates and aligned metadata into a plotting frame.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding array with shape ``(n_samples, n_dimensions)``.
    labels : sequence, optional
        Optional label array aligned with samples.
    metadata : mapping, optional
        Optional column-oriented metadata aligned with samples.
    dimensions : int, default=2
        Number of embedding dimensions to expose in the frame. Must be 2 or 3.

    Returns
    -------
    pandas.DataFrame
        Frame with columns ``x``, ``y`` and optionally ``z`` plus label and
        metadata columns.

    Raises
    ------
    ValueError
        If the embedding is not 2D, if ``dimensions`` is unsupported, or if
        aligned arrays do not match the sample axis.
    """
    embedding = np.asarray(embedding)
    if embedding.ndim != 2:
        raise ValueError("`embedding` must be a 2D array.")
    if dimensions not in {2, 3}:
        raise ValueError("`dimensions` must be 2 or 3.")
    if embedding.shape[1] < dimensions:
        msg = (
            f"`embedding` has only {embedding.shape[1]} dimensions; "
            f"cannot plot {dimensions}."
        )
        raise ValueError(msg)

    n_samples = embedding.shape[0]
    data: dict[str, Any] = {"x": embedding[:, 0], "y": embedding[:, 1]}
    if dimensions == 3:
        data["z"] = embedding[:, 2]

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != n_samples:
            raise ValueError("`labels` must align with the sample axis.")
        if is_categorical(labels):
            valid = labels[~pd.isna(labels)]
            categories = sorted(np.unique(valid).tolist()) if valid.size else []
            data["Label"] = pd.Categorical(labels, categories=categories)
        else:
            data["Label"] = labels

    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("`metadata` must be a mapping of column names to values.")
        for key, value in metadata.items():
            value = np.asarray(value)
            if value.shape[0] != n_samples:
                raise ValueError(f"Metadata column '{key}' must align with samples.")
            data[str(key)] = value

    return pd.DataFrame(data)


def prepare_metrics_frame(metrics: Any, default_method: str = "Method") -> pd.DataFrame:
    """
    Normalize metric inputs into a tidy long-form frame.

    Output columns always include:
    ``method``, ``metric``, ``value``, ``scope``, and ``scope_value``.
    """
    empty = pd.DataFrame(columns=["method", "metric", "value", "scope", "scope_value"])
    if metrics is None:
        return empty

    if hasattr(metrics, "to_frame") and callable(getattr(metrics, "to_frame")):
        metrics = metrics.to_frame()

    if isinstance(metrics, list):
        metrics = pd.DataFrame.from_records(metrics)

    if isinstance(metrics, Mapping):
        records = [
            {
                "method": default_method,
                "metric": key,
                "value": value,
                "scope": "global",
                "scope_value": "global",
            }
            for key, value in filter_metrics(metrics).items()
        ]
        return pd.DataFrame.from_records(records) if records else empty

    if not isinstance(metrics, pd.DataFrame):
        raise TypeError(
            "Metrics must be a mapping, DataFrame, list of records, "
            "or object exposing to_frame()."
        )

    df = metrics.copy()
    if df.empty:
        return empty

    long_cols = {"method", "metric", "value"}
    if long_cols.issubset(df.columns):
        if "scope" not in df.columns:
            df["scope"] = "global"
        if "scope_value" not in df.columns:
            df["scope_value"] = "global"
        return df

    if "method" not in df.columns:
        index_name = df.index.name or "method"
        df = df.reset_index().rename(columns={index_name: "method"})
    if "method" not in df.columns:
        df["method"] = default_method

    id_cols = [
        col
        for col in [
            "method",
            "scope",
            "scope_value",
            "group",
            "condition",
            "pair",
            "subject",
            "session",
            "seed",
            "fold",
        ]
        if col in df.columns
    ]
    metric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in id_cols
    ]
    if not metric_cols:
        return empty

    long_df = df.melt(
        id_vars=id_cols or ["method"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    if "scope" not in long_df.columns:
        long_df["scope"] = "global"
    if "scope_value" not in long_df.columns:
        long_df["scope_value"] = "global"
    return long_df


def _flatten_interpretation_payload(
    payload: Mapping[str, Any], default_method: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for analysis_name, analysis_payload in payload.items():
        if analysis_name == "correlation" and isinstance(analysis_payload, Mapping):
            for dimension, feature_scores in analysis_payload.items():
                if not isinstance(feature_scores, Mapping):
                    continue
                for feature, value in feature_scores.items():
                    if isinstance(value, (int, float, np.number)):
                        records.append(
                            {
                                "method": default_method,
                                "analysis": "correlation",
                                "dimension": str(dimension),
                                "feature": str(feature),
                                "value": float(value),
                            }
                        )
            continue

        if isinstance(analysis_payload, Mapping):
            for feature, value in analysis_payload.items():
                if isinstance(value, (int, float, np.number)):
                    records.append(
                        {
                            "method": default_method,
                            "analysis": str(analysis_name),
                            "feature": str(feature),
                            "value": float(value),
                        }
                    )
    return records


def prepare_interpretation_frame(
    interpretation: Any, default_method: str = "embedding"
) -> pd.DataFrame:
    """
    Normalize interpretation payloads and records into a tidy frame.

    Parameters
    ----------
    interpretation : Any
        One of:

        - interpretation result with ``records`` or ``analysis`` keys
        - list of interpretation records
        - tidy interpretation DataFrame
        - raw correlation payload
        - raw feature-importance mapping
    default_method : str, default="embedding"
        Method name used when raw payloads do not contain one.

    Returns
    -------
    pandas.DataFrame
        Tidy frame with at least ``method``, ``analysis``, ``feature``, and
        ``value`` columns. Correlation rows also carry ``dimension``.
    """
    empty = pd.DataFrame(
        columns=["method", "analysis", "feature", "value", "dimension"]
    )
    if interpretation is None:
        return empty

    if isinstance(interpretation, list):
        interpretation = pd.DataFrame.from_records(interpretation)

    if isinstance(interpretation, pd.DataFrame):
        df = interpretation.copy()
        if "component" in df.columns and "dimension" not in df.columns:
            df = df.rename(columns={"component": "dimension"})
        required = {"method", "analysis", "feature", "value"}
        if not required.issubset(df.columns):
            msg = (
                "Interpretation records must include "
                "method, analysis, feature, and value."
            )
            raise ValueError(msg)
        if "dimension" not in df.columns:
            df["dimension"] = None
        return df

    if isinstance(interpretation, Mapping):
        if "records" in interpretation:
            return prepare_interpretation_frame(
                interpretation["records"], default_method=default_method
            )

        if "analysis" in interpretation and isinstance(
            interpretation["analysis"], Mapping
        ):
            records = _flatten_interpretation_payload(
                interpretation["analysis"], default_method=default_method
            )
            return pd.DataFrame.from_records(records) if records else empty

        if "correlation" in interpretation and isinstance(
            interpretation["correlation"], Mapping
        ):
            records = _flatten_interpretation_payload(
                {"correlation": interpretation["correlation"]},
                default_method=default_method,
            )
            return pd.DataFrame.from_records(records) if records else empty

        numeric_items = {
            str(key): float(value)
            for key, value in interpretation.items()
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        }
        if numeric_items:
            records = [
                {
                    "method": default_method,
                    "analysis": "importance",
                    "feature": key,
                    "value": value,
                    "dimension": None,
                }
                for key, value in numeric_items.items()
            ]
            return pd.DataFrame.from_records(records)

    raise TypeError(
        "Interpretation data must be a mapping, DataFrame, or list of records."
    )


def prepare_feature_scores(
    interpretation: Any,
    *,
    analysis: Optional[str] = None,
    method: Optional[str] = None,
    dimension: Optional[str] = None,
) -> pd.Series:
    """
    Reduce interpretation payloads to one feature-score series.

    Parameters
    ----------
    interpretation : Any
        Raw feature-score mapping or normalized interpretation payload.
    analysis : str, optional
        Interpretation analysis to keep when multiple analyses are present.
    method : str, optional
        Method name to keep when multiple methods are present.
    dimension : str, optional
        Dimension label to keep when the interpretation contains multiple
        dimensions.

    Returns
    -------
    pandas.Series
        Feature scores indexed by feature name, sorted descending.

    Raises
    ------
    ValueError
        If multiple analyses, methods, or dimensions are present and no
        explicit selector is provided.
    """
    if (
        isinstance(interpretation, Mapping)
        and interpretation
        and all(
            isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            for value in interpretation.values()
        )
    ):
        return pd.Series(
            {str(key): float(value) for key, value in interpretation.items()}
        ).sort_values(ascending=False)

    df = prepare_interpretation_frame(interpretation)
    if df.empty:
        raise ValueError("No interpretation records available to plot.")

    if analysis is None:
        analyses = df["analysis"].dropna().unique()
        if len(analyses) != 1:
            raise ValueError("Specify `analysis` when multiple analyses are present.")
        analysis = str(analyses[0])
    df = df[df["analysis"] == analysis]

    if method is None:
        methods = df["method"].dropna().unique()
        if len(methods) > 1:
            raise ValueError("Specify `method` when multiple methods are present.")
    else:
        df = df[df["method"] == method]

    if "dimension" in df.columns and df["dimension"].notna().any():
        if dimension is None and df["dimension"].dropna().nunique() > 1:
            raise ValueError(
                "Specify `dimension` when multiple dimensions are present."
            )
        if dimension is not None:
            df = df[df["dimension"] == dimension]

    if df.empty:
        raise ValueError("No interpretation rows remain after filtering.")

    return (
        df.groupby("feature", dropna=False)["value"].mean().sort_values(ascending=False)
    )


def extract_interpretation_matrix(
    interpretation: Any,
    *,
    analysis: str,
) -> Optional[np.ndarray]:
    """
    Extract a matrix-style interpretation payload when one is available.

    Parameters
    ----------
    interpretation : Any
        Interpretation payload or wrapper containing an ``analysis`` mapping.
    analysis : str
        Analysis key to extract.

    Returns
    -------
    np.ndarray or None
        Interpretation matrix when present, otherwise ``None``.
    """
    payload = interpretation
    if isinstance(payload, Mapping) and "analysis" in payload:
        payload = payload["analysis"]
    if isinstance(payload, Mapping) and analysis in payload:
        payload = payload[analysis]
    if isinstance(payload, Mapping) and "importance_matrix" in payload:
        return np.asarray(payload["importance_matrix"])
    return None


def filter_metric_frame(
    metrics_df: pd.DataFrame,
    *,
    metric: Optional[str] = None,
    scope: Optional[str] = None,
    method: Optional[str | Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Filter a tidy metric frame by metric, scope, and method.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Tidy metric frame from :func:`prepare_metrics_frame`.
    metric : str, optional
        Metric name to keep.
    scope : str, optional
        Scope name to keep.
    method : str or sequence of str, optional
        Method name or names to keep.

    Returns
    -------
    pandas.DataFrame
        Filtered frame.
    """
    df = metrics_df.copy()
    if metric is not None:
        df = df[df["metric"] == metric]
    if scope is not None:
        df = df[df["scope"] == scope]
    if method is not None:
        allowed = {method} if isinstance(method, str) else set(method)
        df = df[df["method"].isin(allowed)]
    return df


def infer_metric_plot_type(
    metrics_df: pd.DataFrame, requested: Optional[str] = None
) -> str:
    """
    Choose a sensible default plot type from filtered metric observations.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Already-filtered tidy metric frame.
    requested : str, optional
        Explicit requested plot type.

    Returns
    -------
    str
        Resolved plot type.
    """
    if requested and requested not in {"auto", None}:
        return requested
    if metrics_df.empty:
        return "bar"

    scope_values = metrics_df["scope_value"].dropna().astype(str).nunique()
    if scope_values > 1:
        return "line"

    repetitions = metrics_df.groupby(["method", "metric"]).size()
    if not repetitions.empty and repetitions.max() > 1:
        return "raincloud"

    if metrics_df["method"].nunique() > 1 and metrics_df["metric"].nunique() > 1:
        return "grouped_bar"

    return "bar"
