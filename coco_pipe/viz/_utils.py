from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .theme import ColorKind

if TYPE_CHECKING:
    from coco_pipe.decoding.result import ExperimentResult


class SensorLayout(NamedTuple):
    names: list[str]
    positions: np.ndarray


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Return a valid-mode moving average for one-dimensional values."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("moving average input must be one-dimensional.")
    if window <= 0:
        raise ValueError("smooth_window must be a positive integer.")
    if window > values.shape[0]:
        raise ValueError("smooth_window cannot exceed the trajectory time axis.")
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def finalize_axes(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    xtick_rotation: float | None = None,
    xtick_ha: str | None = None,
    tick_nbins: int | None = None,
) -> plt.Axes:
    """Apply common axis labels, title, legend, and tick formatting."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    if tick_nbins is not None:
        locator = ticker.MaxNLocator(nbins=tick_nbins)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_nbins))
        if hasattr(ax, "zaxis"):
            ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_nbins))
    if xtick_rotation is not None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(xtick_rotation)
    if xtick_ha is not None:
        for tick in ax.get_xticklabels():
            tick.set_ha(xtick_ha)
    if legend:
        ax.legend(title=legend_title, frameon=False)
    return ax


def select_dimensions(
    data: np.ndarray,
    dims: Sequence[int],
    allowed: set[int] | None = None,
    context: str = "data",
) -> np.ndarray:
    """Validate a 2D array and return selected columns."""
    dims = tuple(dims)
    allowed = {2, 3} if allowed is None else allowed
    if len(dims) not in allowed:
        expected = " or ".join(str(value) for value in sorted(allowed))
        raise ValueError(f"`dims` must contain {expected} dimensions.")
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"`{context}` must be a 2D array.")
    if min(dims) < 0 or values.shape[1] <= max(dims):
        raise ValueError("`dims` must reference valid columns.")
    return values[:, list(dims)]


def get_figure(
    ax: plt.Axes | None,
    figsize: tuple[float, float] | None,
    default: tuple[float, float],
    projection: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Return an existing figure/axis pair or create one with optional projection."""
    if ax is not None:
        return ax.get_figure(), ax
    if projection is None:
        return plt.subplots(figsize=figsize or default)
    fig = plt.figure(figsize=figsize or default)
    return fig, fig.add_subplot(111, projection=projection)


def coerce_decoding_frame(
    obj: "ExperimentResult | pd.DataFrame",
    accessor: str | None = None,
) -> pd.DataFrame:
    """
    Return a tidy decoding frame from an ExperimentResult or DataFrame.

    Unknown objects are rejected explicitly; plot functions should not silently
    construct arbitrary DataFrames.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if accessor is not None and hasattr(obj, accessor):
        method = getattr(obj, accessor)
        if not callable(method):
            raise TypeError(f"{accessor!r} is not callable on {type(obj).__name__}.")
        frame = method()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{accessor}() must return a pandas DataFrame.")
        return frame.copy()
    if accessor is None:
        raise TypeError("accessor is required when coercing an ExperimentResult.")
    raise TypeError(
        f"Expected pandas DataFrame or object with {accessor}(); got "
        f"{type(obj).__name__}."
    )


def coerce_reduction_frame(
    obj: Any,
    accessor: str | None = None,
    prepare_fn: Callable[[Any], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Return a tidy frame from a DimReduction result or compatible input.

    Resolution order:
    1. obj is a DataFrame → return copy directly.
    2. accessor is given and obj.{accessor} is callable -> call it, validate
       the return is a DataFrame.
    3. accessor is given and obj.{accessor} is a non-callable attribute
       (e.g. a dict or list) -> pass the attribute value to prepare_fn.
    4. accessor is absent and prepare_fn is given -> pass obj directly to prepare_fn.
    5. accessor is None and prepare_fn is given -> pass obj directly to prepare_fn.
    6. All other cases -> raise TypeError with a clear message.

    Never return an empty DataFrame silently for unrecognised inputs.
    prepare_fn must itself raise on None/unknown input (see §1.2 and §1.3).
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if accessor is not None:
        if not hasattr(obj, accessor):
            if prepare_fn is not None:
                return prepare_fn(obj)
            raise TypeError(
                f"Expected a DataFrame or object with {accessor!r} attribute; "
                f"got {type(obj).__name__}."
            )
        attr = getattr(obj, accessor)
        if callable(attr):
            result = attr()
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"{accessor}() must return a pandas DataFrame; "
                    f"got {type(result).__name__}."
                )
            return result.copy()
        # Non-callable attribute (e.g. metrics_ dict, interpretation_ dict)
        if prepare_fn is None:
            raise TypeError(
                f"{accessor!r} is not callable and no prepare_fn was provided."
            )
        return prepare_fn(attr)

    if prepare_fn is not None:
        return prepare_fn(obj)

    raise TypeError(
        f"Cannot coerce {type(obj).__name__} to a DataFrame. "
        "Pass a DataFrame, a DimReduction result with an accessor, "
        "or a raw metrics/interpretation payload."
    )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    context: str,
) -> pd.DataFrame:
    """Raise ValueError listing missing columns and the context string."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
    return df


def require_non_empty(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Raise a clear error when filtering leaves no rows."""
    if df.empty:
        raise ValueError(f"No rows available for {context} after filtering.")
    return df


def _scalar_metrics(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Return only global (non-scope-varying) rows for scalar-only plot types."""
    require_columns(frame, ["Method", "Metric", "Value"], context=context)
    if "Scope" in frame.columns:
        data = frame[frame["Scope"] == "global"].copy()
    else:
        data = frame.copy()
    return require_non_empty(data, context)


def _single_method(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Require exactly one method for single-panel plots."""
    if "Method" not in frame.columns:
        return frame
    methods = frame["Method"].dropna().unique().tolist()
    if len(methods) > 1:
        raise ValueError(
            f"{context} requires a single method selection. "
            f"Filter with method=... Found: {methods}"
        )
    return frame


def select_rows(
    df: pd.DataFrame,
    model: str | None = None,
    metric: str | None = None,
    fold: int | None = None,
    group: str | None = None,
) -> pd.DataFrame:
    """Filter DataFrame by any combination of model, metric, fold, group."""
    frame = df.copy()
    filters = {
        "Model": model,
        "Metric": metric,
        "Fold": fold,
        "Group": group,
    }
    for column, value in filters.items():
        if value is not None and column in frame.columns:
            frame = frame[frame[column] == value]
    return frame


def select_reduction_rows(
    df: pd.DataFrame,
    method: str | None = None,
    metric: str | None = None,
    scope: str | None = None,
    analysis: str | None = None,
) -> pd.DataFrame:
    """Filter a tidy reduction frame by any combination of columns."""
    frame = df.copy()
    filters = {
        "Method": method,
        "Metric": metric,
        "Scope": scope,
        "Analysis": analysis,
    }
    for column, value in filters.items():
        if value is not None and column in frame.columns:
            frame = frame[frame[column] == value]
    return frame


def _coerce_series(
    data: pd.Series | Mapping[str, float] | Sequence[float],
    index: Sequence[str] | None = None,
) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    if isinstance(data, Mapping):
        return pd.Series(data).dropna()
    values = np.asarray(list(data), dtype=float)
    idx = list(index) if index is not None else [str(i) for i in range(len(values))]
    return pd.Series(values, index=idx).dropna()


def coerce_sensor_layout(
    info=None,
    coords=None,
    names: list[str] | None = None,
) -> SensorLayout:
    """
    Normalize MNE info, coordinate mapping, or coordinate DataFrame inputs.
    """
    requested = set(names) if names is not None else None

    if info is not None:
        try:
            import mne

            picks = mne.pick_types(info, eeg=True, meg=True, misc=False, exclude=[])
            ch_names = [info["ch_names"][idx] for idx in picks]
            positions = []
            kept = []
            for idx, ch_name in zip(picks, ch_names):
                if requested is not None and ch_name not in requested:
                    continue
                loc = np.asarray(info["chs"][idx]["loc"][:2], dtype=float)
                if np.isfinite(loc).all():
                    kept.append(ch_name)
                    positions.append(loc)
            if kept:
                return SensorLayout(kept, np.asarray(positions, dtype=float))
        except Exception as exc:
            raise ValueError(
                f"Could not construct sensor layout from info: {exc}"
            ) from exc

    if coords is None:
        raise ValueError("A sensor layout requires either info or coords.")

    if isinstance(coords, pd.DataFrame):
        lower = {str(col).lower(): col for col in coords.columns}
        if "x" not in lower or "y" not in lower:
            raise ValueError("coords DataFrame must include x and y columns.")
        if "FeatureName" in coords.columns:
            coord_names = coords["FeatureName"].astype(str).tolist()
        elif "Sensor" in coords.columns:
            coord_names = coords["Sensor"].astype(str).tolist()
        else:
            coord_names = coords.index.astype(str).tolist()
        frame = coords.copy()
        if requested is not None:
            mask = pd.Series(coord_names).isin(requested).to_numpy()
            frame = frame.loc[mask]
            coord_names = [name for name in coord_names if name in requested]
        return SensorLayout(
            coord_names,
            frame[[lower["x"], lower["y"]]].to_numpy(dtype=float),
        )

    if isinstance(coords, Mapping):
        items = [(str(name), value) for name, value in coords.items()]
        if requested is not None:
            items = [(name, value) for name, value in items if name in requested]
        if not items:
            raise ValueError("No matching sensor coordinates were found.")
        coord_names = [name for name, _ in items]
        positions = np.asarray([value[:2] for _, value in items], dtype=float)
        return SensorLayout(coord_names, positions)

    positions = np.asarray(coords, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("coords must be an Nx2 coordinate array.")
    coord_names = (
        list(names) if names is not None else [str(i) for i in range(len(positions))]
    )
    if len(coord_names) != len(positions):
        raise ValueError("names length must match coordinate rows.")
    return SensorLayout(coord_names, positions[:, :2])


def prepare_embedding_frame(
    embedding: np.ndarray,
    labels: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    dimensions: int = 2,
    label_kind: ColorKind = "categorical",
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
    label_kind : {"categorical", "continuous"}, default="categorical"
        How to represent ``labels`` for downstream color mapping.

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
        if label_kind not in {"categorical", "continuous"}:
            raise ValueError("`label_kind` must be 'categorical' or 'continuous'.")
        labels = np.asarray(labels)
        if labels.shape[0] != n_samples:
            raise ValueError("`labels` must align with the sample axis.")
        if label_kind == "categorical":
            valid = labels[~pd.isna(labels)]
            categories = sorted(np.unique(valid).tolist()) if valid.size else []
            data["Label"] = pd.Categorical(labels, categories=categories)
        else:
            data["Label"] = pd.to_numeric(pd.Series(labels), errors="raise").to_numpy()

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
    output_columns = ["Method", "Metric", "Value", "Scope", "ScopeValue"]
    empty = pd.DataFrame(columns=output_columns)
    if metrics is None:
        raise TypeError(
            "`metrics` must be a mapping of scalar values, DataFrame, list of records, "
            "or object exposing to_frame(). Got None."
        )

    if isinstance(metrics, Mapping):
        ignored_metric_keys = {"n_iter_", "n_components", "n_components_"}
        records: list[dict[str, Any]] = []
        for key, value in metrics.items():
            if key in ignored_metric_keys:
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                continue
            records.append(
                {
                    "Method": default_method,
                    "Metric": str(key),
                    "Value": float(value),
                    "Scope": "global",
                    "ScopeValue": "global",
                }
            )
        return pd.DataFrame.from_records(records) if records else empty

    if isinstance(metrics, list):
        metrics = pd.DataFrame.from_records(metrics)
    elif hasattr(metrics, "to_frame") and callable(getattr(metrics, "to_frame")):
        metrics = metrics.to_frame()

    if not isinstance(metrics, pd.DataFrame):
        raise TypeError(
            "`metrics` must be one of: mapping of scalar metric values, "
            "pandas DataFrame, list of metric records, or object exposing "
            "to_frame()."
        )

    df = metrics.copy().rename(
        columns={
            "method": "Method",
            "metric": "Metric",
            "value": "Value",
            "scope": "Scope",
            "scope_value": "ScopeValue",
        }
    )
    if df.empty:
        return empty

    long_cols = {"Method", "Metric", "Value"}
    if long_cols.issubset(df.columns):
        if "Scope" not in df.columns:
            df["Scope"] = "global"
        if "ScopeValue" not in df.columns:
            df["ScopeValue"] = "global"
        return df

    if "Method" not in df.columns:
        index_name = df.index.name or "Method"
        df = df.reset_index().rename(columns={index_name: "Method"})
    if "Method" not in df.columns:
        df["Method"] = default_method

    id_cols = [
        col
        for col in [
            "Method",
            "Scope",
            "ScopeValue",
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
        id_vars=id_cols or ["Method"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    )
    if "Scope" not in long_df.columns:
        long_df["Scope"] = "global"
    if "ScopeValue" not in long_df.columns:
        long_df["ScopeValue"] = "global"
    return long_df


def _records_from_interpretation_payload(
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
                                "Method": default_method,
                                "Analysis": "correlation",
                                "Dimension": str(dimension),
                                "Feature": str(feature),
                                "Value": float(value),
                            }
                        )
            continue

        if isinstance(analysis_payload, Mapping):
            for feature, value in analysis_payload.items():
                if isinstance(value, (int, float, np.number)):
                    records.append(
                        {
                            "Method": default_method,
                            "Analysis": str(analysis_name),
                            "Feature": str(feature),
                            "Value": float(value),
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
    output_columns = ["Method", "Analysis", "Feature", "Value", "Dimension"]
    required_columns = {"Method", "Analysis", "Feature", "Value"}
    empty = pd.DataFrame(columns=output_columns)
    if interpretation is None:
        raise TypeError(
            "`interpretation` must be a mapping payload, DataFrame, "
            "or list of records. Got None."
        )

    if isinstance(interpretation, list):
        interpretation = pd.DataFrame.from_records(interpretation)

    if isinstance(interpretation, pd.DataFrame):
        df = interpretation.copy().rename(
            columns={
                "method": "Method",
                "analysis": "Analysis",
                "feature": "Feature",
                "value": "Value",
                "dimension": "Dimension",
                "component": "Dimension",
            }
        )
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            msg = f"Interpretation records are missing required columns: {missing}."
            raise ValueError(msg)
        if "Dimension" not in df.columns:
            df["Dimension"] = None
        return df

    if isinstance(interpretation, Mapping):
        if "records" in interpretation:
            records = interpretation["records"]
            return prepare_interpretation_frame(records, default_method=default_method)

        analysis_payload = interpretation.get("analysis")
        if isinstance(analysis_payload, Mapping):
            records = _records_from_interpretation_payload(
                analysis_payload, default_method=default_method
            )
            return pd.DataFrame.from_records(records) if records else empty

        correlation_payload = interpretation.get("correlation")
        if isinstance(correlation_payload, Mapping):
            records = _records_from_interpretation_payload(
                {"correlation": correlation_payload},
                default_method=default_method,
            )
            return pd.DataFrame.from_records(records) if records else empty

        records = []
        for feature, value in interpretation.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                continue
            records.append(
                {
                    "Method": default_method,
                    "Analysis": "importance",
                    "Feature": str(feature),
                    "Value": float(value),
                    "Dimension": None,
                }
            )
        if records:
            return pd.DataFrame.from_records(records)

    raise TypeError(
        "`interpretation` must be one of: mapping payload, pandas DataFrame, "
        "or list of interpretation records."
    )


def prepare_feature_scores(
    interpretation: Any,
    analysis: str | None = None,
    method: str | None = None,
    dimension: str | None = None,
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
    if isinstance(interpretation, Mapping) and interpretation:
        numeric_scores: dict[str, float] = {}
        all_values_are_numeric = True
        for feature, value in interpretation.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                all_values_are_numeric = False
                break
            numeric_scores[str(feature)] = float(value)
        if all_values_are_numeric:
            return pd.Series(numeric_scores).sort_values(ascending=False)

    df = prepare_interpretation_frame(interpretation)
    if df.empty:
        raise ValueError("No interpretation records available to plot.")

    if analysis is None:
        analyses = sorted(df["Analysis"].dropna().astype(str).unique().tolist())
        if len(analyses) != 1:
            raise ValueError(
                f"Specify `analysis` when multiple analyses are present: {analyses}."
            )
        analysis = analyses[0]
    df = df[df["Analysis"].astype(str) == str(analysis)]

    if method is None:
        methods = sorted(df["Method"].dropna().astype(str).unique().tolist())
        if len(methods) > 1:
            raise ValueError(
                f"Specify `method` when multiple methods are present: {methods}."
            )
    else:
        df = df[df["Method"].astype(str) == str(method)]

    if "Dimension" in df.columns and df["Dimension"].notna().any():
        dimensions = sorted(df["Dimension"].dropna().astype(str).unique().tolist())
        if dimension is None and len(dimensions) > 1:
            raise ValueError(
                "Specify `dimension` when multiple dimensions are present: "
                f"{dimensions}."
            )
        if dimension is not None:
            df = df[df["Dimension"].astype(str) == str(dimension)]

    if df.empty:
        raise ValueError("No interpretation rows remain after filtering.")

    values = pd.to_numeric(df["Value"], errors="coerce")
    score_frame = df.assign(Value=values).dropna(subset=["Value"])
    if score_frame.empty:
        raise ValueError("No numeric interpretation values remain after filtering.")

    return (
        score_frame.groupby("Feature", dropna=False)["Value"]
        .mean()
        .sort_values(ascending=False)
    )


def filter_metric_frame(
    metrics_df: pd.DataFrame,
    metric: str | None = None,
    scope: str | None = None,
    method: str | Sequence[str] | None = None,
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
    require_columns(metrics_df, ["Method", "Metric", "Scope"], context="metrics frame")
    df = metrics_df.copy()
    if metric is not None:
        df = df[df["Metric"].astype(str) == str(metric)]
    if scope is not None:
        df = df[df["Scope"].astype(str) == str(scope)]
    if method is not None:
        allowed_methods = {method} if isinstance(method, str) else set(method)
        allowed_methods = {str(value) for value in allowed_methods}
        df = df[df["Method"].astype(str).isin(allowed_methods)]
    return df


def prepare_trajectory_metric_series(
    series: Any,
    times: Sequence[float] | np.ndarray | None = None,
    labels: Sequence[Any] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Normalize trajectory metric series into tidy plotting rows."""
    if isinstance(series, Mapping):
        if not series:
            raise ValueError("No trajectory series available to plot.")
        lengths = {len(np.asarray(values).reshape(-1)) for values in series.values()}
        if len(lengths) != 1:
            raise ValueError("All trajectory series must share the same length.")
        n_times = lengths.pop()
        x_vals = np.arange(n_times) if times is None else np.asarray(times)
        if len(x_vals) != n_times:
            raise ValueError("`times` must align with the trajectory time axis.")
        records = []
        for name, values in series.items():
            for time, value in zip(x_vals, np.asarray(values, dtype=float).reshape(-1)):
                records.append(
                    {
                        "Series": str(name),
                        "Time": time,
                        "Value": float(value),
                        "Error": np.nan,
                    }
                )
        return pd.DataFrame.from_records(records)

    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        raise ValueError("trajectory metric series is empty.")
    if arr.ndim == 1:
        x_vals = np.arange(arr.shape[0]) if times is None else np.asarray(times)
        if len(x_vals) != arr.shape[0]:
            raise ValueError("`times` must align with the trajectory time axis.")
        return pd.DataFrame(
            {
                "Series": "Metric",
                "Time": x_vals,
                "Value": arr,
                "Error": np.nan,
            }
        )
    if arr.ndim != 2:
        raise ValueError("Trajectory metric series must be 1D, 2D, or a dict.")

    x_vals = np.arange(arr.shape[1]) if times is None else np.asarray(times)
    if len(x_vals) != arr.shape[1]:
        raise ValueError("`times` must align with the trajectory time axis.")
    if labels is None:
        labels = np.repeat("Metric", arr.shape[0])
    else:
        labels = np.asarray(labels)
        if labels.shape[0] != arr.shape[0]:
            raise ValueError("`labels` must align with the series axis.")

    records = []
    for label in dict.fromkeys(np.asarray(labels).tolist()):
        subset = arr[np.asarray(labels) == label]
        finite_count = np.sum(np.isfinite(subset), axis=0)
        mean = np.full(subset.shape[1], np.nan, dtype=float)
        has_values = finite_count > 0
        mean[has_values] = np.nanmean(subset[:, has_values], axis=0)
        err = np.full(subset.shape[1], np.nan, dtype=float)
        valid = finite_count > 1
        err[valid] = np.nanstd(subset[:, valid], axis=0) / np.sqrt(finite_count[valid])
        for time, value, error in zip(x_vals, mean, err):
            records.append(
                {
                    "Series": str(label),
                    "Time": time,
                    "Value": float(value),
                    "Error": float(error) if np.isfinite(error) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def _scalar_decoding_scores(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Return non-temporal scalar score rows with required decoding score columns."""
    require_columns(frame, ["Model", "Fold", "Metric", "Value"], context=context)
    data = frame[frame["Value"].notna()].copy()
    for column in ["Time", "TrainTime", "TestTime"]:
        if column in data:
            data = data[data[column].isna()]
    return require_non_empty(data, context)


def _single_model_metric(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Require one model/metric pair for decoding plots that render a single panel."""
    keys = [column for column in ["Model", "Metric"] if column in frame]
    if not keys:
        return frame
    pairs = frame[keys].drop_duplicates()
    if len(pairs) > 1:
        details = ", ".join(
            " / ".join(str(row[column]) for column in keys)
            for _, row in pairs.head(5).iterrows()
        )
        raise ValueError(
            f"{context} requires a single model/metric selection. "
            f"Filter with model=... and metric=.... Found: {details}"
        )
    return frame


def _prediction_accuracy(group: pd.DataFrame) -> float:
    """Return accuracy for a grouped prediction frame."""
    return float(np.mean(group["y_true"].to_numpy() == group["y_pred"].to_numpy()))


def prepare_confusion_matrix(
    result_or_matrix: Any,
    model: str | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    """Return an aggregated true-label by predicted-label confusion matrix."""
    frame = coerce_decoding_frame(result_or_matrix, accessor="get_confusion_matrices")
    require_columns(
        frame,
        ["TrueLabel", "PredictedLabel", "Value"],
        context="get_confusion_matrices",
    )
    frame = select_rows(frame, model=model, fold=fold)
    require_non_empty(frame, "confusion matrix")
    return frame.pivot_table(
        index="TrueLabel",
        columns="PredictedLabel",
        values="Value",
        aggfunc="sum",
        fill_value=0,
    )


def _importance_with_metadata(
    result: Any,
    feature_metadata: pd.DataFrame,
    model: str | None = None,
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge feature importances with explicit feature metadata.

    The returned frame contains all metadata columns plus ``_ImportanceValue``,
    a numeric normalized importance column used by sensor-level plots.
    """
    required_metadata = metadata_columns or ["FeatureName"]
    if "FeatureName" not in required_metadata:
        required_metadata = ["FeatureName", *required_metadata]
    imp = coerce_decoding_frame(result, accessor="get_feature_importances")
    imp = select_rows(imp, model=model)
    value_col = "Mean" if "Mean" in imp.columns else "Importance"
    require_columns(imp, ["FeatureName", value_col], context="get_feature_importances")
    require_columns(feature_metadata, required_metadata, context="feature_metadata")
    merged = imp.merge(feature_metadata, on="FeatureName", how="inner")
    require_non_empty(merged, "feature metadata merge")
    merged["_ImportanceValue"] = pd.to_numeric(merged[value_col], errors="coerce")
    require_non_empty(
        merged[merged["_ImportanceValue"].notna()],
        "numeric feature importance metadata",
    )
    return merged


def prepare_curve_group_data(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    mean_only: bool,
) -> list[dict]:
    """
    Extract curve data per model/class group for ROC, PR, and calibration plots.

    Returns a list of dicts with keys ``label``, ``x``, ``y``, ``yerr``
    (std array for mean curves, ``None`` for fold curves), and ``kind``
    (``"mean"`` or ``"fold"``).
    """
    records = []
    group_cols = ["Model"] + (["Class"] if "Class" in frame.columns else [])
    for keys, group in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label_base = str(keys[0])
        if len(keys) > 1:
            label_base = f"{label_base} class {keys[1]}"

        if mean_only and "Fold" in group.columns:
            x_grid = np.unique(
                np.concatenate(
                    [g[x_col].to_numpy(dtype=float) for _, g in group.groupby("Fold")]
                )
            )
            y_values = []
            for _, fold_group in group.groupby("Fold"):
                fold_group = fold_group.sort_values(x_col)
                x = fold_group[x_col].to_numpy(dtype=float)
                y = fold_group[y_col].to_numpy(dtype=float)
                order = np.argsort(x)
                y_values.append(np.interp(x_grid, x[order], y[order]))
            y_stack = np.vstack(y_values)
            mean = np.nanmean(y_stack, axis=0)
            std = np.nanstd(y_stack, axis=0)
            records.append(
                {
                    "label": f"{label_base} (mean)",
                    "x": x_grid,
                    "y": mean,
                    "yerr": std,
                    "kind": "mean",
                }
            )
        else:
            split = (
                group.groupby("Fold") if "Fold" in group.columns else [(None, group)]
            )
            for fold, fold_group in split:
                suffix = f" fold {fold}" if fold is not None else ""
                records.append(
                    {
                        "label": f"{label_base}{suffix}",
                        "x": fold_group[x_col].to_numpy(dtype=float),
                        "y": fold_group[y_col].to_numpy(dtype=float),
                        "yerr": None,
                        "kind": "fold",
                    }
                )
    return records


def prepare_decoding_curve_frame(
    result_or_curve: Any,
    accessor: str,
    required_columns: Sequence[str],
    context: str,
    model: str | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    """Coerce, validate, filter, and require rows for decoding curve plots."""
    frame = coerce_decoding_frame(result_or_curve, accessor=accessor)
    require_columns(frame, required_columns, context=accessor)
    frame = select_rows(frame, model=model, fold=fold)
    return require_non_empty(frame, context)


def prepare_fold_score_data(
    result_or_scores: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return scalar detailed-score rows for fold-dispersion plots."""
    frame = coerce_decoding_frame(result_or_scores, accessor="get_detailed_scores")
    frame = select_rows(frame, model=model, metric=metric)
    return _scalar_decoding_scores(frame, "get_detailed_scores")


def prepare_temporal_score_curve_frame(
    result_or_scores: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return temporal score-summary rows with non-null time values."""
    summary = coerce_decoding_frame(
        result_or_scores, accessor="get_temporal_score_summary"
    )
    require_columns(
        summary,
        ["Model", "Metric", "Time", "Mean"],
        context="get_temporal_score_summary",
    )
    summary = select_rows(summary, model=model, metric=metric)
    return require_non_empty(
        summary[summary["Time"].notna()].copy(), "temporal score curve"
    )


def prepare_temporal_generalization_matrix(
    result_or_scores: Any,
    model: str | None = None,
    metric: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one train-time by test-time matrix and its first source row."""
    summary = coerce_decoding_frame(
        result_or_scores, accessor="get_temporal_score_summary"
    )
    require_columns(
        summary,
        ["Model", "Metric", "TrainTime", "TestTime", "Mean"],
        context="get_temporal_score_summary",
    )
    summary = select_rows(summary, model=model, metric=metric)
    matrix_data = summary[
        summary["TrainTime"].notna() & summary["TestTime"].notna()
    ].copy()
    require_non_empty(matrix_data, "temporal generalization matrix")
    matrix_data = _single_model_metric(matrix_data, "temporal generalization matrix")
    first = matrix_data.iloc[0]
    train_order = pd.unique(matrix_data["TrainTime"])
    test_order = pd.unique(matrix_data["TestTime"])
    matrix = matrix_data.pivot(index="TrainTime", columns="TestTime", values="Mean")
    matrix = matrix.reindex(index=train_order, columns=test_order)
    return matrix, first


def prepare_temporal_statistical_frame(
    result_or_assessment: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return one temporal statistical-assessment frame."""
    frame = coerce_decoding_frame(
        result_or_assessment, accessor="get_statistical_assessment"
    )
    require_columns(
        frame,
        ["Model", "Metric", "Observed", "Time"],
        context="get_statistical_assessment",
    )
    frame = select_rows(frame, model=model, metric=metric)
    frame = frame[frame["Time"].notna()].copy()
    require_non_empty(frame, "temporal statistical assessment")
    return _single_model_metric(frame, "temporal statistical assessment")


def prepare_null_interval_frame(
    result_or_assessment: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return scalar statistical-assessment rows with observed values."""
    frame = coerce_decoding_frame(
        result_or_assessment, accessor="get_statistical_assessment"
    )
    require_columns(
        frame, ["Model", "Metric", "Observed"], context="get_statistical_assessment"
    )
    frame = select_rows(frame, model=model, metric=metric)
    return require_non_empty(
        frame[frame["Observed"].notna()].copy(), "null interval summary"
    )


def prepare_training_history_artifacts(
    result_or_artifacts: Any,
    model: str | None = None,
) -> pd.DataFrame:
    """Return model-artifact rows that contain training-history payloads."""
    artifacts = coerce_decoding_frame(
        result_or_artifacts, accessor="get_model_artifacts"
    )
    require_columns(
        artifacts,
        ["Model", "Key", "ArtifactType", "Value"],
        context="get_model_artifacts",
    )
    artifacts = select_rows(artifacts, model=model)
    rows = artifacts[
        (artifacts["Key"].isin(["training", "validation", "history"]))
        | (artifacts["ArtifactType"] == "history")
    ]
    return require_non_empty(rows, "training history")


def prepare_decoding_score_data(
    result: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return scalar detailed-score rows for score summary plots."""
    frame = coerce_decoding_frame(result, accessor="get_detailed_scores")
    frame = select_rows(frame, model=model, metric=metric)
    return _scalar_decoding_scores(frame, "get_detailed_scores")


def prepare_model_comparison_frame(
    result: Any,
    metric: str = "accuracy",
    reference: str | None = None,
    paired: bool = True,
) -> pd.DataFrame:
    """Return validated model-comparison differences."""
    allow_score_fallback = False
    if isinstance(result, pd.DataFrame):
        comp = result.copy()
    elif paired and hasattr(result, "compare_models_paired") and reference is not None:
        model_names = list(getattr(result, "raw", {}))
        if not model_names:
            raise ValueError("paired model comparison requires result.raw model names.")
        if reference not in model_names:
            raise ValueError(f"reference model not found: {reference}")
        rows = []
        for model_name in model_names:
            if model_name == reference:
                continue
            try:
                comparison = result.compare_models_paired(
                    reference, model_name, metric=metric
                )
            except Exception as exc:
                raise ValueError(
                    "Failed paired model comparison for "
                    f"{reference!r} vs {model_name!r} using metric {metric!r}."
                ) from exc
            rows.extend(comparison.to_dict("records"))
        comp = pd.DataFrame(rows)
    elif hasattr(result, "compare_models"):
        try:
            comp = result.compare_models(metric=metric)
        except Exception as exc:
            raise ValueError(
                f"Failed model comparison using metric {metric!r}."
            ) from exc
    elif hasattr(result, "get_detailed_scores"):
        comp = pd.DataFrame()
        allow_score_fallback = True
    else:
        raise TypeError("result must be an ExperimentResult or comparison DataFrame.")

    if comp.empty and not allow_score_fallback:
        raise ValueError(f"No model comparison rows available for metric {metric!r}.")
    if comp.empty:
        scores = prepare_decoding_score_data(result, metric=metric)
        means = scores.groupby("Model")["Value"].mean().sort_values(ascending=False)
        if reference is None:
            reference = str(means.index[0])
        if reference not in means.index:
            raise ValueError(f"reference model not found: {reference}")
        comp = pd.DataFrame(
            {
                "ModelA": reference,
                "ModelB": [model for model in means.index if model != reference],
                "Difference": [
                    float(means[reference] - means[model])
                    for model in means.index
                    if model != reference
                ],
            }
        )
    require_columns(comp, ["Difference"], context="model comparison")
    return require_non_empty(comp, "model comparison")


def prepare_fit_diagnostics_frame(
    result: Any,
    by: str = "Model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fit diagnostics and one timing row per model/fold."""
    frame = coerce_decoding_frame(result, accessor="get_fit_diagnostics")
    require_columns(frame, [by, "TotalTime"], context="get_fit_diagnostics")
    data = frame.drop_duplicates([col for col in ["Model", "Fold"] if col in frame])
    return frame, require_non_empty(data, "fit diagnostics")


def prepare_probability_diagnostics_summary(
    result: Any,
    model: str | None = None,
    metric: str | None = None,
) -> pd.DataFrame:
    """Return mean probability-diagnostic values by model and metric."""
    frame = coerce_decoding_frame(result, accessor="get_probability_diagnostics")
    require_columns(
        frame, ["Model", "Metric", "Value"], context="get_probability_diagnostics"
    )
    frame = select_rows(frame, model=model, metric=metric)
    require_non_empty(frame, "probability diagnostics")
    return (
        frame.groupby(["Model", "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index()
        .sort_values(["Model", "Metric"])
    )


def prepare_prediction_accuracy_scores(
    result: Any,
    group_columns: Sequence[str],
    model: str | None = None,
    context: str = "prediction diagnostics",
) -> pd.DataFrame:
    """Return accuracy scores grouped by model plus caller-selected columns."""
    if not group_columns:
        raise ValueError("group_columns must contain at least one column.")
    frame = coerce_decoding_frame(result, accessor="get_predictions")
    columns = ["Model", *group_columns, "y_true", "y_pred"]
    require_columns(frame, columns, context="get_predictions")
    frame = select_rows(frame, model=model)
    require_non_empty(frame, context)
    return (
        frame.groupby(["Model", *group_columns], dropna=False)
        .apply(_prediction_accuracy)
        .reset_index(name="Value")
    )


def prepare_regression_prediction_data(
    result: Any,
    model: str | None = None,
    fold: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return aligned numeric observed and predicted regression values."""
    frame = coerce_decoding_frame(result, accessor="get_predictions")
    require_columns(frame, ["y_true", "y_pred"], context="get_predictions")
    frame = select_rows(frame, model=model, fold=fold)
    require_non_empty(frame, "regression diagnostics")
    y_true = pd.to_numeric(frame["y_true"], errors="coerce")
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        raise ValueError("No numeric regression predictions available.")
    return y_true[valid], y_pred[valid]


def prepare_search_results_frame(
    result: Any,
    model: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return filtered and rank-sorted hyperparameter-search rows."""
    frame = coerce_decoding_frame(result, accessor="get_search_results")
    require_columns(
        frame, ["Model", "Rank", "MeanTestScore"], context="get_search_results"
    )
    frame = select_rows(frame, model=model)
    frame = require_non_empty(frame, "search results").sort_values(["Model", "Rank"])
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive when provided.")
        frame = frame.groupby("Model", group_keys=False).head(top_n)
        frame = require_non_empty(frame, "search results")
    return frame


def prepare_feature_stability_series(
    result: Any,
    model: str | None = None,
    top_n: int | None = 25,
) -> pd.Series:
    """Return mean feature-selection frequencies sorted descending."""
    frame = coerce_decoding_frame(result, accessor="get_feature_stability")
    frame = select_rows(frame, model=model)
    require_columns(
        frame,
        ["FeatureName", "SelectionFrequency"],
        context="get_feature_stability",
    )
    require_non_empty(frame, "feature stability")
    series = (
        frame.groupby("FeatureName")["SelectionFrequency"]
        .mean()
        .sort_values(ascending=False)
    )
    return series.head(top_n) if top_n is not None else series


def prepare_feature_score_series(
    result: Any,
    model: str | None = None,
    top_n: int | None = 25,
) -> pd.Series:
    """Return mean univariate feature scores sorted descending."""
    frame = coerce_decoding_frame(result, accessor="get_feature_scores")
    frame = select_rows(frame, model=model)
    require_columns(frame, ["FeatureName", "Score"], context="get_feature_scores")
    require_non_empty(frame, "feature scores")
    scores = frame.groupby("FeatureName")["Score"].mean().sort_values(ascending=False)
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive when provided.")
        scores = scores.head(top_n)
    return scores


def prepare_trajectory_data(
    X: np.ndarray,
    times: Sequence[float] | np.ndarray | None = None,
    labels: Sequence[Any] | np.ndarray | None = None,
    values: np.ndarray | None = None,
    dimensions: int = 2,
    smooth_window: int | None = 1,
    downsample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    """Validate and align trajectory tensor inputs for static/interactive plots."""
    trajectories = np.asarray(X, dtype=float)
    if trajectories.ndim != 3:
        raise ValueError(
            "`X` must be a 3D trajectory tensor with shape "
            "(n_trajectories, n_times, n_dimensions)."
        )
    if dimensions not in {2, 3}:
        raise ValueError("`dimensions` must be 2 or 3.")
    if trajectories.shape[2] < dimensions:
        raise ValueError(
            f"`X` has only {trajectories.shape[2]} dimensions; "
            f"cannot plot {dimensions}."
        )
    if downsample < 1:
        raise ValueError("downsample must be a positive integer.")

    n_trajectories, n_times, _ = trajectories.shape
    time_values = np.arange(n_times) if times is None else np.asarray(times)
    if len(time_values) != n_times:
        raise ValueError("`times` must align with the trajectory time axis.")

    label_values = None
    if labels is not None:
        label_values = np.asarray(labels)
        if label_values.shape[0] != n_trajectories:
            raise ValueError("`labels` must align with the trajectory axis.")

    value_array = None
    if values is not None:
        value_array = np.asarray(values, dtype=float)
        if value_array.shape != (n_trajectories, n_times):
            raise ValueError("`values` must have shape (n_trajectories, n_times).")

    window = 1 if smooth_window is None else int(smooth_window)
    if window < 1:
        raise ValueError("smooth_window must be a positive integer.")
    if window > 1:
        trajectories = np.asarray(
            [
                np.stack(
                    [
                        _moving_average_1d(traj[:, dim], window)
                        for dim in range(traj.shape[1])
                    ],
                    axis=1,
                )
                for traj in trajectories
            ]
        )
        time_values = _moving_average_1d(time_values, window)
        if value_array is not None:
            value_array = np.asarray(
                [_moving_average_1d(row, window) for row in value_array]
            )

    if downsample > 1:
        trajectories = trajectories[:, ::downsample, :]
        time_values = time_values[::downsample]
        if value_array is not None:
            value_array = value_array[:, ::downsample]

    return (
        trajectories[:, :, :dimensions],
        time_values,
        label_values,
        value_array,
        dimensions,
    )


def prepare_loss_history(
    loss_history: Sequence[float] | np.ndarray,
    scope: str | None = None,
) -> np.ndarray:
    """Return a non-empty one-dimensional reducer loss-history array."""
    losses = np.asarray(loss_history, dtype=float).reshape(-1)
    if losses.size == 0:
        raise ValueError(
            "loss_history is empty. This plot is only valid for reducers that "
            "track iterative training loss (neural/stochastic methods). "
            "Linear reducers (PCA, SVD) do not produce a loss history."
        )
    if scope is not None and scope not in ("train", "val"):
        raise ValueError(f"scope must be 'train' or 'val', got {scope!r}")
    return losses


def prepare_eigenvalue_curves(
    values: Mapping[str, Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray,
    max_components: int | None = None,
) -> list[dict[str, Any]]:
    """Return mean/SEM eigenvalue curves for one or more labelled conditions."""
    if max_components is not None and max_components < 1:
        raise ValueError("max_components must be a positive integer or None.")
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("values dict is empty.")
        items = list(values.items())
    else:
        items = [("Individual", values)]

    records: list[dict[str, Any]] = []
    for label, raw in items:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError("eigenvalue arrays must be 1D or 2D.")
        if arr.shape[1] == 0:
            raise ValueError("eigenvalue arrays must contain at least one component.")
        cap = (
            min(max_components, arr.shape[1])
            if max_components is not None
            else arr.shape[1]
        )
        arr = arr[:, :cap]
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else None
        records.append(
            {
                "label": str(label),
                "components": np.arange(1, cap + 1),
                "mean": mean,
                "sem": sem,
                "cumulative": np.cumsum(mean),
            }
        )
    return records


def prepare_shepard_distances(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    sample_size: int = 1000,
    random_state: int | None = None,
    distances: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return finite original/embedded pairwise distances and their correlation."""
    if isinstance(distances, Mapping) and {"original", "embedded"} <= set(distances):
        dist_high = np.asarray(distances["original"], dtype=float)
        dist_low = np.asarray(distances["embedded"], dtype=float)
    else:
        from coco_pipe.dim_reduction.evaluation.metrics import shepard_diagram_data

        dist_high, dist_low = shepard_diagram_data(
            X_orig, X_emb, sample_size=sample_size, random_state=random_state
        )
        dist_high = np.asarray(dist_high, dtype=float)
        dist_low = np.asarray(dist_low, dtype=float)
    if dist_high.shape != dist_low.shape:
        raise ValueError("Shepard distance arrays must have matching shapes.")
    valid = np.isfinite(dist_high) & np.isfinite(dist_low)
    dist_high = dist_high[valid]
    dist_low = dist_low[valid]
    if dist_high.size == 0:
        raise ValueError("No valid pairwise distances to plot in Shepard diagram.")
    corr = np.corrcoef(dist_high, dist_low)[0, 1] if dist_high.size > 1 else np.nan
    return dist_high, dist_low, float(corr)


def prepare_streamline_inputs(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated 2D embedding coordinates and velocity vectors."""
    points = np.asarray(X_emb, dtype=float)
    vectors = np.asarray(V_emb, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Streamlines currently only supported for 2D.")
    if vectors.shape != points.shape:
        raise ValueError("`V_emb` must have the same shape as `X_emb`.")
    return points, vectors


def prepare_streamline_grid(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    grid_density: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate sparse 2D velocities onto a regular grid."""
    if grid_density < 2:
        raise ValueError("grid_density must be at least 2.")
    points, vectors = prepare_streamline_inputs(X_emb, V_emb)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    xi = np.linspace(x_min, x_max, grid_density)
    yi = np.linspace(y_min, y_max, grid_density)
    Xi, Yi = np.meshgrid(xi, yi)
    from scipy.interpolate import griddata

    Ui = griddata(points, vectors[:, 0], (Xi, Yi), method="linear")
    Vi = griddata(points, vectors[:, 1], (Xi, Yi), method="linear")
    missing = ~np.isfinite(Ui) | ~np.isfinite(Vi)
    if missing.any():
        Ui_nearest = griddata(points, vectors[:, 0], (Xi, Yi), method="nearest")
        Vi_nearest = griddata(points, vectors[:, 1], (Xi, Yi), method="nearest")
        Ui = np.where(missing, Ui_nearest, Ui)
        Vi = np.where(missing, Vi_nearest, Vi)
    return Xi, Yi, Ui, Vi


def prepare_trajectory_separation_series(
    separation: Mapping[Any, Sequence[float] | np.ndarray],
    times: Sequence[float] | np.ndarray | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Return ranked trajectory-separation series with display labels and x-values."""
    if not separation:
        raise ValueError("separation is empty.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be a positive integer or None.")

    time_values = None if times is None else np.asarray(times)
    items = []
    for pair, values in separation.items():
        series = np.asarray(values, dtype=float).reshape(-1)
        if series.size == 0:
            raise ValueError("separation series must not be empty.")
        x_values = np.arange(len(series)) if time_values is None else time_values
        if len(x_values) != len(series):
            raise ValueError("times must align with separation arrays.")
        finite = series[np.isfinite(series)]
        rank_value = float(np.nanmax(finite)) if finite.size else float("-inf")
        label = (
            f"{pair[0]} vs {pair[1]}"
            if isinstance(pair, tuple) and len(pair) == 2
            else str(pair)
        )
        items.append({"label": label, "x": x_values, "y": series, "rank": rank_value})
    items.sort(key=lambda item: item["rank"], reverse=True)
    return items[:top_n] if top_n is not None else items


def prepare_component_loadings_frame(
    components: np.ndarray,
    feature_names: Sequence[str] | None = None,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Return a labelled component-loading matrix."""
    if components is None:
        raise ValueError("components array is required.")
    matrix = np.asarray(components, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("components must be a 2D array.")
    if n_components is not None:
        if n_components < 1:
            raise ValueError("n_components must be a positive integer or None.")
        matrix = matrix[:, :n_components]
    names = (
        list(feature_names)
        if feature_names is not None
        else [f"Feature {idx}" for idx in range(matrix.shape[0])]
    )
    if len(names) != matrix.shape[0]:
        raise ValueError("feature_names must match the number of rows in components.")
    return pd.DataFrame(
        matrix,
        index=names,
        columns=[f"Component {i + 1}" for i in range(matrix.shape[1])],
    )
