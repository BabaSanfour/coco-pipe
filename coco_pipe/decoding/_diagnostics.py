"""
Decoding Diagnostics & Tidy Data Helpers
========================================
Functions for expanding and tidying raw decoding results into DataFrames.
"""

from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ._metrics import get_metric_spec


def time_value(index: int, time_axis: Optional[Sequence[Any]]) -> Any:
    """
    Map a raw integer index to a meaningful scientific time value.

    This helper ensures that temporal decoding results (from sliding or
    generalizing estimators) are human-readable by aligning array indices
    with the actual experiment time points (e.g., mapping index 0 to -0.2s).

    Parameters
    ----------
    index : int
        The raw integer index from the results array.
    time_axis : Sequence[Any], optional
        A sequence of time values (e.g., a numpy array of seconds)
        corresponding to the temporal dimension of the data.

    Returns
    -------
    Any
        The scientific time value if the axis is provided and index is in range;
        otherwise, returns the raw index.

    Examples
    --------
    >>> time_value(0, [-0.2, -0.1, 0.0])
    -0.2
    >>> time_value(5, None)
    5
    """
    if time_axis is None or index >= len(time_axis):
        return index
    return time_axis[index]


def score_rows(
    model: str,
    fold_idx: int,
    metric: str,
    score: Any,
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """
    Expand scalar or temporal fold scores into tidy data rows.

    This function unrolls raw result arrays into a flat list of dictionaries,
    automatically handling three distinct scientific patterns:
    1. Scalar: Standard decoding (1 row per fold/metric).
    2. 1D Array: Sliding Estimator (N rows per fold, mapping 'Time').
    3. 2D Array: Generalizing Estimator (N*M rows, mapping 'TrainTime' and 'TestTime').

    Parameters
    ----------
    model : str
        Name of the estimator.
    fold_idx : int
        The cross-validation fold index.
    metric : str
        The name of the scoring metric.
    score : Any
        The raw score (float, 1D array, or 2D array).
    time_axis : Sequence[Any], optional
        The scientific time points for temporal mapping. Default is None.

    Returns
    -------
    list[Dict[str, Any]]
        A list of "tidy" rows ready for DataFrame conversion.

    Examples
    --------
    >>> score_rows("svc", 0, "accuracy", 0.8)
    [{'Model': 'svc', 'Fold': 0, 'Metric': 'accuracy', 'Value': 0.8}]
    """
    score = np.asarray(score)
    rows = []

    if score.ndim == 0:
        return [
            {
                "Model": model,
                "Fold": fold_idx,
                "Metric": metric,
                "Value": float(score),
            }
        ]

    if score.ndim == 1:
        for t_idx, val in enumerate(score):
            rows.append(
                {
                    "Model": model,
                    "Fold": fold_idx,
                    "Metric": metric,
                    "Time": time_value(t_idx, time_axis),
                    "Value": val,
                }
            )
        return rows

    if score.ndim == 2:
        for t_tr in range(score.shape[0]):
            for t_te in range(score.shape[1]):
                rows.append(
                    {
                        "Model": model,
                        "Fold": fold_idx,
                        "Metric": metric,
                        "TrainTime": time_value(t_tr, time_axis),
                        "TestTime": time_value(t_te, time_axis),
                        "Value": score[t_tr, t_te],
                    }
                )
        return rows

    return [{"Model": model, "Fold": fold_idx, "Metric": metric, "Value": score}]


def prediction_rows(
    model: str,
    fold_idx: int,
    preds: Dict[str, Any],
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """
    Expand raw predictions from a results dictionary into tidy data rows.

    This function is the primary engine for converting raw estimator outputs
    into analyzable DataFrames. It automatically handles the expansion of
    standard, sliding (temporal), and generalizing (train-time x test-time)
    predictions into a flat, tabular format while aligning metadata.

    Parameters
    ----------
    model : str
        Name of the estimator.
    fold_idx : int
        The cross-validation fold index.
    preds : Dict[str, Any]
        Raw predictions dictionary containing 'y_true', 'y_pred',
        and optionally 'y_proba', 'sample_id', and 'sample_metadata'.
    time_axis : Sequence[Any], optional
        Scientific time points for coordinate mapping. Default is None.

    Returns
    -------
    list[Dict[str, Any]]
        A list of "tidy" records (list of dictionaries).

    Examples
    --------
    >>> preds = {"y_true": [0], "y_pred": [0], "sample_id": ["s1"]}
    >>> # result = prediction_rows("svc", 0, preds)
    """
    y_true = np.asarray(preds["y_true"])
    y_pred = np.asarray(preds["y_pred"])
    y_proba = np.asarray(preds["y_proba"]) if "y_proba" in preds else None
    y_score = np.asarray(preds["y_score"]) if "y_score" in preds else None

    n_samples = len(y_true)
    sample_index = np.asarray(preds.get("sample_index", np.arange(n_samples)))
    sample_id = np.asarray(preds.get("sample_id", sample_index))
    groups = optional_values(preds.get("group"), n_samples)
    metadata = preds.get("sample_metadata") or {}

    # 1. Determine Expansion Factor (n_exp) and Temporal Coordinates
    pattern = "standard"
    n_exp = 1
    coords = {}

    if y_pred.ndim == 2:  # Sliding
        pattern = "sliding"
        n_times = y_pred.shape[1]
        n_exp = n_times
        coords["Time"] = np.tile(
            [time_value(t, time_axis) for t in range(n_times)], n_samples
        )
    elif y_pred.ndim == 3:  # Generalizing
        pattern = "generalizing"
        n_tr, n_te = y_pred.shape[1], y_pred.shape[2]
        n_exp = n_tr * n_te
        tr_vals = [time_value(t, time_axis) for t in range(n_tr)]
        te_vals = [time_value(t, time_axis) for t in range(n_te)]
        coords["TrainTime"] = np.tile(np.repeat(tr_vals, n_te), n_samples)
        coords["TestTime"] = np.tile(np.tile(te_vals, n_tr), n_samples)

    # 2. Vectorized Backbone Construction
    data = {
        "Model": [model] * (n_samples * n_exp),
        "Fold": [fold_idx] * (n_samples * n_exp),
        "SampleIndex": np.repeat(sample_index, n_exp),
        "SampleID": np.repeat(sample_id, n_exp),
        "Group": np.repeat(groups, n_exp),
        "y_true": np.repeat([row_value(y_true, i) for i in range(n_samples)], n_exp),
        "y_pred": y_pred.ravel(),
    }

    # Add temporal coordinates
    data.update(coords)

    # Inject metadata (broadcasted)
    for key, values in metadata.items():
        v_arr = np.asarray(values, dtype=object)
        data[key] = np.repeat(v_arr[:n_samples], n_exp)

    df = pd.DataFrame(data)

    # 3. Add Probabilities (Standard / Sliding / Generalizing)
    if y_proba is not None:
        if pattern == "standard":
            if y_proba.ndim == 1:
                df["y_proba_0"] = 1.0 - y_proba
                df["y_proba_1"] = y_proba
            elif y_proba.ndim == 2:
                for c in range(y_proba.shape[1]):
                    df[f"y_proba_{c}"] = y_proba[:, c]
        elif pattern == "sliding":
            # 2D: (samples, times) -> binary, return proba for class 1
            if y_proba.ndim == 2:
                df["y_proba_0"] = 1.0 - y_proba.ravel()
                df["y_proba_1"] = y_proba.ravel()
            # 3D: (samples, times, classes) -> multiclass
            elif y_proba.ndim == 3:
                # MNE SlidingEstimator returns (samples, times, classes)
                for c in range(y_proba.shape[2]):
                    df[f"y_proba_{c}"] = y_proba[:, :, c].ravel()
        elif pattern == "generalizing":
            # 3D: (samples, tr, te) -> binary
            if y_proba.ndim == 3:
                df["y_proba_0"] = 1.0 - y_proba.ravel()
                df["y_proba_1"] = y_proba.ravel()
            # 4D: (samples, tr, te, classes) -> multiclass
            elif y_proba.ndim == 4:
                # MNE GeneralizingEstimator returns (samples, tr, te, classes)
                for c in range(y_proba.shape[3]):
                    df[f"y_proba_{c}"] = y_proba[:, :, :, c].ravel()

    # 4. Add Decision Scores
    if y_score is not None:
        if pattern == "standard":
            if y_score.ndim == 1:
                df["y_score"] = y_score
            elif y_score.ndim == 2:
                for c in range(y_score.shape[1]):
                    df[f"y_score_{c}"] = y_score[:, c]
        else:  # Temporal scores are less common but supported
            df["y_score"] = y_score.ravel()

    return df.to_dict(orient="records")


def row_value(values: np.ndarray, row_idx: int) -> Any:
    """
    Extract a value from an array while ensuring JSON serialization safety.

    This helper extracts a single row/item and converts any nested NumPy
    arrays into standard Python lists. This is critical for ensuring that
    the final tidy records can be serialized to JSON without errors.

    Parameters
    ----------
    values : np.ndarray
        The source array (e.g., y_true or metadata).
    row_idx : int
        The index to extract.

    Returns
    -------
    Any
        The extracted value, converted to a list if it was an array.
    """
    val = values[row_idx]
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def optional_values(values: Optional[Any], length: int) -> np.ndarray:
    """
    Ensure a sequence exists and has the correct length for broadcasting.

    This helper is used during tidy data expansion to handle optional
    fields (like 'group' or custom metadata). If the source is None,
    it generates a 'ghost' array of Nones, ensuring that downstream
    vectorized operations (like np.repeat) do not crash.

    Parameters
    ----------
    values : Any, optional
        The source sequence or None.
    length : int
        The required length of the resulting array.

    Returns
    -------
    np.ndarray
        An array of the specified length.
    """
    if values is None:
        return np.full(length, None, dtype=object)
    return np.asarray(values)


def proba_matrix(group: pd.DataFrame, n_classes: int) -> Optional[np.ndarray]:
    """
    Re-assemble a probability matrix from tidy prediction columns.

    This is an "inverse tidy" operation used by statistical assessment
    routines. It finds columns like 'y_proba_0', 'y_proba_1', etc.,
    and packs them back into a single 2D NumPy array.

    Parameters
    ----------
    group : pd.DataFrame
        A DataFrame containing prediction rows.
    n_classes : int
        The expected number of classes (determines how many columns to look for).

    Returns
    -------
    np.ndarray, optional
        A (n_samples, n_classes) matrix if all columns are present
        and valid; otherwise None.

    Examples
    --------
    >>> df = pd.DataFrame({"y_proba_0": [0.8, 0.2], "y_proba_1": [0.2, 0.8]})
    >>> proba_matrix(df, 2)
    array([[0.8, 0.2],
           [0.2, 0.8]])
    """
    cols = [f"y_proba_{idx}" for idx in range(n_classes)]
    if not all(c in group for c in cols):
        return None
    pb = group[cols]
    if pb.isna().any().any():
        return None
    return pb.to_numpy(dtype=float)


def unit_indices(group: pd.DataFrame, unit: str) -> list[np.ndarray]:
    """
    Identify row-index blocks for unit-based resampling.

    In scientific assessment, samples are often non-independent (e.g.,
    multiple trials from the same subject). This helper identifies the
    blocks of indices belonging to each independent unit, enabling
    'Block Bootstrapping' or unit-level permutations.

    Parameters
    ----------
    group : pd.DataFrame
        The tidy prediction DataFrame.
    unit : str
        The level of independence (e.g., 'sample', 'subject', 'session').

    Returns
    -------
    list[np.ndarray]
        A list of index arrays, one per independent unit.

    Raises
    ------
    ValueError
        If the unit is unknown or if required columns are missing/empty.
    """
    return _get_unit_blocks(group, unit)


def paired_unit_indices(merged: pd.DataFrame, unit: str) -> list[np.ndarray]:
    """
    Identify row-index blocks for paired unit-based resampling.

    Used specifically for model comparisons. Ensures that when
    performing paired permutation tests, the indices for both models
    are retrieved from a merged DataFrame while maintaining pairing
    consistency (e.g., Subject 1 in Model A paired with Subject 1 in Model B).

    Parameters
    ----------
    merged : pd.DataFrame
        The merged tidy DataFrame containing results for two models.
    unit : str
        The level of independence.

    Returns
    -------
    list[np.ndarray]
        A list of index arrays for the units in the merged Frame.

    Raises
    ------
    ValueError
        If the unit is unknown or if required columns are missing/empty.
    """
    return _get_unit_blocks(merged, unit, suffix="_A")


def _get_unit_blocks(df: pd.DataFrame, unit: str, suffix: str = "") -> list[np.ndarray]:
    """Identify blocks of row indices belonging to the same unit."""
    values = _resolve_unit_values(df, unit, suffix)
    return [np.flatnonzero(values == v) for v in pd.unique(values)]


def _resolve_unit_values(df: pd.DataFrame, unit: str, suffix: str = "") -> np.ndarray:
    """Extract the column identifying individual units with fallback logic."""
    valid_units = {"sample", "epoch", "group", "subject", "session", "site"}
    if unit not in valid_units:
        raise ValueError(f"unit must be one of {valid_units}, got '{unit}'.")

    if unit in {"sample", "epoch"}:
        col = "SampleID"
    elif unit == "group":
        col = f"Group{suffix}"
    else:
        col = f"{unit.capitalize()}{suffix}"

    if col not in df or df[col].isna().all():
        raise ValueError(f"unit='{unit}' requires a non-empty '{col}' column.")

    return df[col].to_numpy()


def score_frame(frame: pd.DataFrame, metric: str) -> float:
    """
    Score a tidy prediction frame using the specified metric.

    This dispatcher automatically routes the correct columns from the tidy
    DataFrame to the underlying Scikit-Learn scorer based on the metric's
    required response method (labels, probabilities, or decision scores).

    Parameters
    ----------
    frame : pd.DataFrame
        The tidy prediction DataFrame containing 'y_true', 'y_pred',
        and optional 'y_proba_X' or 'y_score' columns.
    metric : str
        The name of the metric to compute (e.g., 'roc_auc', 'accuracy').

    Returns
    -------
    float
        The calculated scientific score.

    Raises
    ------
    ValueError
        If the required columns for the metric are missing (e.g., scoring
        ROC-AUC without probabilities) or if binary-only metrics are
        applied to multiclass data.
    """
    metric_spec = get_metric_spec(metric)
    y_true = frame["y_true"].to_numpy()

    # 1. Label-based metrics
    if metric_spec.response_method == "predict":
        return float(metric_spec.scorer(y_true, frame["y_pred"].to_numpy()))

    # 2. Probability-based metrics
    proba_cols = sorted(
        [col for col in frame.columns if col.startswith("y_proba_")],
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    if metric_spec.response_method in {"proba", "proba_or_score"} and proba_cols:
        proba = frame[proba_cols].to_numpy(dtype=float)
        labels = sorted(pd.unique(y_true).tolist())
        if metric == "brier_score":
            if proba.shape[1] != 2:
                raise ValueError("brier_score supports binary classification only.")
            return float(metric_spec.scorer(y_true, proba[:, 1]))

        if metric == "roc_auc" and proba.shape[1] == 2:
            return float(metric_spec.scorer(y_true, proba[:, 1]))

        if metric in {"average_precision", "pr_auc"}:
            if proba.shape[1] == 2:
                return float(metric_spec.scorer(y_true, proba[:, 1]))
            # AP multiclass: requires binarized y_true
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_true, classes=labels)
            return float(metric_spec.scorer(y_bin, proba))

        if metric == "log_loss":
            return float(metric_spec.scorer(y_true, proba, labels=labels))

        # Fallback for custom multiclass metrics that support multi_class='ovr'.
        # Standard metrics like average_precision are handled explicitly above
        # as they require custom binarization logic.
        return float(metric_spec.scorer(y_true, proba, multi_class="ovr"))

    # 3. Decision-function based metrics
    if (
        metric_spec.response_method in {"score", "proba_or_score"}
        and "y_score" in frame
    ):
        return float(metric_spec.scorer(y_true, frame["y_score"].to_numpy(dtype=float)))

    raise ValueError(f"Metric '{metric}' cannot be scored from available predictions.")


def scalar_prediction_frame(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a prediction DataFrame to include only scalar results.

    Excludes rows with temporal coordinates (Time, TrainTime, TestTime),
    which is a common requirement for standard non-temporal diagnostics.

    Parameters
    ----------
    preds : pd.DataFrame
        The input prediction DataFrame.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only scalar (standard) results.
    """
    if preds.empty:
        return preds

    mask = pd.Series(True, index=preds.index)
    for col in ["Time", "TrainTime", "TestTime"]:
        if col in preds:
            mask &= preds[col].isna()

    return preds[mask]


def confusion_matrix_frame(
    preds: pd.DataFrame,
    labels: Sequence[Any],
    normalize: Optional[str] = None,
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute and tidy confusion matrices from a prediction frame.

    Parameters
    ----------
    preds : pd.DataFrame
        Prediction frame containing 'y_true' and 'y_pred'.
    labels : Sequence[Any]
        The set of class labels to include in the matrix.
    normalize : {'true', 'pred', 'all'}, optional
        Normalization strategy for the matrix.
    group_cols : list[str], optional
        Columns to group by (e.g., ['Model', 'Fold']).

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame with grouping columns, TrueLabel, PredictedLabel, and Value.
    """
    from sklearn.metrics import confusion_matrix

    group_cols = group_cols or ["Model", "Fold"]
    frames = []

    for names, group in preds.groupby(group_cols):
        matrix = confusion_matrix(
            group["y_true"], group["y_pred"], labels=labels, normalize=normalize
        )
        df_m = pd.DataFrame(matrix, index=labels, columns=labels)
        df_m.index.name = "TrueLabel"
        df_m.columns.name = "PredictedLabel"
        df_m = df_m.stack().reset_index(name="Value")

        # Map group column names to their values
        if isinstance(names, (list, tuple)):
            for col, val in zip(group_cols, names):
                df_m[col] = val
        else:
            df_m[group_cols[0]] = names
        frames.append(df_m)

    if not frames:
        return pd.DataFrame(
            columns=group_cols + ["TrueLabel", "PredictedLabel", "Value"]
        )

    return pd.concat(frames, ignore_index=True)[
        group_cols + ["TrueLabel", "PredictedLabel", "Value"]
    ]


def curve_score_groups(
    preds: pd.DataFrame,
    model: Optional[str] = None,
    require_probability: bool = False,
    pos_label: Optional[Any] = None,
) -> Iterator[tuple[str, int, Any, np.ndarray, np.ndarray]]:
    """
    Yield binary or one-vs-rest score groups for curve plotting.

    This helper handles the complexity of resolving positive labels,
    identifying probability columns, and falling back to decision scores
    across multiple models and folds.

    Yields
    ------
    model_name : str
    fold_idx : int
    class_label : Any
    y_binary : np.ndarray
    y_score : np.ndarray
    """
    if model is not None:
        preds = preds[preds["Model"] == model]

    if preds.empty:
        return

    for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
        y_true = group["y_true"].to_numpy()
        unique_labels = sorted(pd.unique(y_true).tolist())
        if len(unique_labels) < 2:
            continue

        # Binary Case
        if len(unique_labels) == 2:
            if pos_label is not None:
                if pos_label not in unique_labels:
                    continue
                target_label = pos_label
            else:
                target_label = unique_labels[1]

            l_idx = unique_labels.index(target_label)
            p_col = f"y_proba_{l_idx}"

            if p_col in group and group[p_col].notna().all():
                y_score = group[p_col].to_numpy(dtype=float)
            elif (
                not require_probability
                and "y_score" in group
                and group["y_score"].notna().all()
            ):
                y_score = group["y_score"].to_numpy(dtype=float)
                if l_idx == 0:
                    y_score = -y_score
            else:
                continue
            yield m_name, f_idx, target_label, (y_true == target_label), y_score
            continue

        # Multiclass Case (One-vs-Rest)
        for c_idx, label in enumerate(unique_labels):
            p_col, s_col = f"y_proba_{c_idx}", f"y_score_{c_idx}"
            if p_col in group and group[p_col].notna().all():
                y_score = group[p_col].to_numpy(dtype=float)
            elif (
                not require_probability
                and s_col in group
                and group[s_col].notna().all()
            ):
                y_score = group[s_col].to_numpy(dtype=float)
            else:
                continue
            yield m_name, f_idx, label, (y_true == label), y_score
