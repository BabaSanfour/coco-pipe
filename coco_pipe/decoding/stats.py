"""
Finite-sample statistical assessment for decoding results.

This module separates descriptive performance from inferential claims. The
default inferential path reruns the complete decoding experiment under label
permutations so learned preprocessing, feature selection, tuning, and
calibration remain inside each null pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import via the package re-export rather than the ``result`` submodule so
    # the type-only reference does not register as an import cycle edge.
    from coco_pipe.decoding import ExperimentResult

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, false_discovery_control, norm

from ._metrics import get_metric_spec
from .configs import StatisticalAssessmentConfig

logger = logging.getLogger(__name__)


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjusted p-values and rejections; NaN stays NaN."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        adjusted[valid] = false_discovery_control(values[valid], method="bh")
    return adjusted, adjusted <= alpha


def correct_sweep_pvalues(
    results: pd.DataFrame,
    *,
    p_column: str = "p_value",
    family_columns: Sequence[str] = (
        "target",
        "input_mode",
        "analysis_mode",
        "model",
        "selection_mode",
    ),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply BH-FDR independently within configured decoding sweep families."""
    if p_column not in results.columns:
        raise KeyError(f"Missing p-value column: {p_column}")
    out = results.copy()
    out["p_value_fdr"] = np.nan
    out["significant_fdr"] = False
    present = [column for column in family_columns if column in out.columns]
    groups = (
        [out.index]
        if not present
        else [frame.index for _, frame in out.groupby(present, dropna=False)]
    )
    for indices in groups:
        index = list(indices)
        adjusted, rejected = benjamini_hochberg(
            out.loc[index, p_column].to_numpy(dtype=float),
            alpha=alpha,
        )
        out.loc[index, "p_value_fdr"] = adjusted
        out.loc[index, "significant_fdr"] = rejected
    return out


TEMPORAL_COLUMNS = ["Time", "TrainTime", "TestTime"]


def _coordinate_groups(
    frame: pd.DataFrame, columns: list[str]
) -> dict[tuple, np.ndarray]:
    """Split `frame` by coordinate columns once, as tuple-keyed row positions.

    Two details are handled here rather than at each call site: `groupby` on a
    one-element list yields scalar keys today and tuple keys in a future pandas,
    and the returned positions must be positional (`.iloc`) rather than index
    labels, because a bootstrap resample repeats index labels.
    """
    if not columns:
        return {(): np.arange(len(frame))}
    by = columns[0] if len(columns) == 1 else columns
    return {
        key if isinstance(key, tuple) else (key,): positions
        for key, positions in frame.groupby(by, sort=False).indices.items()
    }


def aggregate_predictions_for_inference(
    predictions: pd.DataFrame,
    metric: str,
    task: str = "classification",
    unit_of_inference: str = "sample",
    custom_unit_column: str | None = None,
    custom_aggregation: str = "mean",
    require_single_prediction: bool = False,
) -> pd.DataFrame:
    """
    Aggregate prediction rows to independent units for inference.

    This ensures that each independent unit (e.g., a subject or a specific trial)
    contributes exactly one prediction per temporal coordinate to the statistical
    test, preventing pseudoreplication.

    Scientific Rationale
    --------------------
    Inferential statistics assume independence between observations. In EEG/MEG,
    multiple epochs from the same subject are correlated. By aggregating
    predictions to the 'subject' level before calculating p-values, we ensure
    the degrees of freedom in the test reflect the number of independent
    biological units rather than the number of recorded segments.

    Parameters
    ----------
    predictions : pd.DataFrame
        Raw predictions from the experiment.
    metric : str
        The metric to optimize aggregation for (e.g., 'accuracy').
    task : str, default='classification'
        Task type ('classification' or 'regression').
    unit_of_inference : str, default='sample'
        The level at which independence is assumed ('sample', 'subject', or 'custom').
    custom_unit_column : str, optional
        Column name in metadata to use as the independence unit if
        unit_of_inference is 'custom'.
    custom_aggregation : str, default='mean'
        Aggregation mode ('mean' or 'majority').
    require_single_prediction : bool, default=False
        If True, ensures that each unit has exactly one prediction per coordinate.

    Returns
    -------
    aggregated_df : pd.DataFrame
        Aggregated predictions with an 'InferentialUnitID' column.

    Raises
    ------
    ValueError
        If the unit column is missing or aggregation is incompatible with the task.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.decoding.stats import aggregate_predictions_for_inference
    >>> df = pd.DataFrame(
    ...     {
    ...         "Subject": ["S1", "S1"],
    ...         "y_true": [1, 1],
    ...         "y_pred": [1, 0],
    ...         "SampleID": [0, 1],
    ...     }
    ... )
    >>> res = aggregate_predictions_for_inference(
    ...     df, "accuracy", unit_of_inference="Subject"
    ... )

    See Also
    --------
    ExperimentResult.get_predictions : Tidy prediction accessor.
    """
    if predictions.empty:
        return predictions.copy()

    frame = predictions
    temporal_cols = [
        col for col in TEMPORAL_COLUMNS if col in frame and frame[col].notna().any()
    ]
    # 1. Resolve Unit Column (Explicitly)
    if unit_of_inference == "sample":
        unit_col, aggregation = "SampleID", "identity"
    else:
        unit_col = (
            custom_unit_column if unit_of_inference == "custom" else unit_of_inference
        )
        if unit_col not in frame.columns:
            normalized = {str(column).casefold(): column for column in frame.columns}
            resolved = normalized.get(str(unit_col).casefold())
            if resolved is None:
                raise ValueError(
                    f"Inference unit '{unit_col}' not found in result columns. "
                    f"Available: {list(frame.columns)}"
                )
            unit_col = resolved
        aggregation = custom_aggregation

    if unit_of_inference == "sample":
        # Fast path: No aggregation needed
        return frame.rename(columns={unit_col: "InferentialUnitID"})

    # 2. Perform Aggregation
    if task != "classification" and aggregation == "majority":
        raise ValueError("majority aggregation is only valid for classification.")

    group_cols = [unit_col, *temporal_cols]
    proba_cols = sorted(
        [col for col in frame.columns if col.startswith("y_proba_")],
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )

    agg_dict = {"y_true": "first"}
    if task == "classification":
        if aggregation == "mean":
            if not proba_cols:
                raise ValueError("mean aggregation requires probability columns.")
            for col in proba_cols:
                agg_dict[col] = "mean"
        elif aggregation == "majority":
            # Avoid slow lambda/mode: count occurrences and pick first mode
            agg_dict["y_pred"] = lambda x: x.value_counts().index[0]
            if proba_cols:
                for col in proba_cols:
                    agg_dict[col] = "mean"
    else:  # regression
        agg_dict["y_pred"] = "mean"

    # Execute Aggregation
    res = frame.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    res = res.rename(columns={unit_col: "InferentialUnitID"})

    # 3. Resolve y_pred for classification mean-aggregation
    if task == "classification" and aggregation == "mean":
        labels = sorted(pd.unique(frame["y_true"]).tolist())
        probs = res[proba_cols].to_numpy()
        # Fast vectorized label assignment
        res["y_pred"] = np.array(labels)[np.argmax(probs, axis=1)]

    return res


def binomial_accuracy_test(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    p0: float | None,
    alpha: float = 0.05,
    ci_method: str = "wilson",
) -> dict[str, Any]:
    """
    Exact upper-tail binomial test for top-1 classification accuracy.

    This test computes the probability of obtaining at least the observed number
    of correct predictions under the null hypothesis (theoretical chance).

    Scientific Rationale
    --------------------
    For classification tasks with a known number of categories, the number of
    correct predictions follows a Binomial distribution B(n, p0) under the null
    hypothesis. This exact test is more robust than z-tests for small sample
    sizes and provides a rigorous bound for 'chance-level' performance.

    Parameters
    ----------
    y_true : Sequence[Any]
        Actual ground-truth labels.
    y_pred : Sequence[Any]
        Predicted labels.
    p0 : float
        The theoretical chance level (e.g., 0.5 for binary classification).
    alpha : float, default=0.05
        Significance level for p-values and confidence intervals.
    ci_method : str, default='wilson'
        Method for calculating confidence intervals ('wilson' or 'clopper_pearson').

    Returns
    -------
    result : dict
        Dictionary containing 'observed' accuracy, 'p_value', 'n_eff',
        'chance_threshold', and confidence intervals.

    Raises
    ------
    ValueError
        If p0 is missing or input arrays are empty.

    Examples
    --------
    >>> from coco_pipe.decoding.stats import binomial_accuracy_test
    >>> res = binomial_accuracy_test([1, 0, 1], [1, 1, 1], p0=0.5)
    >>> print(res["p_value"])

    See Also
    --------
    run_statistical_assessment : Full-pipeline assessment driver.
    """
    if p0 is None:
        raise ValueError("Analytical binomial testing requires an explicit p0.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_eff = len(y_true)
    if n_eff == 0:
        raise ValueError("Cannot run a binomial test with zero predictions.")

    correct = y_true == y_pred
    k_correct = int(np.sum(correct))
    observed = k_correct / n_eff

    # Upper-tail (Is accuracy > p0?)
    p_upper = float(binom.sf(k_correct - 1, n_eff, p0))

    # Chance Threshold (Smallest k such that P(X >= k) <= alpha)
    k_alpha = int(binom.isf(alpha, n_eff, p0)) + 1
    if binom.sf(k_alpha - 2, n_eff, p0) <= alpha:
        k_alpha -= 1

    ci_lower, ci_upper = _accuracy_ci(k_correct, n_eff, alpha, ci_method)
    return {
        "observed": observed,
        "k_correct": k_correct,
        "n_eff": n_eff,
        "p_value": p_upper,
        "chance_threshold": k_alpha / n_eff,
        "chance_threshold_count": k_alpha,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def run_statistical_assessment(
    observed_result: Any,
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    sample_ids: np.ndarray,
    sample_metadata: pd.DataFrame | None,
    feature_names: Sequence[str] | None,
    time_axis: np.ndarray | None,
    observation_level: str,
    inferential_unit: str,
) -> dict[str, Any]:
    """
    Orchestrate the statistical assessment of experiment results.

    Resolves the chosen statistical method (binomial, permutation or sign
    flip) and dispatches analysis for each model and metric.

    Scientific Rationale
    --------------------
    Statistical significance in decoding is often non-trivial due to temporal
    autocorrelations and multiple comparisons. This orchestrator handles
    analytical binomial tests (fast, theoretical chance), full-pipeline
    permutation tests (rigorous, empirical null; labels permuted between
    units, within strata, or between blocks within strata, as
    ``chance.permutation_scheme`` sets) and group-level sign-flip tests of
    leave-one-group-out fold scores (no refits).

    Parameters
    ----------
    observed_result : ExperimentResult
        The result of the actual experiment run.
    experiment_config : ExperimentConfig
        The full configuration of the experiment.
    X, y : np.ndarray
        The raw features and targets.
    groups : np.ndarray, optional
        CV grouping vector.
    sample_ids : np.ndarray
        Unique identifiers for samples.
    sample_metadata : pd.DataFrame, optional
        Metadata for unit resolution.
    feature_names : list of str, optional
        Names of input features.
    time_axis : np.ndarray, optional
        Time coordinates for temporal data.
    observation_level : str
        Level of input rows ('sample' or 'epoch').
    inferential_unit : str
        Definition of statistical independence ('sample' or 'subject').

    Returns
    -------
    assessment_payload : dict
        Summary containing 'rows' (standardized results), 'nulls' and 'meta',
        which records the permutation scheme and its strata and block columns.

    Raises
    ------
    ValueError
        If the experiment produced no predictions; if a unit, strata or block
        column is missing; if a permuted unit or block holds more than one
        label, or the scoring unit spans several permuted units; or, for the
        sign-flip test, if a test fold holds more than one group or the metric
        has no known chance level and ``chance.p0`` is unset.

    Examples
    --------
    >>> # Internal use within Experiment.run()
    >>> # res = run_statistical_assessment(observed, config, X, y, ...)

    See Also
    --------
    binomial_accuracy_test : Core analytical test.
    assess_post_hoc_permutation : Fast post-hoc alternative.
    """
    stats_config = experiment_config.statistical_assessment
    unit = inferential_unit
    # A unit set in the config takes precedence over the one run() was given.
    if stats_config.unit_of_inference == "sample":
        unit = "sample"
    elif stats_config.unit_of_inference in {"group_mean", "group_majority"}:
        unit = "group"
        stats_config = stats_config.model_copy(
            update={
                "custom_aggregation": stats_config.unit_of_inference.removeprefix(
                    "group_"
                )
            }
        )
    elif stats_config.unit_of_inference == "custom":
        if not stats_config.custom_unit_column:
            raise ValueError(
                "unit_of_inference='custom' requires custom_unit_column to be set."
            )
        unit = "custom"
    chance = stats_config.chance
    if chance.permutation_scheme != "units" and stats_config.unit_of_inference is None:
        # These schemes exchange single samples or blocks of samples, so the
        # score is computed per sample unless a nested unit is set explicitly.
        unit = "sample"
    metrics = experiment_config.get_all_evaluation_metrics()
    if stats_config.metrics:
        metrics = [m for m in metrics if m in stats_config.metrics]
    rows: list[dict[str, Any]] = []
    nulls: dict[str, dict[str, Any]] = {}

    predictions = observed_result.get_predictions()
    if predictions.empty or "Model" not in predictions.columns:
        n_groups = len(np.unique(groups)) if groups is not None else 0
        raise ValueError(
            "Statistical assessment cannot run because the experiment "
            "produced no per-fold predictions. This typically means too "
            f"few groups ({n_groups}) for the requested CV "
            f"(``{experiment_config.cv.strategy}``, "
            f"``n_splits={experiment_config.cv.n_splits}``) or an "
            "earlier fold failure. Either increase the number of "
            "subjects/groups, reduce ``cv.n_splits``, or disable "
            "``statistical_assessment.enabled``."
        )

    for model in observed_result.raw:
        if "error" in observed_result.raw[model]:
            continue
        model_predictions = observed_result.get_predictions()
        model_predictions = model_predictions[model_predictions["Model"] == model]
        for metric in metrics:
            method = stats_config.chance.method
            if method == "auto":
                method = (
                    "binomial"
                    if (
                        metric == "accuracy"
                        and stats_config.chance.p0
                        and chance.permutation_scheme == "units"
                    )
                    else "permutation"
                )

            if method == "sign_flip":
                flip_rows, flip_null = _run_sign_flip_assessment(
                    model, metric, model_predictions, stats_config
                )
                rows.extend(flip_rows)
                if chance.store_null_distribution:
                    nulls.setdefault(model, {})[metric] = flip_null
                continue

            if method == "binomial":
                rows.extend(
                    _run_binomial_assessment(
                        model,
                        metric,
                        model_predictions,
                        experiment_config.task,
                        stats_config,
                        unit,
                    )
                )
                continue

            perm_rows, perm_null = _run_permutation_assessment(
                model,
                metric,
                observed_result,
                experiment_config,
                X,
                y,
                groups,
                sample_ids,
                sample_metadata,
                feature_names,
                time_axis,
                observation_level,
                inferential_unit,
                stats_config,
                unit,
            )
            rows.extend(perm_rows)
            if stats_config.chance.store_null_distribution and perm_null is not None:
                nulls.setdefault(model, {})[metric] = perm_null

    return {
        "rows": rows,
        "nulls": nulls,
        "meta": {
            "enabled": True,
            "method": stats_config.chance.method,
            "resolved_unit_of_inference": unit,
            "metrics": metrics,
            "n_permutations": stats_config.chance.n_permutations,
            "alpha": stats_config.confidence_intervals.alpha,
            "temporal_correction": stats_config.chance.temporal_correction,
            "permutation_scheme": chance.permutation_scheme,
            "strata_column": chance.strata_column,
            "block_column": chance.block_column,
        },
    }


def _run_binomial_assessment(
    model: str,
    metric: str,
    predictions: pd.DataFrame,
    task: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> list[dict[str, Any]]:
    """
    Internal driver for analytical binomial significance testing.

    Examples
    --------
    >>> # rows = _run_binomial_assessment(
    >>> #     "LR", "accuracy", preds, "classification", config, "subject"
    >>> # )
    """
    if task != "classification" or metric != "accuracy":
        raise ValueError(
            "Analytical binomial testing only supports classification accuracy."
        )

    aggregated = aggregate_predictions_for_inference(
        predictions,
        metric=metric,
        task=task,
        unit_of_inference=unit,
        custom_unit_column=config.custom_unit_column,
        custom_aggregation=config.custom_aggregation,
        require_single_prediction=True,
    )
    p0 = config.chance.p0
    if p0 == "auto":
        n_classes = len(pd.unique(aggregated["y_true"]))
        p0 = 1.0 / n_classes

    temporal_cols = [
        col
        for col in TEMPORAL_COLUMNS
        if col in aggregated and aggregated[col].notna().any()
    ]

    n_units = aggregated["InferentialUnitID"].nunique()

    if not temporal_cols:
        result = binomial_accuracy_test(
            aggregated["y_true"],
            aggregated["y_pred"],
            p0=p0,
            alpha=config.confidence_intervals.alpha,
            ci_method=config.confidence_intervals.method,
        )
        return [
            _build_binomial_row(
                model, metric, result, unit, p0, n_units, config, (), ()
            )
        ]

    rows = []
    for key, group in aggregated.groupby(temporal_cols, dropna=False):
        coord_key = (key,) if not isinstance(key, tuple) else key
        result = binomial_accuracy_test(
            group["y_true"],
            group["y_pred"],
            p0=p0,
            alpha=config.confidence_intervals.alpha,
            ci_method=config.confidence_intervals.method,
        )
        rows.append(
            _build_binomial_row(
                model,
                metric,
                result,
                unit,
                p0,
                n_units,
                config,
                coord_key,
                temporal_cols,
            )
        )
    return rows


def _build_binomial_row(
    model: str,
    metric: str,
    result: dict[str, Any],
    unit: str,
    p0: float,
    n_units: int,
    config: StatisticalAssessmentConfig,
    key: tuple,
    temporal_cols: list[str],
) -> dict[str, Any]:
    """Format a binomial test result into a standardized result row."""
    coord = _coord_dict(key, temporal_cols)
    return {
        "Model": model,
        "Metric": metric,
        "Observed": result["observed"],
        "InferentialUnit": unit,
        "NEff": n_units,
        "NullMethod": "binomial",
        "NPermutations": None,
        "P0": p0,
        "PValue": result["p_value"],
        "CILower": result["ci_lower"],
        "CIUpper": result["ci_upper"],
        "CorrectionMethod": "none",
        "ChanceThreshold": result["chance_threshold"],
        "Significant": result["p_value"] <= config.confidence_intervals.alpha,
        "Assumptions": "classification accuracy; one prediction per unit",
        "Caveat": f"Independence assumed at the '{unit}' level.",
        **coord,
    }


def _run_permutation_assessment(
    model: str,
    metric: str,
    observed_result: Any,
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    sample_ids: np.ndarray,
    sample_metadata: pd.DataFrame | None,
    feature_names: Sequence[str] | None,
    time_axis: np.ndarray | None,
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Internal driver for full-pipeline permutation testing.

    Examples
    --------
    >>> # rows, nulls = _run_permutation_assessment(
    >>> #     "LR", "accuracy", res, cfg, X, y, ...
    >>> # )
    """
    observed_predictions = observed_result.get_predictions()
    observed_predictions = observed_predictions[observed_predictions["Model"] == model]
    observed_agg = aggregate_predictions_for_inference(
        observed_predictions,
        metric=metric,
        task=experiment_config.task,
        unit_of_inference=unit,
        custom_unit_column=config.custom_unit_column,
        custom_aggregation=config.custom_aggregation,
    )
    observed_scores = _score_by_coordinates(observed_agg, metric)
    score_keys = list(observed_scores.keys())
    n_units = observed_agg["InferentialUnitID"].nunique()
    temporal_cols = [
        col
        for col in TEMPORAL_COLUMNS
        if col in observed_agg and observed_agg[col].notna().any()
    ]

    null_array = _run_permutation_loop(
        model=model,
        metric=metric,
        score_keys=score_keys,
        experiment_config=experiment_config,
        X=X,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        sample_metadata=sample_metadata,
        feature_names=feature_names,
        time_axis=time_axis,
        observation_level=observation_level,
        inferential_unit=inferential_unit,
        config=config,
        unit=unit,
    )

    # Bootstrap Observed Scores for Confidence Intervals
    boot_array = _bootstrap_scores(
        observed_agg,
        metric=metric,
        score_keys=score_keys,
        n_bootstraps=1000,
        random_state=config.random_state,
    )

    obs_ci_lower = np.nanpercentile(boot_array, 2.5, axis=0)
    obs_ci_upper = np.nanpercentile(boot_array, 97.5, axis=0)

    observed_array = np.asarray([observed_scores[key] for key in score_keys])
    rows = _build_permutation_rows(
        model=model,
        metric=metric,
        observed_array=observed_array,
        null_array=null_array,
        obs_ci_lower=obs_ci_lower,
        obs_ci_upper=obs_ci_upper,
        score_keys=score_keys,
        temporal_cols=temporal_cols,
        unit=unit,
        n_units=n_units,
        config=config,
        task=experiment_config.task,
    )

    null_payload = None
    if config.chance.store_null_distribution:
        null_payload = {
            "metric": metric,
            "unit": unit,
            "coordinates": [_coord_dict(key, temporal_cols) for key in score_keys],
            "values": null_array,
        }
    return rows, null_payload


def _run_permutation_loop(
    model: str,
    metric: str,
    score_keys: list[tuple],
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    sample_ids: np.ndarray,
    sample_metadata: pd.DataFrame | None,
    feature_names: Sequence[str] | None,
    time_axis: np.ndarray | None,
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> np.ndarray:
    """
    Execute the core permutation loop using parallel processing.
    """
    # Import via the package re-export rather than the ``experiment`` submodule
    # so this lazy rerun dependency does not register as an import cycle edge.
    from coco_pipe.decoding import Experiment

    rng = np.random.default_rng(config.random_state)
    # run() capitalises metadata columns, so they are matched ignoring case.
    columns = {
        str(column).casefold(): column
        for column in (sample_metadata.columns if sample_metadata is not None else [])
    }
    target_col = config.custom_unit_column if unit == "custom" else unit
    if unit == "sample":
        unit_values = np.arange(len(y))
    elif groups is not None and unit in {"Group", "group"}:
        unit_values = np.asarray(groups)
    elif str(target_col).casefold() in columns:
        unit_values = sample_metadata[columns[str(target_col).casefold()]].to_numpy()
    else:
        raise ValueError(f"Could not resolve unit values for '{unit}'.")
    if config.chance.permutation_scheme == "units":
        unique_units, unit_map_idx = np.unique(unit_values, return_inverse=True)
        n_unique = len(unique_units)

        unit_labels_orig = []
        for u in unique_units:
            mask = unit_values == u
            u_y = y[mask]
            if experiment_config.task == "classification":
                if np.unique(u_y).size > 1:
                    raise ValueError(
                        f"The label varies within inference unit {u!r}; a "
                        "permutation test swaps whole units, so each unit needs "
                        "one label. Use a unit inside which the target is "
                        "constant, or keep the units and permute inside them "
                        "with chance.permutation_scheme='within_strata' or "
                        "'blocks_within_strata'."
                    )
                unit_labels_orig.append(u_y[0])
            else:
                unit_labels_orig.append(np.mean(u_y))
        unit_labels_orig = np.array(unit_labels_orig)

        def _permuted_labels(local_rng):
            perm_idx = local_rng.permutation(n_unique)
            return unit_labels_orig[perm_idx][unit_map_idx]

    else:
        _permuted_labels = _scheme_label_permuter(
            y, unit, unit_values, config, groups, sample_metadata
        )

    import joblib

    parallel = joblib.Parallel(n_jobs=experiment_config.n_jobs)

    def _run_single_permutation(p_idx, seed):
        local_rng = np.random.default_rng(seed)
        y_perm = _permuted_labels(local_rng)
        perm_config = experiment_config.model_copy(deep=True)
        # A permutation run must not recursively trigger its own statistical
        # assessment: that nests permutation-within-permutation (runaway cost and
        # degenerate inner folds). Score the permuted fit only.
        if perm_config.statistical_assessment is not None:
            perm_config.statistical_assessment.enabled = False
        p_res = Experiment(perm_config).run(
            X,
            y_perm,
            groups=groups,
            feature_names=feature_names,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            observation_level=observation_level,
            inferential_unit=inferential_unit,
            time_axis=time_axis,
        )
        # A failed draw cannot be dropped: the null would no longer be the
        # permutation distribution.
        failure = p_res.raw.get(model)
        if isinstance(failure, dict) and failure.get("status") == "failed":
            raise RuntimeError(
                f"Model {model!r} failed on permuted labels: {failure['error']}"
            )
        p_preds = p_res.get_predictions()
        p_preds = p_preds[p_preds["Model"] == model]
        p_agg = aggregate_predictions_for_inference(
            p_preds,
            metric=metric,
            task=experiment_config.task,
            unit_of_inference=unit,
            custom_unit_column=config.custom_unit_column,
            custom_aggregation=config.custom_aggregation,
        )
        p_scores = _score_by_coordinates(p_agg, metric)
        return [p_scores[key] for key in score_keys]

    seeds = rng.integers(0, 2**32, size=config.chance.n_permutations)
    results = parallel(
        joblib.delayed(_run_single_permutation)(i, seeds[i])
        for i in range(config.chance.n_permutations)
    )

    return np.array(results)


def _metadata_values(
    name: str,
    role: str,
    groups: np.ndarray | None,
    sample_metadata: pd.DataFrame | None,
) -> np.ndarray:
    """One value per sample from the CV groups or a sample-metadata column.

    ``"group"`` names the CV groups; any other name is matched ignoring case,
    because run() capitalises metadata columns.
    """
    if str(name).casefold() == "group" and groups is not None:
        return np.asarray(groups)
    available = list(sample_metadata.columns) if sample_metadata is not None else []
    columns = {str(column).casefold(): column for column in available}
    column = columns.get(str(name).casefold())
    if column is None:
        raise ValueError(
            f"{role} '{name}' not found in sample_metadata. Available: {available}"
            + ("" if groups is None else "; 'group' names the CV groups.")
        )
    return sample_metadata[column].to_numpy()


def _label_permuter(
    y: np.ndarray,
    strata: np.ndarray,
    blocks: np.ndarray | None = None,
) -> tuple[Callable[[np.random.Generator], np.ndarray], np.ndarray]:
    """Build a draw of permuted labels that never crosses a stratum.

    Without *blocks*, the labels of the samples of each stratum are shuffled
    among those samples, so each stratum keeps its class counts. With *blocks*,
    a block is one value of *blocks* within one stratum; it must carry a single
    label, and the block labels of each stratum are shuffled among its blocks.

    Returns the draw, a callable taking a generator, and the exchangeable unit
    of every sample: its position without blocks, its block id with them.
    """
    y = np.asarray(y)
    strata = np.asarray(strata)
    frame = pd.DataFrame({"stratum": strata})
    if frame["stratum"].isna().any():
        raise ValueError(
            f"{int(frame['stratum'].isna().sum())} samples have no stratum; every "
            "sample needs one for a stratified permutation."
        )
    if blocks is None:
        rows = list(frame.groupby("stratum", sort=False).indices.values())
        exchange_ids = np.arange(len(y))
        informative = any(np.unique(y[r]).size > 1 for r in rows)

        def draw(rng: np.random.Generator) -> np.ndarray:
            out = y.copy()
            for r in rows:
                out[r] = y[rng.permutation(r)]
            return out

    else:
        frame["block"] = np.asarray(blocks)
        if frame["block"].isna().any():
            raise ValueError(
                f"{int(frame['block'].isna().sum())} samples have no block; every "
                "sample needs one for a block permutation."
            )
        by_block = frame.groupby(["stratum", "block"], sort=False)
        exchange_ids = by_block.ngroup().to_numpy()
        strata_blocks: dict[Any, list[np.ndarray]] = {}
        for (stratum, block), r in by_block.indices.items():
            labels = pd.unique(y[r])
            if len(labels) > 1:
                block, stratum = (
                    v.item() if isinstance(v, np.generic) else v
                    for v in (block, stratum)
                )
                raise ValueError(
                    f"Block {block!r} of stratum {stratum!r} holds more than one "
                    f"label ({sorted(labels.tolist())}); "
                    "permutation_scheme='blocks_within_strata' moves labels block "
                    "by block, so each block needs a single label."
                )
            strata_blocks.setdefault(stratum, []).append(r)
        layout = [
            (block_rows, np.array([y[r[0]] for r in block_rows]))
            for block_rows in strata_blocks.values()
        ]
        informative = any(np.unique(labels).size > 1 for _, labels in layout)

        def draw(rng: np.random.Generator) -> np.ndarray:
            out = y.copy()
            for block_rows, labels in layout:
                for r, label in zip(
                    block_rows, labels[rng.permutation(len(labels))], strict=True
                ):
                    out[r] = label
            return out

    if not informative:
        raise ValueError(
            "No stratum holds two exchangeable "
            + ("samples" if blocks is None else "blocks")
            + " with different labels, so every permutation returns the observed "
            "labels and the null is degenerate."
        )
    return draw, exchange_ids


def permute_labels_within_strata(
    y: Sequence[Any],
    strata: Sequence[Any],
    *,
    blocks: Sequence[Any] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Permute labels without moving any of them across a stratum.

    Without *blocks*, sample labels are shuffled inside each stratum, which
    keeps each stratum's class counts. With *blocks*, each block (one value of
    *blocks* within one stratum) must carry a single label, and block labels
    are shuffled between the blocks of each stratum. This is the draw the
    full-pipeline permutation test makes for
    ``permutation_scheme="within_strata"`` and ``"blocks_within_strata"``.

    Parameters
    ----------
    y : sequence
        One label per sample.
    strata : sequence
        One stratum per sample (e.g. the subject).
    blocks : sequence, optional
        One block per sample (e.g. the run), for a blocked design.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for the draw.

    Returns
    -------
    y_perm : np.ndarray
        The permuted labels, aligned with *y*.

    Raises
    ------
    ValueError
        If a block holds more than one label, or no stratum holds two
        exchangeable units with different labels.

    Examples
    --------
    >>> from coco_pipe.decoding import permute_labels_within_strata
    >>> y = [0, 1, 0, 1, 0, 1]
    >>> subjects = ["S1", "S1", "S1", "S2", "S2", "S2"]
    >>> y_perm = permute_labels_within_strata(y, subjects, random_state=0)
    """
    draw, _ = _label_permuter(
        np.asarray(y),
        np.asarray(strata),
        None if blocks is None else np.asarray(blocks),
    )
    return draw(np.random.default_rng(random_state))


def _scheme_label_permuter(
    y: np.ndarray,
    unit: str,
    unit_values: np.ndarray,
    config: StatisticalAssessmentConfig,
    groups: np.ndarray | None,
    sample_metadata: pd.DataFrame | None,
) -> Callable[[np.random.Generator], np.ndarray]:
    """The label draw of a stratified scheme, checked against the scoring unit."""
    chance = config.chance
    strata = _metadata_values(
        chance.strata_column, "Strata column", groups, sample_metadata
    )
    blocks = None
    if chance.permutation_scheme == "blocks_within_strata":
        blocks = _metadata_values(
            chance.block_column, "Block column", groups, sample_metadata
        )
    draw, exchange_ids = _label_permuter(y, strata, blocks)
    if unit != "sample":
        # A scored unit takes one label (or one mean target); the draw keeps it
        # one label only if the unit lies inside a single exchangeable unit.
        spans = pd.Series(exchange_ids).groupby(unit_values).nunique()
        if (spans > 1).any():
            exchanged = "sample" if blocks is None else f"'{chance.block_column}' block"
            raise ValueError(
                f"Inference unit {spans.index[spans > 1][0]!r} spans more than one "
                f"{exchanged}, but permutation_scheme="
                f"'{chance.permutation_scheme}' moves labels {exchanged} by "
                f"{exchanged}. Use unit_of_inference='sample' or a unit nested "
                "inside what is permuted."
            )
    return draw


def _permutation_description(
    config: StatisticalAssessmentConfig, unit: str, task: str
) -> tuple[str, str]:
    """The Assumptions and Caveat texts of a full-pipeline permutation row."""
    chance = config.chance
    if chance.permutation_scheme == "within_strata":
        return (
            "full outer-CV pipeline rerun with labels shuffled within each "
            f"'{chance.strata_column}' stratum",
            f"Labels assumed exchangeable within '{chance.strata_column}' strata.",
        )
    if chance.permutation_scheme == "blocks_within_strata":
        return (
            f"full outer-CV pipeline rerun with '{chance.block_column}' block "
            f"labels shuffled within each '{chance.strata_column}' stratum",
            f"Labels assumed exchangeable between '{chance.block_column}' blocks "
            f"within '{chance.strata_column}' strata.",
        )
    return (
        "full outer-CV pipeline rerun under label permutations; "
        "regression targets averaged by unit"
        if task == "regression" and unit != "sample"
        else "full outer-CV pipeline rerun under label permutations",
        f"Independence assumed at the '{unit}' level.",
    )


# Chance levels the sign-flip test knows without an explicit p0; accuracy and
# balanced accuracy use one over the number of classes.
_SIGN_FLIP_CHANCE = {
    "roc_auc": 0.5,
    "roc_auc_ovr_weighted": 0.5,
    "matthews_corrcoef": 0.0,
    "cohen_kappa": 0.0,
}


def _sign_flip_chance(metric: str, p0: Any, y_true: pd.Series) -> float:
    """The chance level per-fold scores are compared with."""
    if p0 is not None and p0 != "auto":
        return float(p0)
    if metric in {"accuracy", "balanced_accuracy"}:
        return 1.0 / y_true.nunique()
    if metric in _SIGN_FLIP_CHANCE:
        return _SIGN_FLIP_CHANCE[metric]
    raise ValueError(
        f"method='sign_flip' has no known chance level for {metric!r}; set "
        "chance.p0 explicitly."
    )


def _run_sign_flip_assessment(
    model: str,
    metric: str,
    predictions: pd.DataFrame,
    config: StatisticalAssessmentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group-level sign-flip test of per-fold scores against chance.

    Each test fold must hold exactly one group, as in leave-one-group-out CV,
    so a fold score is a group score. Under the null, each group's score minus
    chance is symmetric about zero; the null of the mean flips those signs at
    random, independently per group but jointly over time, so the max-stat
    correction sees the temporal dependence. Nothing is refit.
    """
    if "Group" not in predictions or predictions["Group"].isna().any():
        raise ValueError(
            "method='sign_flip' needs CV groups, one per test fold "
            "(leave-one-group-out)."
        )
    groups_per_fold = predictions.groupby("Fold")["Group"].nunique()
    if (groups_per_fold != 1).any():
        fold = groups_per_fold.index[groups_per_fold != 1][0]
        raise ValueError(
            f"method='sign_flip' needs one group per test fold "
            f"(leave-one-group-out); fold {fold} tests "
            f"{groups_per_fold[fold]} groups."
        )
    folds_per_group = predictions.groupby("Group")["Fold"].nunique()
    if (folds_per_group > 1).any():
        raise ValueError(
            f"Group {folds_per_group.index[folds_per_group > 1][0]!r} is tested in "
            "more than one fold; method='sign_flip' needs one score per group."
        )

    chance = _sign_flip_chance(metric, config.chance.p0, predictions["y_true"])
    temporal_cols = [
        col
        for col in TEMPORAL_COLUMNS
        if col in predictions and predictions[col].notna().any()
    ]
    score_keys = list(_score_by_coordinates(predictions, metric).keys())
    fold_scores = []
    for fold, frame in predictions.groupby("Fold", sort=True):
        scores = _score_by_coordinates(frame, metric)
        values = np.array([scores.get(key, np.nan) for key in score_keys], float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"The {metric} score of fold {fold} (group "
                f"{frame['Group'].iloc[0]!r}) is not finite; method='sign_flip' "
                "needs a score for every group."
            )
        fold_scores.append(values)
    scores = np.vstack(fold_scores)
    n_groups = len(scores)
    deviations = scores - chance

    sign_seed, boot_seed = np.random.SeedSequence(config.random_state).spawn(2)
    signs = np.random.default_rng(sign_seed).choice(
        [-1.0, 1.0], size=(config.chance.n_permutations, n_groups)
    )
    null_array = chance + signs @ deviations / n_groups
    observed_array = scores.mean(axis=0)
    # p-values use the one-sample t, so max-stat weighs timepoints by their
    # spread across groups. Flips keep the sum of squared deviations, so each
    # draw's t follows from its mean alone. Discrete scores tie often, so t is
    # rounded to make ties exact whatever the summation order.
    sum_sq = (deviations**2).sum(axis=0)

    def t_stat(mean_score: np.ndarray) -> np.ndarray:
        m = mean_score - chance
        se = np.sqrt(
            np.maximum(sum_sq - n_groups * m**2, 0) / (n_groups * (n_groups - 1))
        )
        infinite = np.where(m == 0, 0.0, np.copysign(np.inf, m))
        return np.round(np.divide(m, se, out=infinite, where=se > 0), 9)

    resampled = np.random.default_rng(boot_seed).integers(
        0, n_groups, size=(1000, n_groups)
    )
    boot_array = scores[resampled].mean(axis=1)

    rows = _build_permutation_rows(
        model=model,
        metric=metric,
        observed_array=observed_array,
        null_array=null_array,
        obs_ci_lower=np.percentile(boot_array, 2.5, axis=0),
        obs_ci_upper=np.percentile(boot_array, 97.5, axis=0),
        score_keys=score_keys,
        temporal_cols=temporal_cols,
        unit="group",
        n_units=n_groups,
        config=config,
        task="classification",
        null_method="sign_flip_group",
        statistic=(t_stat(observed_array), t_stat(null_array)),
        assumptions=(
            f"one group per test fold; per-group {metric} minus chance "
            f"({chance:g}) symmetric about zero under the null; no refits"
        ),
        caveat="Independence assumed at the 'group' level.",
    )
    for row in rows:
        row["P0"] = chance
    null_payload = {
        "metric": metric,
        "unit": "group",
        "coordinates": [_coord_dict(key, temporal_cols) for key in score_keys],
        "values": null_array,
    }
    return rows, null_payload


def _build_permutation_rows(
    model: str,
    metric: str,
    observed_array: np.ndarray,
    null_array: np.ndarray,
    obs_ci_lower: np.ndarray,
    obs_ci_upper: np.ndarray,
    score_keys: list[tuple[Any, ...]],
    temporal_cols: list[str],
    unit: str,
    n_units: int,
    config: StatisticalAssessmentConfig,
    task: str,
    null_method: str = "permutation_full_pipeline",
    statistic: tuple[np.ndarray, np.ndarray] | None = None,
    assumptions: str | None = None,
    caveat: str | None = None,
) -> list[dict[str, Any]]:
    """Format a permutation test result into a standardized result row.

    p-values compare *observed_array* with *null_array*, or the observed and
    null arrays of *statistic* when it is given.
    """
    if assumptions is None or caveat is None:
        default_assumptions, default_caveat = _permutation_description(
            config, unit, task
        )
        assumptions = assumptions or default_assumptions
        caveat = caveat or default_caveat
    metric_spec = get_metric_spec(metric)
    greater_is_better = metric_spec.greater_is_better

    stat_observed, stat_null = statistic or (observed_array, null_array)
    p_values = _empirical_p_values(stat_observed, stat_null, greater_is_better)

    corrected = _correct_p_values(
        stat_observed,
        stat_null,
        p_values,
        config.chance.temporal_correction,
        metric_spec.greater_is_better,
    )

    null_median = np.nanmedian(null_array, axis=0)
    null_lower = np.nanpercentile(null_array, 2.5, axis=0)
    null_upper = np.nanpercentile(null_array, 97.5, axis=0)

    rows = []
    for idx, key in enumerate(score_keys):
        coord = _coord_dict(key, temporal_cols)
        rows.append(
            {
                "Model": model,
                "Metric": metric,
                "Observed": observed_array[idx],
                "InferentialUnit": unit,
                "NEff": n_units,
                "NullMethod": null_method,
                "NPermutations": config.chance.n_permutations,
                "P0": null_median[idx],
                "PValue": p_values[idx],
                "CILower": obs_ci_lower[idx],
                "CIUpper": obs_ci_upper[idx],
                "CorrectionMethod": config.chance.temporal_correction,
                "CorrectedPValue": corrected[idx],
                "ChanceThreshold": np.nanpercentile(
                    null_array[:, idx], 95 if greater_is_better else 5
                ),
                "NullMedian": null_median[idx],
                "NullLower": null_lower[idx],
                "NullUpper": null_upper[idx],
                "Significant": corrected[idx] <= config.confidence_intervals.alpha,
                "Assumptions": assumptions,
                "Caveat": caveat,
                **coord,
            }
        )
    return rows


def _require_same_samples(merged: pd.DataFrame, n_a: int, n_b: int) -> None:
    """Refuse a paired test unless every row of both runs pairs with matching labels."""
    if not len(merged) == n_a == n_b:
        raise ValueError(
            "A paired comparison needs both runs scored on exactly the same samples: "
            f"run A has {n_a} prediction rows, run B has {n_b}, {len(merged)} pair up."
        )
    if "y_true_A" in merged and not merged["y_true_A"].equals(merged["y_true_B"]):
        raise ValueError("Paired runs disagree on y_true for the same SampleID.")


def paired_prediction_test(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    metric: str,
    unit: str = "group",
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Paired permutation test between two runs' out-of-fold predictions.

    Each frame holds one model's predictions (``get_predictions()`` filtered to
    that model, or its ``predictions.csv``). Rows are paired by ``SampleID`` and
    ``y_true``; the runs may have used different folds. Raises ``ValueError``
    unless both runs scored exactly the same samples. Model predictions are
    swapped within *unit* (default: the CV group), so significance reflects the
    number of independent units rather than rows.
    """
    from ._diagnostics import scalar_prediction_frame

    frame_a = scalar_prediction_frame(predictions_a)
    frame_b = scalar_prediction_frame(predictions_b)
    keys = ["SampleID", "y_true"]
    merged = frame_a.merge(frame_b, on=keys, suffixes=("_A", "_B"))
    _require_same_samples(merged, len(frame_a), len(frame_b))
    return assess_paired_comparison(
        merged,
        metric=metric,
        unit=unit,
        n_permutations=n_permutations,
        random_state=random_state,
    )


def run_paired_permutation_assessment(
    results_a: ExperimentResult,
    results_b: ExperimentResult,
    model: str,
    metric: str,
    config: StatisticalAssessmentConfig,
) -> pd.DataFrame:
    """
    Run a paired permutation test to compare two experimental results.

    Tests the null hypothesis that the difference between two models is zero
    by randomly swapping model labels within each independent unit.

    Scientific Rationale
    --------------------
    This function performs a rigorous comparison of two experimental pipelines
    by aligning predictions at the 'SampleID' level and performing
    within-unit label swaps. This ensures that the comparison is not biased
    by unit-specific performance baselines and correctly estimates the
    p-value for the observed performance delta.

    Parameters
    ----------
    results_a, results_b : ExperimentResult
        The results of the two experiments to compare.
    model : str
        The name of the model to compare.
    metric : str
        Metric to use for the comparison.
    config : StatisticalAssessmentConfig
        Configuration for permutations and significance.

    Returns
    -------
    paired_df : pd.DataFrame
        DataFrame with Difference, PValue, and confidence intervals.

    Examples
    --------
    >>> # diff = run_paired_permutation_assessment(res1, res2, 'LR', 'accuracy', config)

    See Also
    --------
    assess_paired_comparison : Fast post-hoc alternative.

    Examples
    --------
    >>> # paired_df = run_paired_permutation_assessment(
    >>> #     res_a, res_b, "LR", "accuracy", config
    >>> # )
    """
    from ._diagnostics import score_frame

    preds_a = results_a.get_predictions()
    preds_b = results_b.get_predictions()
    preds_a = preds_a[preds_a["Model"] == model]
    preds_b = preds_b[preds_b["Model"] == model]

    # Pair by sample, not by fold: each run's prediction is out-of-fold, and two
    # runs over the same samples can still draw different folds (e.g. different
    # row order), which would silently drop pairs from a fold-keyed join.
    merge_cols = ["SampleID"]
    temporal_cols = [c for c in TEMPORAL_COLUMNS if c in preds_a]
    merge_cols.extend(temporal_cols)

    # `unit_of_inference` names a *concept* (sample/group/custom), not a literal
    # predictions column, so it must be resolved before use. "group_mean" and
    # "group_majority" both mean "swap by CV group" here — the aggregation mode
    # they also encode elsewhere is irrelevant to this permutation, which swaps
    # whole units rather than aggregating rows.
    if config.unit_of_inference == "sample" or config.unit_of_inference is None:
        unit_col = "SampleID"
    elif config.unit_of_inference == "custom":
        if not config.custom_unit_column:
            raise ValueError(
                "unit_of_inference='custom' requires custom_unit_column to be set."
            )
        unit_col = config.custom_unit_column
    elif config.unit_of_inference in {"group_mean", "group_majority"}:
        unit_col = "Group"
    else:
        unit_col = config.unit_of_inference
    if unit_col not in preds_a.columns:
        raise ValueError(
            f"unit_of_inference resolved to column {unit_col!r}, which is not "
            f"present in the predictions ({sorted(preds_a.columns)})."
        )
    if unit_col not in merge_cols:
        merge_cols.append(unit_col)

    merged = pd.merge(preds_a, preds_b, on=merge_cols, suffixes=("_A", "_B"))
    if merged.empty:
        raise ValueError("Could not align predictions for paired test.")
    _require_same_samples(merged, len(preds_a), len(preds_b))

    def get_diff(group: pd.DataFrame) -> float:
        s_a = score_frame(
            group.filter(regex=".*_A$|SampleID|y_true").rename(
                columns=lambda x: x[:-2] if x.endswith("_A") else x
            ),
            metric,
        )
        s_b = score_frame(
            group.filter(regex=".*_B$|SampleID|y_true").rename(
                columns=lambda x: x[:-2] if x.endswith("_B") else x
            ),
            metric,
        )
        return s_a - s_b

    obs_scores_dummy = _score_by_coordinates(preds_a, metric)
    score_keys = list(obs_scores_dummy.keys())

    # Split once into per-coordinate (e.g. per-timepoint) row groups via a
    # single native groupby, instead of re-scanning the whole merged frame
    # with a boolean mask for every (permutation, coordinate) pair below —
    # that loop is O(n_permutations * n_coordinates^2 * n_rows) otherwise,
    # dominated by repeated full-frame comparisons.
    group_indices = _coordinate_groups(merged, temporal_cols)

    observed_diff_array = np.zeros(len(score_keys))
    for idx, key in enumerate(score_keys):
        observed_diff_array[idx] = get_diff(merged.iloc[group_indices[key]])

    boot_results = _bootstrap_scores_paired(
        merged,
        metric=metric,
        score_keys=score_keys,
        temporal_cols=temporal_cols,
        unit_col=unit_col,
        n_bootstraps=getattr(config, "n_bootstraps", 1000),
        random_state=config.random_state,
    )

    obs_ci_lower = np.nanpercentile(boot_results, 2.5, axis=0)
    obs_ci_upper = np.nanpercentile(boot_results, 97.5, axis=0)

    import joblib

    n_perm = config.chance.n_permutations
    perm_rng = np.random.default_rng(config.random_state + 1)
    unique_units = merged[unit_col].unique()
    n_units = len(unique_units)

    def _run_single_perm(seed):
        local_rng = np.random.default_rng(seed)
        swaps = local_rng.choice([False, True], size=n_units)
        swap_units = unique_units[swaps]

        perm_merged = merged.copy()
        if np.any(swaps):
            mask = merged[unit_col].isin(swap_units)
            cols_a = [c for c in merged.columns if c.endswith("_A")]
            for c_a in cols_a:
                c_b = c_a[:-2] + "_B"
                a_vals = merged.loc[mask, c_a].copy()
                perm_merged.loc[mask, c_a] = merged.loc[mask, c_b]
                perm_merged.loc[mask, c_b] = a_vals

        # Swapping only rewrites the "_A"/"_B" columns in place on a copy of
        # `merged`, so row membership per coordinate is unchanged and the
        # groupby done once above still applies.
        p_diffs = np.empty(len(score_keys))
        for idx, key in enumerate(score_keys):
            p_diffs[idx] = get_diff(perm_merged.iloc[group_indices[key]])
        return p_diffs

    seeds = perm_rng.integers(0, 2**32, size=n_perm)
    n_jobs = getattr(config, "n_jobs", 1)
    null_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_single_perm)(s) for s in seeds
    )
    null_array = np.array(null_results)

    p_values = _empirical_p_values(
        observed_diff_array, null_array, greater_is_better=True, two_sided=True
    )
    corrected = _correct_p_values(
        observed_diff_array,
        null_array,
        p_values,
        config.chance.temporal_correction,
        greater_is_better=True,
    )

    null_median = np.nanmedian(null_array, axis=0)
    null_lower = np.nanpercentile(null_array, 2.5, axis=0)
    null_upper = np.nanpercentile(null_array, 97.5, axis=0)

    rows = []
    for idx, key in enumerate(score_keys):
        coord = _coord_dict(key, temporal_cols)
        rows.append(
            {
                "Model": model,
                "Metric": metric,
                "Comparison": "Paired Difference (A-B)",
                "Observed": observed_diff_array[idx],
                "InferentialUnit": unit_col,
                "NEff": n_units,
                "NullMethod": "paired_permutation",
                "NPermutations": n_perm,
                "P0": null_median[idx],
                "PValue": p_values[idx],
                "CILower": obs_ci_lower[idx],
                "CIUpper": obs_ci_upper[idx],
                "CorrectionMethod": config.chance.temporal_correction,
                "CorrectedPValue": corrected[idx],
                "NullMedian": null_median[idx],
                "NullLower": null_lower[idx],
                "NullUpper": null_upper[idx],
                "Significant": corrected[idx] <= config.confidence_intervals.alpha,
                "Caveat": f"Independence assumed at the '{unit_col}' level.",
                **coord,
            }
        )

    return pd.DataFrame(rows)


def _score_by_coordinates(
    frame: pd.DataFrame, metric: str
) -> dict[tuple[Any, ...], float]:
    """Score predictions across all temporal coordinates."""
    from ._diagnostics import score_frame

    temporal_cols = [
        col for col in TEMPORAL_COLUMNS if col in frame and frame[col].notna().any()
    ]
    if not temporal_cols:
        return {(): score_frame(frame, metric)}

    m_spec = get_metric_spec(metric)
    if m_spec.response_method == "predict" and metric in {
        "accuracy",
        "zero_one_loss",
        "hamming_loss",
    }:
        # A mean per coordinate, so a unit drawn twice by the bootstrap
        # counts twice.
        hits = frame["y_true"].to_numpy() == frame["y_pred"].to_numpy()
        if metric != "accuracy":
            hits = ~hits
        means = (
            pd.Series(hits, index=frame.index, dtype=float)
            .groupby([frame[col] for col in temporal_cols], dropna=False)
            .mean()
        )
        return {
            key if isinstance(key, tuple) else (key,): float(value)
            for key, value in means.items()
        }

    scores = {}
    for key, group in frame.groupby(temporal_cols, dropna=False):
        coord_key = (key,) if not isinstance(key, tuple) else key
        scores[coord_key] = score_frame(group, metric)
    return scores


def _bootstrap_engine(
    units: np.ndarray,
    unit_map: dict[Any, pd.DataFrame],
    score_func: Callable[[pd.DataFrame], np.ndarray],
    n_bootstraps: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Core engine for unit-based bootstrap resampling.

    Examples
    --------
    >>> # boot_dist = _bootstrap_engine(units, unit_map, score_func, n_bootstraps=100)
    """
    rng = np.random.default_rng(random_state)
    n_units = len(units)
    # Each resample takes its rows from one table by position. Concatenating a
    # frame per sampled unit, n_bootstraps times, dominated the whole assessment.
    table = pd.concat([unit_map[u] for u in units])
    stops = np.cumsum([len(unit_map[u]) for u in units])
    positions = {
        u: np.arange(stop - len(unit_map[u]), stop)
        for u, stop in zip(units, stops, strict=True)
    }
    results = []
    for _ in range(n_bootstraps):
        boot_units = rng.choice(units, size=n_units, replace=True)
        rows = np.concatenate([positions[u] for u in boot_units])
        results.append(score_func(table.iloc[rows]))
    return np.array(results)


def _bootstrap_scores(
    frame: pd.DataFrame,
    metric: str,
    score_keys: list[tuple],
    n_bootstraps: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """Resample independent units with replacement and re-score."""
    unique_units = frame["InferentialUnitID"].unique()
    unit_map = {u: frame[frame["InferentialUnitID"] == u] for u in unique_units}

    def score_func(df: pd.DataFrame) -> np.ndarray:
        boot_scores = _score_by_coordinates(df, metric)
        return np.array([boot_scores.get(key, np.nan) for key in score_keys])

    return _bootstrap_engine(
        unique_units, unit_map, score_func, n_bootstraps, random_state
    )


def _bootstrap_scores_paired(
    merged: pd.DataFrame,
    metric: str,
    score_keys: list[tuple],
    temporal_cols: list[str],
    unit_col: str,
    n_bootstraps: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """Resample independent units for paired differences."""
    from ._diagnostics import score_frame

    unique_units = merged[unit_col].unique()
    unit_map = {u: merged[merged[unit_col] == u] for u in unique_units}

    def get_diff(group: pd.DataFrame) -> float:
        s_a = score_frame(
            group.filter(regex=".*_A$|SampleID|y_true").rename(
                columns=lambda x: x[:-2] if x.endswith("_A") else x
            ),
            metric,
        )
        s_b = score_frame(
            group.filter(regex=".*_B$|SampleID|y_true").rename(
                columns=lambda x: x[:-2] if x.endswith("_B") else x
            ),
            metric,
        )
        return s_a - s_b

    def score_func(df: pd.DataFrame) -> np.ndarray:
        # One groupby per resample instead of a full-frame boolean scan per
        # coordinate: the latter is O(n_coordinates^2 * n_rows) per bootstrap
        # and dominates the whole assessment at typical temporal resolutions.
        groups = _coordinate_groups(df, temporal_cols)
        res = np.empty(len(score_keys))
        for idx, key in enumerate(score_keys):
            positions = groups.get(key)
            res[idx] = np.nan if positions is None else get_diff(df.iloc[positions])
        return res

    return _bootstrap_engine(
        unique_units, unit_map, score_func, n_bootstraps, random_state
    )


def _empirical_p_values(
    observed: np.ndarray,
    null: np.ndarray,
    greater_is_better: bool,
    two_sided: bool = False,
) -> np.ndarray:
    """
    Calculate empirical p-values from a null distribution.

    Uses the recommended (k+1)/(n+1) estimator to avoid p=0.
    Supports both one-sided and asymmetric two-sided calculations.

    Parameters
    ----------
    observed : np.ndarray
        The observed scores.
    null : np.ndarray
        Null distribution array (permutations, coordinates).
    greater_is_better : bool
        Metric directionality.
    two_sided : bool, default=False
        Whether to compute a two-sided p-value.

    Returns
    -------
    p_values : np.ndarray
        Empirical p-values for each coordinate.
    """
    if two_sided:
        p_upper = (np.sum(null >= observed, axis=0) + 1) / (null.shape[0] + 1)
        p_lower = (np.sum(null <= observed, axis=0) + 1) / (null.shape[0] + 1)
        return np.minimum(1.0, 2 * np.minimum(p_upper, p_lower))

    if greater_is_better:
        return (np.sum(null >= observed, axis=0) + 1) / (null.shape[0] + 1)
    return (np.sum(null <= observed, axis=0) + 1) / (null.shape[0] + 1)


def _correct_p_values(
    observed: np.ndarray,
    null: np.ndarray,
    p_values: np.ndarray,
    method: str,
    greater_is_better: bool,
) -> np.ndarray:
    """
    Apply multiple-comparison correction across temporal coordinates.

    The methods are those ``ChanceAssessmentConfig.temporal_correction``
    allows: "fdr_bh" (Benjamini-Hochberg), "max_stat" (against the maximum of
    the null over coordinates) and "none".

    Parameters
    ----------
    observed : np.ndarray
        The observed scores.
    null : np.ndarray
        Null distribution array.
    p_values : np.ndarray
        Raw p-values.
    method : str
        Correction method name.
    greater_is_better : bool
        Metric directionality.

    Returns
    -------
    corrected_p : np.ndarray
        Corrected p-values.
    """
    if method == "none" or observed.size == 1:
        return p_values
    if method == "fdr_bh":
        return false_discovery_control(p_values, method="bh")
    if method == "max_stat":
        if greater_is_better:
            max_null = np.nanmax(null, axis=1)
            return (np.sum(max_null[:, None] >= observed[None, :], axis=0) + 1) / (
                null.shape[0] + 1
            )
        min_null = np.nanmin(null, axis=1)
        return (np.sum(min_null[:, None] <= observed[None, :], axis=0) + 1) / (
            null.shape[0] + 1
        )
    raise ValueError(f"Unknown temporal correction: {method}.")


def _accuracy_ci(
    k_correct: np.ndarray,
    n_eff: np.ndarray,
    alpha: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for accuracy, vectorized.

    Parameters
    ----------
    k_correct : np.ndarray
        Number of correct predictions.
    n_eff : np.ndarray
        Effective sample size.
    alpha : float
        Significance level.
    method : str
        CI method ('wilson' or 'clopper_pearson').

    Returns
    -------
    lower, upper : tuple of np.ndarray
        Lower and upper CI bounds.
    """
    if method == "clopper_pearson":
        lower = beta.ppf(alpha / 2, k_correct, n_eff - k_correct + 1)
        lower = np.where(k_correct == 0, 0.0, lower)
        upper = beta.ppf(1 - alpha / 2, k_correct + 1, n_eff - k_correct)
        upper = np.where(k_correct == n_eff, 1.0, upper)
        return lower, upper

    if method != "wilson":
        raise ValueError("ci_method must be 'wilson' or 'clopper_pearson'.")

    z = norm.ppf(1 - alpha / 2)
    phat = k_correct / n_eff
    denom = 1 + z**2 / n_eff
    center = (phat + z**2 / (2 * n_eff)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n_eff)) / n_eff) / denom
    return np.maximum(0.0, center - half), np.minimum(1.0, center + half)


def _coord_dict(key: tuple[Any, ...], names: list[str]) -> dict[str, Any]:
    """Map a coordinate tuple to its dimension names."""
    result = {"Time": None, "TrainTime": None, "TestTime": None}
    if not names or len(key) == 0:
        return result

    for i, name in enumerate(names):
        if i < len(key):
            result[name] = key[i]
    return result


def assess_post_hoc_permutation(
    res: dict[str, Any],
    metric: str = "accuracy",
    unit: str | None = None,
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Perform a post-hoc label permutation assessment on out-of-fold predictions.

    Shuffles labels relative to fixed predictions to estimate the null
    distribution under exchangeability.

    Scientific Rationale
    --------------------
    Unlike full-pipeline permutations, post-hoc permutations do not rerun
    feature selection or tuning. This makes them significantly faster but
    potentially over-optimistic if those steps 'leaked' label information.
    However, if the independence unit (e.g., subject) is respected during
    the shuffle, it provides a valid test of whether the model's predictions
    are significantly associated with the labels beyond chance.

    Parameters
    ----------
    res : dict
        Result dictionary from ExperimentResult.raw.
    metric : str, default='accuracy'
        The metric to evaluate.
    unit : str, optional
        Level of independence (e.g., 'subject').
    n_permutations : int, default=1000
        Number of null permutations.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    posthoc_df : pd.DataFrame
        DataFrame with Observed score, PValue, and Significant status.

    Examples
    --------
    >>> # posthoc = assess_post_hoc_permutation(res.raw['LR'], metric='accuracy')

    See Also
    --------
    run_statistical_assessment : Full-pipeline assessment driver.
    """
    from ._diagnostics import prediction_rows, score_frame

    preds = []
    for fold_idx, p_list in enumerate(res.get("predictions", [])):
        rows = prediction_rows(model="temp", fold_idx=fold_idx, preds=p_list)
        preds.extend([r for r in rows if r.get("Time") is None])

    if not preds:
        raise ValueError("No scalar predictions found for post-hoc assessment.")

    df = pd.DataFrame(preds)
    y_true = df["y_true"].to_numpy()
    obs_score = score_frame(df, metric)

    rng = np.random.default_rng(random_state)
    null_scores = np.zeros(n_permutations)

    u_col = unit if unit != "sample" else None
    if u_col is None and "Group" in df.columns:
        u_col = "Group"

    if u_col is not None and u_col in df.columns:
        unique_units = df[u_col].unique()
        label_map = df.groupby(u_col)["y_true"].apply(list).to_dict()
        for i in range(n_permutations):
            shuffled_units = rng.permutation(unique_units)
            unit_map = dict(zip(unique_units, shuffled_units, strict=False))
            new_labels = []
            for val in df[u_col]:
                target_u = unit_map[val]
                new_labels.append(rng.choice(label_map[target_u]))
            new_df = df.copy()
            new_df["y_true"] = new_labels
            null_scores[i] = score_frame(new_df, metric)
    else:
        for i in range(n_permutations):
            new_labels = rng.permutation(y_true)
            new_df = df.copy()
            new_df["y_true"] = new_labels
            null_scores[i] = score_frame(new_df, metric)

    spec = get_metric_spec(metric)
    if spec.greater_is_better:
        p_val = (np.sum(null_scores >= obs_score) + 1) / (n_permutations + 1)
    else:
        p_val = (np.sum(null_scores <= obs_score) + 1) / (n_permutations + 1)

    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "Observed": obs_score,
                "PValue": float(p_val),
                "Significant": p_val < 0.05,
                "NullMethod": "posthoc_label_permutation",
                "NPermutations": n_permutations,
                "InferentialUnit": unit or "sample",
                "NullLower": float(np.quantile(null_scores, 0.025)),
                "NullUpper": float(np.quantile(null_scores, 0.975)),
            }
        ]
    )


def assess_paired_comparison(
    merged: pd.DataFrame,
    metric: str = "accuracy",
    unit: str | None = None,
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Perform a paired permutation test between two models.

    Tests the null hypothesis that the difference between two models is zero
    by randomly swapping model labels within each independent unit.

    Scientific Rationale
    --------------------
    To compare two models (A and B), we test if the observed difference in
    performance is greater than what would be expected by chance if the labels
    'A' and 'B' were interchangeable. By swapping labels within units (e.g.,
    within subject), we control for subject-specific performance baselines and
    focus on the model-driven variance.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged prediction frame with suffixes '_A' and '_B'.
    metric : str, default='accuracy'
        Metric to evaluate.
    unit : str, optional
        Level of independence (e.g., 'subject').
    n_permutations : int, default=1000
        Number of permutations.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame with ScoreA, ScoreB, Difference, and PValue.

    Examples
    --------
    >>> # comp = assess_paired_comparison(merged_df, metric='accuracy')

    See Also
    --------
    run_paired_permutation_assessment : Full-pipeline paired comparison.
    """

    coord_cols = [c for c in ["Time", "TrainTime", "TestTime"] if c in merged]
    if coord_cols:
        results = []
        for coords, group in merged.groupby(coord_cols):
            res = _assess_paired_comparison_internal(
                group, metric, unit, n_permutations, random_state
            )
            # Add coordinates back
            if len(coord_cols) == 1:
                res[coord_cols[0]] = coords
            else:
                for i, col in enumerate(coord_cols):
                    res[col] = coords[i]
            results.append(res)
        return pd.concat(results, ignore_index=True)

    return _assess_paired_comparison_internal(
        merged, metric, unit, n_permutations, random_state
    )


def _assess_paired_comparison_internal(
    merged: pd.DataFrame,
    metric: str,
    unit: str | None,
    n_permutations: int,
    random_state: int | None,
) -> pd.DataFrame:
    """Internal core for paired comparison on a single coordinate."""
    from ._diagnostics import paired_unit_indices, score_frame

    paired_cols = [col for col in merged.columns if col.endswith(("_A", "_B"))]

    def _prediction_frame(suffix: str) -> pd.DataFrame:
        suffixed_cols = [col for col in merged.columns if col.endswith(suffix)]
        base_cols = [col.removesuffix(suffix) for col in suffixed_cols]
        drop_cols = list(dict.fromkeys(base_cols + paired_cols))
        return pd.concat(
            [
                merged.drop(columns=drop_cols, errors="ignore"),
                merged[suffixed_cols].rename(
                    columns=dict(zip(suffixed_cols, base_cols, strict=True))
                ),
            ],
            axis=1,
        ).copy()

    # Assemble the model-specific columns in one operation. Repeated column
    # assignment fragments wide probability frames and makes every subsequent
    # permutation copy unnecessarily expensive.
    frame_a = _prediction_frame("_A")
    frame_b = _prediction_frame("_B")

    score_a = score_frame(frame_a, metric)
    score_b = score_frame(frame_b, metric)
    obs_diff = score_a - score_b

    u_indices = paired_unit_indices(merged, unit or "sample")
    n_units = len(u_indices)
    rng = np.random.default_rng(random_state)

    # Swap whole units with one row mask per permutation (vectorised; a per-unit
    # iloc loop is O(n_units) pandas writes per permutation).
    unit_of_row = np.full(len(merged), -1)
    for unit_idx, rows in enumerate(u_indices):
        unit_of_row[rows] = unit_idx
    covered = unit_of_row >= 0
    swap_cols = ["y_pred"]
    if "y_score" in frame_a.columns:
        swap_cols.append("y_score")
    swap_cols.extend(c for c in frame_a.columns if c.startswith("y_proba_"))
    values_a = {col: frame_a[col].to_numpy() for col in swap_cols}
    values_b = {col: frame_b[col].to_numpy() for col in swap_cols}

    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swaps = rng.choice([True, False], size=n_units)
        mask = np.zeros(len(merged), dtype=bool)
        mask[covered] = swaps[unit_of_row[covered]]
        perm_a = frame_a.copy()
        perm_b = frame_b.copy()
        for col in swap_cols:
            perm_a[col] = np.where(mask, values_b[col], values_a[col])
            perm_b[col] = np.where(mask, values_a[col], values_b[col])

        null_diffs[i] = score_frame(perm_a, metric) - score_frame(perm_b, metric)

    p_val = (np.sum(np.abs(null_diffs) >= np.abs(obs_diff)) + 1) / (n_permutations + 1)

    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "ScoreA": score_a,
                "ScoreB": score_b,
                "Difference": obs_diff,
                "PValue": float(p_val),
                "Significant": p_val < 0.05,
                "NUnits": n_units,
                "NPermutations": n_permutations,
            }
        ]
    )


def assess_bootstrap_ci(
    res: dict[str, Any],
    metric: str = "accuracy",
    unit: str | None = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Estimate uncertainty of a metric via bootstrapping over units.

    This function computes the observed metric on the provided results
    and then generates a distribution of scores by resampling independent
    units with replacement.

    Scientific Rationale
    --------------------
    Bootstrapping provides a non-parametric estimate of the sampling
    distribution of the metric. By resampling at the 'unit' level (e.g.,
    subjects rather than individual trials), we account for within-unit
    correlations and avoid pseudoreplication, ensuring that the confidence
    intervals accurately reflect the uncertainty at the intended level of
    inference.

    Parameters
    ----------
    res : dict
        Result dictionary for a single model from ExperimentResult.raw.
    metric : str, default='accuracy'
        Metric to evaluate.
    unit : str, optional
        The level of independence (e.g., 'subject').
    n_bootstraps : int, default=1000
        Number of bootstrap iterations.
    ci : float, default=0.95
        Confidence level (0.95 for 95% intervals).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    bootstrap_df : pd.DataFrame
        DataFrame with estimate, CILower, and CIUpper.

    Examples
    --------
    >>> # ci_df = assess_bootstrap_ci(res.raw['LR'], unit='subject')

    See Also
    --------
    binomial_accuracy_test : Analytical CI alternative.
    """
    from ._diagnostics import prediction_rows, score_frame, unit_indices

    preds = []
    for fold_idx, p_list in enumerate(res.get("predictions", [])):
        rows = prediction_rows(model="temp", fold_idx=fold_idx, preds=p_list)
        preds.extend([r for r in rows if r.get("Time") is None])

    if not preds:
        raise ValueError("No scalar predictions found for bootstrap.")

    df = pd.DataFrame(preds)
    obs_score = score_frame(df, metric)

    u_indices = unit_indices(df, unit)
    unit_map = {i: df.iloc[idx] for i, idx in enumerate(u_indices)}
    unique_units = np.arange(len(u_indices))

    def score_func(sample_df: pd.DataFrame) -> np.ndarray:
        try:
            return np.array([score_frame(sample_df, metric)])
        except Exception:
            return np.array([np.nan])

    boot_scores = _bootstrap_engine(
        unique_units, unit_map, score_func, n_bootstraps, random_state
    ).flatten()

    alpha = (1 - ci) / 2

    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "Estimate": obs_score,
                "CILower": float(np.nanquantile(boot_scores, alpha)),
                "CIUpper": float(np.nanquantile(boot_scores, 1 - alpha)),
                "Unit": unit or "sample",
                "NUnits": len(u_indices),
                "NBootstraps": n_bootstraps,
            }
        ]
    )


def apply_multiple_comparison_correction(
    df: pd.DataFrame,
    p_col: str = "PValue",
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply multiple comparison correction to a DataFrame of results.

    Scientific Rationale
    --------------------
    When testing multiple hypotheses (e.g., across many timepoints or models),
    the probability of a Type I error (false positive) increases. This
    utility controls the expected proportion of false discoveries with the
    Benjamini-Hochberg False Discovery Rate (FDR) procedure.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame containing p-values.
    p_col : str, default='PValue'
        Name of the column containing raw p-values.
    method : str, default='fdr_bh'
        ``"fdr_bh"`` (Benjamini-Hochberg) or ``"none"``.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    corrected_df : pd.DataFrame
        The DataFrame with updated 'PValueCorrected' and 'Significant' columns.

    Examples
    --------
    >>> # corrected = apply_multiple_comparison_correction(results_df, method='fdr_bh')
    """
    if method not in ("fdr_bh", "none"):
        raise ValueError(f"Unknown correction {method!r}; use 'fdr_bh' or 'none'.")
    if df.empty or len(df) < 2 or method == "none":
        if "Significant" not in df.columns and not df.empty:
            df["Significant"] = df[p_col] < alpha
        return df

    corrected, reject = benjamini_hochberg(df[p_col], alpha=alpha)

    df = df.copy()
    df["PValueCorrected"] = corrected
    df["Significant"] = reject
    df["CorrectionMethod"] = method
    return df
