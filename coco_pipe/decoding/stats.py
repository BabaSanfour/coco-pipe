"""
Finite-sample statistical assessment for decoding results.

This module separates descriptive performance from inferential claims. The
default inferential path reruns the complete decoding experiment under label
permutations so learned preprocessing, feature selection, tuning, and
calibration remain inside each null pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:
    from .result import ExperimentResult

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, false_discovery_control, norm

from ._metrics import get_metric_spec
from .configs import StatisticalAssessmentConfig

logger = logging.getLogger(__name__)

TEMPORAL_COLUMNS = ["Time", "TrainTime", "TestTime"]


def aggregate_predictions_for_inference(
    predictions: pd.DataFrame,
    metric: str,
    task: str = "classification",
    unit_of_inference: str = "sample",
    custom_unit_column: Optional[str] = None,
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
    >>> df = pd.DataFrame({
    ...     'Subject': ['S1', 'S1'], 'y_true': [1, 1], 'y_pred': [1, 0],
    ...     'SampleID': [0, 1]
    ... })
    >>> res = aggregate_predictions_for_inference(
    ...     df, 'accuracy', unit_of_inference='Subject'
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
            raise ValueError(
                f"Inference unit '{unit_col}' not found in result columns. "
                f"Available: {list(frame.columns)}"
            )
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
    p0: Optional[float],
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
    >>> print(res['p_value'])

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
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
) -> dict[str, Any]:
    """
    Orchestrate the statistical assessment of experiment results.

    Resolves the chosen statistical method (binomial or permutation) and
    dispatches analysis for each model and metric.

    Scientific Rationale
    --------------------
    Statistical significance in decoding is often non-trivial due to temporal
    autocorrelations and multiple comparisons. This orchestrator handles
    either analytical binomial tests (fast, theoretical chance) or full-pipeline
    permutation tests (rigorous, empirical null) to provide scientifically
    grounded inferential claims about model performance.

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
        Summary containing 'rows' (standardized results) and 'nulls'.

    Examples
    --------
    >>> # Internal use within Experiment.run()
    >>> # res = run_statistical_assessment(observed, config, X, y, ...)

    See Also
    --------
    binomial_accuracy_test : Core analytical test.
    assess_post_hoc_permutation : Fast post-hoc alternative.
    """
    stats_config = experiment_config.evaluation
    unit = inferential_unit
    metrics = experiment_config.get_all_evaluation_metrics()
    rows: list[dict[str, Any]] = []
    nulls: dict[str, dict[str, Any]] = {}

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
                    if (metric == "accuracy" and stats_config.chance.p0)
                    else "permutation"
                )

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
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
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
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> np.ndarray:
    """
    Execute the core permutation loop using parallel processing.
    """
    from .experiment import Experiment

    rng = np.random.default_rng(config.random_state)
    if unit == "sample":
        unit_values = np.arange(len(y))
    elif sample_metadata is not None and unit in sample_metadata.columns:
        unit_values = sample_metadata[unit].to_numpy()
    elif groups is not None and unit in {"Group", "group"}:
        unit_values = np.asarray(groups)
    else:
        target_col = config.custom_unit_column if unit == "custom" else unit
        if sample_metadata is not None and target_col in sample_metadata.columns:
            unit_values = sample_metadata[target_col].to_numpy()
        else:
            raise ValueError(f"Could not resolve unit values for '{unit}'.")
    unique_units, unit_map_idx = np.unique(unit_values, return_inverse=True)
    n_unique = len(unique_units)

    unit_labels_orig = []
    for u in unique_units:
        mask = unit_values == u
        u_y = y[mask]
        if experiment_config.task == "classification":
            unit_labels_orig.append(u_y[0])
        else:
            unit_labels_orig.append(np.mean(u_y))
    unit_labels_orig = np.array(unit_labels_orig)

    import joblib

    parallel = joblib.Parallel(n_jobs=experiment_config.n_jobs)

    def _run_single_permutation(p_idx, seed):
        local_rng = np.random.default_rng(seed)
        perm_idx = local_rng.permutation(n_unique)
        y_perm = unit_labels_orig[perm_idx][unit_map_idx]
        perm_config = experiment_config.model_copy()
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
) -> list[dict[str, Any]]:
    """Format a permutation test result into a standardized result row."""
    metric_spec = get_metric_spec(metric)
    greater_is_better = metric_spec.greater_is_better

    p_values = _empirical_p_values(
        observed_array,
        null_array,
        greater_is_better,
    )

    corrected = _correct_p_values(
        observed_array,
        null_array,
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
                "NullMethod": "permutation_full_pipeline",
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
                "Assumptions": (
                    "full outer-CV pipeline rerun under label permutations; "
                    "regression targets averaged by unit"
                    if task == "regression" and unit != "sample"
                    else "full outer-CV pipeline rerun under label permutations"
                ),
                "Caveat": f"Independence assumed at the '{unit}' level.",
                **coord,
            }
        )
    return rows


def run_paired_permutation_assessment(
    results_a: "ExperimentResult",
    results_b: "ExperimentResult",
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

    merge_cols = ["SampleID", "Fold"]
    temporal_cols = [c for c in TEMPORAL_COLUMNS if c in preds_a]
    merge_cols.extend(temporal_cols)

    unit_col = (
        config.unit_of_inference if config.unit_of_inference != "sample" else "SampleID"
    )
    if unit_col in preds_a and unit_col not in merge_cols:
        merge_cols.append(unit_col)

    merged = pd.merge(preds_a, preds_b, on=merge_cols, suffixes=("_A", "_B"))
    if merged.empty:
        raise ValueError("Could not align predictions for paired test.")

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

    observed_diff_array = np.zeros(len(score_keys))
    for idx, key in enumerate(score_keys):
        m = np.ones(len(merged), dtype=bool)
        for i, c in enumerate(temporal_cols):
            m &= merged[c] == key[i]
        observed_diff_array[idx] = get_diff(merged[m])

    boot_results = _bootstrap_scores_paired(
        merged,
        metric=metric,
        score_keys=score_keys,
        temporal_cols=temporal_cols,
        unit_col=unit_col,
        n_bootstraps=1000,
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

        p_diffs = np.empty(len(score_keys))
        for idx, key in enumerate(score_keys):
            m = np.ones(len(perm_merged), dtype=bool)
            for j, c in enumerate(temporal_cols):
                m &= perm_merged[c] == key[j]
            p_diffs[idx] = get_diff(perm_merged[m])
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
        y_true_mat = frame.pivot(
            index="InferentialUnitID", columns=temporal_cols, values="y_true"
        )
        y_pred_mat = frame.pivot(
            index="InferentialUnitID", columns=temporal_cols, values="y_pred"
        )

        if metric == "accuracy":
            scores_array = (y_true_mat.values == y_pred_mat.values).mean(axis=0)
        else:
            scores_array = (y_true_mat.values != y_pred_mat.values).mean(axis=0)

        return dict(zip(y_true_mat.columns, scores_array))

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
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Core engine for unit-based bootstrap resampling.

    Examples
    --------
    >>> # boot_dist = _bootstrap_engine(units, unit_map, score_func, n_bootstraps=100)
    """
    rng = np.random.default_rng(random_state)
    n_units = len(units)
    results = []
    for _ in range(n_bootstraps):
        boot_units = rng.choice(units, size=n_units, replace=True)
        boot_frame = pd.concat([unit_map[u] for u in boot_units])
        results.append(score_func(boot_frame))
    return np.array(results)


def _bootstrap_scores(
    frame: pd.DataFrame,
    metric: str,
    score_keys: list[tuple],
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None,
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
    random_state: Optional[int] = None,
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
        res = np.empty(len(score_keys))
        for idx, key in enumerate(score_keys):
            m = np.ones(len(df), dtype=bool)
            for j, c in enumerate(temporal_cols):
                m &= df[c] == key[j]
            res[idx] = get_diff(df[m])
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

    Supported methods include standard corrections (Bonferroni, FDR) and
    permutation-based Max-Stat (recommended for cluster-based temporal data).

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
    if method == "bonferroni":
        return np.minimum(1.0, p_values * len(p_values))
    if method == "fdr_bh":
        return false_discovery_control(p_values, method="bh")
    if method == "fdr_by":
        return false_discovery_control(p_values, method="by")
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
    unit: Optional[str] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
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
            unit_map = dict(zip(unique_units, shuffled_units))
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
    unit: Optional[str] = None,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
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
    unit: Optional[str],
    n_permutations: int,
    random_state: Optional[int],
) -> pd.DataFrame:
    """Internal core for paired comparison on a single coordinate."""
    from ._diagnostics import paired_unit_indices, score_frame

    frame_a = merged.copy()
    for col in merged.columns:
        if col.endswith("_A"):
            frame_a[col[:-2]] = merged[col]

    frame_b = merged.copy()
    for col in merged.columns:
        if col.endswith("_B"):
            frame_b[col[:-2]] = merged[col]

    score_a = score_frame(frame_a, metric)
    score_b = score_frame(frame_b, metric)
    obs_diff = score_a - score_b

    u_indices = paired_unit_indices(merged, unit or "sample")
    n_units = len(u_indices)
    rng = np.random.default_rng(random_state)

    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swaps = rng.choice([True, False], size=n_units)
        perm_a = frame_a.copy()
        perm_b = frame_b.copy()

        for unit_idx, should_swap in enumerate(swaps):
            if should_swap:
                idx = u_indices[unit_idx]
                swap_cols = ["y_pred"]
                if "y_score" in frame_a.columns:
                    swap_cols.append("y_score")
                swap_cols.extend(
                    [c for c in frame_a.columns if c.startswith("y_proba_")]
                )
                for col in swap_cols:
                    temp = perm_a.iloc[idx, perm_a.columns.get_loc(col)].copy()
                    perm_a.iloc[idx, perm_a.columns.get_loc(col)] = perm_b.iloc[
                        idx, perm_b.columns.get_loc(col)
                    ]
                    perm_b.iloc[idx, perm_b.columns.get_loc(col)] = temp

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
    unit: Optional[str] = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
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
    utility applies standard corrections like Bonferroni (strict) or False
    Discovery Rate (FDR; Benjamini-Hochberg) to control the family-wise error
    rate or the expected proportion of false discoveries.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame containing p-values.
    p_col : str, default='PValue'
        Name of the column containing raw p-values.
    method : str, default='fdr_bh'
        Correction method (e.g., 'fdr_bh', 'bonferroni').
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    corrected_df : pd.DataFrame
        The DataFrame with updated 'PValueCorrected' and 'Significant' columns.

    Examples
    --------
    >>> # corrected = apply_multiple_comparison_correction(results_df, method='fdr_bh')

    See Also
    --------
    statsmodels.stats.multitest.multipletests : Underlying implementation.
    """
    from statsmodels.stats.multitest import multipletests

    if df.empty or len(df) < 2 or not method:
        if "Significant" not in df.columns and not df.empty:
            df["Significant"] = df[p_col] < alpha
        return df

    reject, corrected, _, _ = multipletests(
        df[p_col].to_numpy(), alpha=alpha, method=method
    )

    df = df.copy()
    df["PValueCorrected"] = corrected
    df["Significant"] = reject
    df["CorrectionMethod"] = method
    return df
