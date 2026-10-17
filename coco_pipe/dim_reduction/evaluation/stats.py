"""
Statistical tests for trajectory scalar metrics.

Functions
---------
paired_condition_stats
    Paired t-test across all condition pairs, with FDR correction.
grouped_condition_stats
    Paired t-test across grouped condition sets, with FDR correction.
permutation_null_separation_auc
    Label-permutation test of the separation AUC of one contrast.
permutation_familywise_separation_auc
    Max-statistic permutation test over several results and contrasts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import TrajectoryResult

from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_rel

from coco_pipe.utils import permute_labels_within_strata

from .geometry import trajectory_separation

__all__ = [
    "grouped_condition_stats",
    "paired_condition_stats",
    "permutation_familywise_separation_auc",
    "permutation_null_separation_auc",
]


def paired_condition_stats(
    scalar_df: pd.DataFrame,
    conditions: list[str],
    metric_col: str = "metric",
    condition_col: str = "condition",
    subject_col: str = "subject",
    value_col: str = "value",
    min_pairs: int = 3,
) -> pd.DataFrame:
    """
    Run a paired t-test for every (metric, condition pair) combination.

    Parameters
    ----------
    scalar_df : pd.DataFrame
        Long-format table of scalar trajectory metrics.  Must contain at
        least ``metric_col``, ``condition_col``, ``subject_col``, and
        ``value_col``.
    conditions : list[str]
        Ordered list of condition labels to include in the comparisons.
    metric_col : str, default="metric"
    condition_col : str, default="condition"
    subject_col : str, default="subject"
    value_col : str, default="value"
    min_pairs : int, default=3
        Minimum number of matched subject pairs required to run a t-test.

    Returns
    -------
    pd.DataFrame
        Columns: ``metric``, ``comparison``, ``n``, ``mean_diff``, ``t``,
        ``p_uncorrected``, ``p_fdr``.
        Empty DataFrame if no valid pairs are found.
    """
    rows = []

    for metric, df_metric in scalar_df.groupby(metric_col):
        wide = df_metric.pivot_table(
            index=subject_col,
            columns=condition_col,
            values=value_col,
            aggfunc="mean",
        )

        for c1, c2 in combinations(conditions, 2):
            if c1 not in wide.columns or c2 not in wide.columns:
                continue
            pair = wide[[c1, c2]].dropna()
            if len(pair) < min_pairs:
                continue

            stat, p = ttest_rel(pair[c1], pair[c2], nan_policy="omit")
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{c1} vs {c2}",
                    "n": len(pair),
                    "mean_diff": float(pair[c1].mean() - pair[c2].mean()),
                    "t": float(stat),
                    "p_uncorrected": float(p),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric",
                "comparison",
                "n",
                "mean_diff",
                "t",
                "p_uncorrected",
                "p_fdr",
            ]
        )

    out = pd.DataFrame(rows)
    out["p_fdr"] = false_discovery_control(out["p_uncorrected"].fillna(1.0))
    return out


def grouped_condition_stats(
    scalar_df: pd.DataFrame,
    condition_sets: dict[str, dict[str, list[str]]],
    metric_col: str = "metric",
    condition_col: str = "condition",
    subject_col: str = "subject",
    value_col: str = "value",
    min_pairs: int = 3,
) -> pd.DataFrame:
    """
    Run paired t-tests across grouped condition sets with FDR correction.

    Conditions within each group are averaged per subject before testing,
    allowing multi-condition groups (e.g. "Positive = [happy, excited]").

    Parameters
    ----------
    scalar_df : pd.DataFrame
        Long-format table of scalar trajectory metrics.
    condition_sets : dict[str, dict[str, list[str]]]
        Mapping ``{set_name: {group_label: [condition, ...]}}`` defining how
        conditions are grouped for each comparison set.
    metric_col, condition_col, subject_col, value_col : str
        Column name overrides.
    min_pairs : int, default=3
        Minimum number of matched subject pairs required to run a t-test.

    Returns
    -------
    pd.DataFrame
        Columns: ``set``, ``metric``, ``comparison``, ``n``, ``mean_diff``,
        ``t``, ``p_uncorrected``, ``p_fdr``.
        Empty DataFrame if no valid pairs are found.
    """
    rows = []

    for set_name, groups in condition_sets.items():
        for metric, df_metric in scalar_df.groupby(metric_col):
            group_frames = []
            for group_label, conds in groups.items():
                tmp = (
                    df_metric[df_metric[condition_col].isin(conds)]
                    .groupby(subject_col, as_index=False)[value_col]
                    .mean()
                    .rename(columns={value_col: "value"})
                )
                tmp["group"] = group_label
                group_frames.append(tmp)

            if not group_frames:
                continue

            long = pd.concat(group_frames, ignore_index=True)
            wide = long.pivot_table(
                index=subject_col, columns="group", values="value", aggfunc="mean"
            )

            for g1, g2 in combinations(groups.keys(), 2):
                if g1 not in wide.columns or g2 not in wide.columns:
                    continue
                pair = wide[[g1, g2]].dropna()
                if len(pair) < min_pairs:
                    continue

                stat, p = ttest_rel(pair[g1], pair[g2], nan_policy="omit")
                rows.append(
                    {
                        "set": set_name,
                        "metric": metric,
                        "comparison": f"{g1} vs {g2}",
                        "n": len(pair),
                        "mean_diff": float(pair[g1].mean() - pair[g2].mean()),
                        "t": float(stat),
                        "p_uncorrected": float(p),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "set",
                "metric",
                "comparison",
                "n",
                "mean_diff",
                "t",
                "p_uncorrected",
                "p_fdr",
            ]
        )

    out = pd.DataFrame(rows)
    out["p_fdr"] = false_discovery_control(out["p_uncorrected"].fillna(1.0))
    return out


def _window_mask(times: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    """Return the time points inside ``window`` (all of them without one)."""
    if window is None:
        return np.ones_like(times, dtype=bool)
    return (times >= window[0]) & (times <= window[1])


def _separation_auc(
    result: TrajectoryResult,
    codes: np.ndarray,
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    method: str,
    mask: np.ndarray,
) -> float:
    """Separation AUC between the trials coded ``codes_a`` and ``codes_b``."""
    in_a = np.isin(codes, codes_a)
    selected = in_a | np.isin(codes, codes_b)
    binary = np.where(in_a[selected], 0, 1)
    curve = trajectory_separation(result.trajectories[selected], binary, method=method)
    return float(np.trapezoid(curve[(0, 1)][mask], result.times[mask]))


def _permutation_p_value(observed: float, null: np.ndarray) -> float:
    """``(1 + #(null >= observed)) / (n_perm + 1)``; NaN when any input is NaN."""
    if np.isnan(observed) or np.isnan(null).any():
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


def permutation_null_separation_auc(
    result: TrajectoryResult,
    group_a: Sequence,
    group_b: Sequence,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
    method: str = "centroid",
    window: tuple[float, float] | None = None,
    within_subject: bool = False,
) -> tuple[float, np.ndarray, float]:
    """Label-shuffle null on between-group separation AUC.

    Parameters
    ----------
    result : TrajectoryResult
        The TrajectoryResult container holding trajectories and metadata.
    group_a, group_b : Sequence
        The condition labels to compare, as stored in ``result.conditions``
        (names or integers).
    n_perm : int, default=200
    rng : np.random.Generator, optional
    method : str, default="centroid"
        Separation method passed to :func:`trajectory_separation`. Any method
        that function supports may be used, e.g. ``"mahalanobis"``.
    window : tuple of float, optional
        Time window (tmin, tmax) to restrict the AUC calculation.
    within_subject : bool, default=False
        When True, permute labels within each subject
        (:func:`~coco_pipe.utils.permute_labels_within_strata`), so each
        subject keeps its trial counts per condition. The default shuffles
        labels globally.

    Returns
    -------
    observed_auc : float
    null_aucs : np.ndarray of shape (n_perm,)
        AUC values under ``n_perm`` random label shuffles.
    p_value : float
        ``(1 + #(null_aucs >= observed_auc)) / (n_perm + 1)``, NaN when the
        observed or a null AUC is NaN.

    Raises
    ------
    ValueError
        If a label in ``group_a`` or ``group_b`` is not a condition of
        ``result``.

    See Also
    --------
    permutation_familywise_separation_auc : Max-statistic correction over
        several results and contrasts.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    codes_a = result._condition_codes_of(group_a)
    codes_b = result._condition_codes_of(group_b)
    _, codes = result._condition_coding()
    mask = _window_mask(result.times, window)

    observed = _separation_auc(result, codes, codes_a, codes_b, method, mask)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = (
            permute_labels_within_strata(codes, result.subjects, random_state=rng)
            if within_subject
            else rng.permutation(codes)
        )
        null[i] = _separation_auc(result, shuffled, codes_a, codes_b, method, mask)
    return observed, null, _permutation_p_value(observed, null)


def permutation_familywise_separation_auc(
    results: Mapping[Hashable, TrajectoryResult],
    contrasts: Sequence[tuple[Any, Any]],
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
    method: str = "centroid",
    window: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Family-wise max-statistic test of separation AUC over results and contrasts.

    Each permutation draws one within-subject relabelling and applies it to
    every result and every contrast; the null of the family is the maximum
    separation AUC over all of them per permutation. Each observed AUC gets a
    p-value corrected for the family-wise error rate. The trajectories are
    used as given: a reducer that produced them is not refit.

    Parameters
    ----------
    results : mapping of key to TrajectoryResult
        The family, such as one result per frequency band. Every result holds
        the same subjects and conditions in the same row order.
    contrasts : sequence of (group_a, group_b)
        Pairs of condition groups, each a label or a list of labels as stored
        in ``conditions``, such as ``(["famous", "unfamiliar"], "scrambled")``.
    n_perm : int, default=1000
        Number of permutations.
    rng : np.random.Generator, optional
        Generator for the relabellings. Defaults to ``default_rng(0)``.
    method : str, default="centroid"
        Separation method passed to :func:`trajectory_separation`.
    window : tuple of float, optional
        Time window (tmin, tmax) of the AUC.

    Returns
    -------
    table : pd.DataFrame
        One row per (key, contrast): ``key``, ``contrast`` (``"A_vs_B"``, the
        labels of a group joined by ``"+"``), ``observed_auc`` and ``p_fwer``,
        ``(1 + #(max_null >= observed_auc)) / (n_perm + 1)``.
    max_null : np.ndarray of shape (n_perm,)
        Maximum AUC over the family under each permutation.

    Raises
    ------
    ValueError
        If ``results`` or ``contrasts`` is empty, the results do not hold the
        same subjects and conditions in the same order, or a contrast names an
        unknown condition.

    See Also
    --------
    permutation_null_separation_auc : One result and one contrast.

    Notes
    -----
    The maximum is taken over raw AUCs, so members should be on comparable
    scales (for example embeddings of one fitted reducer); otherwise the
    member with the largest scale dominates the null.
    """
    if not results or not contrasts:
        raise ValueError("`results` and `contrasts` must each hold at least one entry.")
    if rng is None:
        rng = np.random.default_rng(0)

    members = list(results.items())
    reference = members[0][1]
    for key, member in members[1:]:
        if not (
            np.array_equal(member.subjects, reference.subjects)
            and np.array_equal(member.conditions, reference.conditions)
        ):
            raise ValueError(
                f"Result {key!r} does not hold the subjects and conditions of "
                f"{members[0][0]!r} in the same order."
            )

    _, codes = reference._condition_coding()
    coded = [
        (
            _contrast_name(group_a, group_b),
            reference._condition_codes_of(group_a),
            reference._condition_codes_of(group_b),
        )
        for group_a, group_b in contrasts
    ]
    masks = {key: _window_mask(member.times, window) for key, member in members}

    def _family(labels: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    _separation_auc(
                        member, labels, codes_a, codes_b, method, masks[key]
                    )
                    for _, codes_a, codes_b in coded
                ]
                for key, member in members
            ]
        )

    observed = _family(codes)
    max_null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = permute_labels_within_strata(
            codes, reference.subjects, random_state=rng
        )
        max_null[i] = np.max(_family(shuffled))

    table = pd.DataFrame(
        [
            {
                "key": key,
                "contrast": name,
                "observed_auc": observed[m, c],
                "p_fwer": _permutation_p_value(observed[m, c], max_null),
            }
            for m, (key, _) in enumerate(members)
            for c, (name, _, _) in enumerate(coded)
        ]
    )
    return table, max_null


def _contrast_name(group_a: Any, group_b: Any) -> str:
    """Name a contrast ``"A_vs_B"``, joining the labels of a group by ``"+"``."""

    def _join(group: Any) -> str:
        return "+".join(str(label) for label in np.atleast_1d(np.asarray(group)))

    return f"{_join(group_a)}_vs_{_join(group_b)}"
