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


def _pooled_separation(
    trajectories: np.ndarray, in_a: np.ndarray, in_b: np.ndarray, method: str
) -> np.ndarray:
    """Separation curve between the trials ``in_a`` and the trials ``in_b``."""
    selected = in_a | in_b
    binary = np.where(in_a[selected], 0, 1)
    return trajectory_separation(trajectories[selected], binary, method=method)[(0, 1)]


def _separation_auc(
    result: TrajectoryResult,
    codes: np.ndarray,
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    method: str,
    mask: np.ndarray,
    per_subject: bool,
) -> float:
    """Separation AUC between the trials coded ``codes_a`` and ``codes_b``."""
    in_a = np.isin(codes, codes_a)
    in_b = np.isin(codes, codes_b)
    if per_subject:
        curves = []
        for subject in np.unique(result.subjects):
            rows = result.subjects == subject
            if not (in_a[rows].any() and in_b[rows].any()):
                raise ValueError(
                    f"Subject {subject!r} has no trial in one group of the contrast; "
                    "per_subject=True needs both groups in every subject."
                )
            curves.append(
                _pooled_separation(
                    result.trajectories[rows], in_a[rows], in_b[rows], method
                )
            )
        curve = np.mean(curves, axis=0)
    else:
        curve = _pooled_separation(result.trajectories, in_a, in_b, method)
    return float(np.trapezoid(curve[mask], result.times[mask]))


def _check_strata(
    result: TrajectoryResult, strata: Any, per_subject: bool
) -> np.ndarray:
    """Return ``strata`` as one value per trajectory, nested in subjects if asked."""
    strata = np.asarray(strata)
    if strata.shape != result.subjects.shape:
        raise ValueError(
            f"`strata` must hold one value per trajectory "
            f"{result.subjects.shape}; got shape {strata.shape}."
        )
    if per_subject:
        subjects_per_stratum = pd.Series(result.subjects).groupby(strata).nunique()
        if subjects_per_stratum.max() > 1:
            raise ValueError(
                "per_subject=True needs strata nested in subjects, so every "
                "permutation keeps each subject's trials per group."
            )
    return strata


def _check_finite(observed: float, null: np.ndarray, what: str) -> None:
    """Raise when a separation AUC is NaN, which leaves the p-value undefined."""
    if np.isnan(observed) or np.isnan(null).any():
        raise ValueError(
            f"The separation AUC of {what} is NaN: its trajectories have missing "
            "time points inside the window, so the p-value is undefined."
        )


def permutation_null_separation_auc(
    result: TrajectoryResult,
    group_a: Sequence,
    group_b: Sequence,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
    method: str = "centroid",
    window: tuple[float, float] | None = None,
    strata: Sequence | None = None,
    per_subject: bool = False,
) -> tuple[float, np.ndarray, float]:
    """Label-permutation test of between-group separation AUC.

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
    strata : sequence, optional
        One stratum per trajectory; labels are permuted only inside a stratum
        (:func:`~coco_pipe.utils.permute_labels_within_strata`). Choose them so
        that only the tested factor moves: ``result.subjects`` for a
        within-subject test, or subject x class x side to test speed among
        speed x class x side conditions. By default labels are shuffled across
        all trajectories.
    per_subject : bool, default=False
        Average the separation curves of the subjects instead of separating
        the pooled trials. Needs strata nested in subjects.

    Returns
    -------
    observed_auc : float
    null_aucs : np.ndarray of shape (n_perm,)
        AUC values under ``n_perm`` random label permutations.
    p_value : float
        ``(1 + #(null_aucs >= observed_auc)) / (n_perm + 1)``.

    Raises
    ------
    ValueError
        If a label in ``group_a`` or ``group_b`` is not a condition of
        ``result``, ``strata`` does not fit, or an AUC is NaN.

    See Also
    --------
    permutation_familywise_separation_auc : Max-statistic correction over
        several results and contrasts.

    Notes
    -----
    By default the statistic is the separation of the pooled trials, for
    example the distance between the centroids of all group-a and all group-b
    trials. With ``per_subject=True`` it is the mean of the subjects' own
    separation curves, the subject as the unit; with few trials per subject
    that mean is biased upward by noise. Both are valid under permutation but
    measure different things.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if per_subject and strata is None:
        raise ValueError("per_subject=True needs strata nested in subjects.")

    codes_a = result._condition_codes_of(group_a)
    codes_b = result._condition_codes_of(group_b)
    _, codes = result._condition_coding()
    if strata is not None:
        strata = _check_strata(result, strata, per_subject)
    mask = _window_mask(result.times, window)

    def _auc(labels: np.ndarray) -> float:
        return _separation_auc(
            result, labels, codes_a, codes_b, method, mask, per_subject
        )

    observed = _auc(codes)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = (
            rng.permutation(codes)
            if strata is None
            else permute_labels_within_strata(codes, strata, random_state=rng)
        )
        null[i] = _auc(shuffled)
    _check_finite(observed, null, "the contrast")
    return observed, null, float((1 + np.sum(null >= observed)) / (n_perm + 1))


def permutation_familywise_separation_auc(
    results: Mapping[Hashable, TrajectoryResult],
    contrasts: Sequence[tuple[Any, Any]],
    strata: Any,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
    method: str = "centroid",
    window: tuple[float, float] | None = None,
    per_subject: bool = False,
    standardize: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Family-wise max-statistic test of separation AUC over results and contrasts.

    Each permutation relabels the trials inside ``strata`` and applies the
    same relabelling to every result. The null of the family is the maximum
    statistic over all results and contrasts per permutation, and each
    observed statistic gets a p-value corrected for the family-wise error
    rate. The trajectories are used as given: a reducer that produced them is
    not refit.

    Parameters
    ----------
    results : mapping of key to TrajectoryResult
        The family, such as one result per frequency band. Every result holds
        the same subjects and conditions in the same row order.
    contrasts : sequence of (group_a, group_b)
        Pairs of condition groups, each a label or a list of labels as stored
        in ``conditions``, such as ``(["famous", "unfamiliar"], "scrambled")``.
    strata : array-like of shape (n_trials,) or (n_contrasts, n_trials)
        Labels are permuted only inside a stratum. Choose them so that only
        the tested factor moves: subject x class x side to test speed among
        speed x class x side conditions. One row is shared by every contrast,
        and each permutation draws one relabelling for all of them. One row
        per contrast lets contrasts of different factors use their own strata;
        each permutation then draws one relabelling per contrast, which still
        controls the family-wise error, more conservatively.
    n_perm : int, default=1000
        Number of permutations.
    rng : np.random.Generator, optional
        Generator for the relabellings. Defaults to ``default_rng(0)``.
    method : str, default="centroid"
        Separation method passed to :func:`trajectory_separation`.
    window : tuple of float, optional
        Time window (tmin, tmax) of the AUC.
    per_subject : bool, default=False
        Average the separation curves of the subjects instead of separating
        the pooled trials. Needs strata nested in subjects.
    standardize : bool, default=True
        Take the maximum over ``z = (AUC - null mean) / null sd``, each
        (result, contrast) standardised by its own null, so results on
        different scales, such as one PCA per band, weigh equally. With
        ``False`` the maximum is over raw AUCs, which suits results on one
        scale.

    Returns
    -------
    table : pd.DataFrame
        One row per (key, contrast): ``key``, ``contrast`` (``"A_vs_B"``, the
        labels of a group joined by ``"+"``), ``observed_auc``, ``null_mean``,
        ``null_sd``, ``z`` and ``p_fwer``,
        ``(1 + #(max_null >= statistic)) / (n_perm + 1)`` with the statistic
        ``z`` or ``observed_auc``.
    max_null : np.ndarray of shape (n_perm,)
        Maximum statistic over the family under each permutation.
    null_aucs : np.ndarray of shape (n_perm, n_results, n_contrasts)
        AUC of every result and contrast under each permutation.

    Raises
    ------
    ValueError
        If ``results`` or ``contrasts`` is empty, the results do not hold the
        same subjects and conditions in the same order, ``strata`` does not
        fit, a contrast names an unknown condition, an AUC is NaN, or a null
        is constant when ``standardize`` is true.

    See Also
    --------
    permutation_null_separation_auc : One result and one contrast.

    Notes
    -----
    The statistic is that of :func:`permutation_null_separation_auc`: the
    separation of the pooled trials by default, the mean of the subjects'
    separation curves with ``per_subject=True``.
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

    strata_rows = np.asarray(strata)
    shared = strata_rows.ndim == 1
    if not shared and strata_rows.shape[0] != len(contrasts):
        raise ValueError(
            f"`strata` must have one row, or one row per contrast "
            f"({len(contrasts)}); got shape {strata_rows.shape}."
        )
    strata_per_contrast = [
        _check_strata(reference, row, per_subject)
        for row in (strata_rows[None] if shared else strata_rows)
    ]

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

    def _family(labels_per_contrast: list[np.ndarray]) -> np.ndarray:
        return np.array(
            [
                [
                    _separation_auc(
                        member, labels, a, b, method, masks[key], per_subject
                    )
                    for (_, a, b), labels in zip(
                        coded, labels_per_contrast, strict=True
                    )
                ]
                for key, member in members
            ]
        )

    def _draw() -> list[np.ndarray]:
        draws = [
            permute_labels_within_strata(codes, row, random_state=rng)
            for row in strata_per_contrast
        ]
        return draws * len(coded) if shared else draws

    observed = _family([codes] * len(coded))
    null_aucs = np.array([_family(_draw()) for _ in range(n_perm)])
    for m, (key, _) in enumerate(members):
        for c, (name, _, _) in enumerate(coded):
            _check_finite(observed[m, c], null_aucs[:, m, c], f"{key!r}, {name!r}")

    null_mean = null_aucs.mean(axis=0)
    null_sd = null_aucs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - null_mean) / null_sd
    if standardize:
        constant = np.argwhere(null_sd == 0)
        if constant.size:
            m, c = constant[0]
            raise ValueError(
                f"The null of {members[m][0]!r}, {coded[c][0]!r} is constant, so it "
                "cannot be standardised."
            )
        statistic = z
        max_null = ((null_aucs - null_mean) / null_sd).max(axis=(1, 2))
    else:
        statistic = observed
        max_null = null_aucs.max(axis=(1, 2))
    table = pd.DataFrame(
        [
            {
                "key": key,
                "contrast": name,
                "observed_auc": observed[m, c],
                "null_mean": null_mean[m, c],
                "null_sd": null_sd[m, c],
                "z": z[m, c],
                "p_fwer": (1 + np.sum(max_null >= statistic[m, c])) / (n_perm + 1),
            }
            for m, (key, _) in enumerate(members)
            for c, (name, _, _) in enumerate(coded)
        ]
    )
    return table, max_null, null_aucs


def _contrast_name(group_a: Any, group_b: Any) -> str:
    """Name a contrast ``"A_vs_B"``, joining the labels of a group by ``"+"``."""

    def _join(group: Any) -> str:
        return "+".join(str(label) for label in np.atleast_1d(np.asarray(group)))

    return f"{_join(group_a)}_vs_{_join(group_b)}"
