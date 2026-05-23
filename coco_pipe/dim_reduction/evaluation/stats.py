"""
Statistical tests for trajectory scalar metrics.

Functions
---------
paired_condition_stats
    Paired t-test across all condition pairs, with FDR correction.
grouped_condition_stats
    Paired t-test across grouped condition sets, with FDR correction.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

__all__ = [
    "paired_condition_stats",
    "grouped_condition_stats",
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
    _, p_fdr, _, _ = multipletests(out["p_uncorrected"].fillna(1.0), method="fdr_bh")
    out["p_fdr"] = p_fdr
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
    _, p_fdr, _, _ = multipletests(out["p_uncorrected"].fillna(1.0), method="fdr_bh")
    out["p_fdr"] = p_fdr
    return out
