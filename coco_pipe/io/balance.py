#!/usr/bin/env python3
"""
coco_pipe/io/balance.py
-----------------------
Utilities to rebalance imbalanced target classes with optional covariate-aware sampling.

Two main modes:
- Random class balancing via under/over-sampling.
- Covariate-aware balancing that preserves distributions across specified covariates
  (e.g., Age and Sex) by stratified sampling within covariate-defined strata.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)
from .utils import row_quality_score, select_cleanest_rows


def _make_strata(
    df: pd.DataFrame,
    covariates: List[str],
    n_bins: int = 5,
    binning: str = "quantile",
) -> pd.Series:
    """Build a categorical stratum label per row from covariates.

    Numeric covariates are binned; categorical/boolean are used as-is.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    covariates : list of str
        Covariate column names.
    n_bins : int
        Number of bins for numeric covariates (quantile or uniform depending on `binning`).
    binning : {"quantile", "uniform"}
        Binning strategy for numeric covariates.

    Returns
    -------
    pd.Series
        A categorical series where each value is a stratum identifier tuple encoded as a string.
    """
    labels: List[pd.Series] = []
    for cov in covariates:
        s = df[cov]
        if is_numeric_dtype(s) or is_datetime64_any_dtype(s):
            # Bin numeric/datetime
            if binning == "uniform":
                b = pd.cut(s, bins=n_bins)
            else:
                # quantile binning; drop duplicate edges for constant cols
                try:
                    b = pd.qcut(s, q=n_bins, duplicates="drop")
                except Exception:
                    # fallback to uniform if qcut fails
                    b = pd.cut(s, bins=n_bins)
            labels.append(b.astype(str).fillna("NA"))
        else:
            # Treat object/category/bool as categories
            labels.append(s.astype(str).fillna("NA"))

    if len(labels) == 1:
        strata = labels[0]
    else:
        # Concatenate into a single label
        df_tmp = pd.concat(labels, axis=1)
        strata = df_tmp.astype(str).agg("|".join, axis=1)
    return strata.astype("category")


def _sample_per_class(
    df: pd.DataFrame,
    target: str,
    size_per_class: Dict[Any, int],
    random_state: Optional[int],
    replace: bool,
    prefer_clean: bool = False,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)
    for cls, n in size_per_class.items():
        sub = df[df[target] == cls]
        if n <= 0:
            continue
        if prefer_clean:
            # Compute row quality for this subset
            q = row_quality_score(sub, exclude_cols=exclude_cols)
            if not replace:
                # Deterministically take the top-n cleanest rows
                # To avoid ties bias, break ties with random shuffle first
                sub_shuf = sub.sample(frac=1.0, random_state=rng.integers(0, 1 << 32))
                idx_top = q.loc[sub_shuf.index].sort_values(kind='mergesort').index[:n]
                parts.append(sub_shuf.loc[idx_top])
            else:
                # Weighted sampling favoring cleaner rows: weight = 1/(1+score)
                w = (1.0 / (1.0 + q)).astype(float)
                parts.append(
                    sub.sample(
                        n=n,
                        replace=True,
                        weights=w,
                        random_state=rng.integers(0, 1 << 32),
                    )
                )
        else:
            if n <= len(sub) and not replace:
                parts.append(sub.sample(n=n, replace=False, random_state=rng.integers(0, 1 << 32)))
            else:
                # oversample with replacement if needed
                parts.append(sub.sample(n=n, replace=True, random_state=rng.integers(0, 1 << 32)))
    return pd.concat(parts).sample(frac=1.0, random_state=random_state)  # shuffle


def balance_dataset(
    df: pd.DataFrame,
    target: str,
    strategy: str = "undersample",
    covariates: Optional[List[str]] = None,
    n_bins: int = 5,
    binning: str = "quantile",
    random_state: Optional[int] = 42,
    prefer_clean_rows: bool = False,
    grid_balance: Optional[List[str]] = None,
    require_full_grid: bool = False,
) -> pd.DataFrame:
    """
    Balance imbalanced classes by random sampling, with optional covariate-aware
    stratification and grid-wise equalization across specified covariate levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and a target column.
    target : str
        Name of the target column with class labels.
    strategy : {"undersample", "oversample", "auto"}
        Sampling strategy. "undersample" reduces all classes to the minority count.
        "oversample" increases all classes to the majority count (with replacement).
        "auto" picks undersample if it keeps >= 50% of data, else oversample.
    covariates : list of str, optional
        If provided, balance within strata defined by these covariates (e.g., ["age", "sex"]).
    n_bins : int, default=5
        Number of bins for numeric covariates when creating strata.
    binning : {"quantile", "uniform"}, default="quantile"
        Binning method for numeric covariates.
    random_state : int or None, default=42
        Seed for reproducibility.
    prefer_clean_rows : bool, default=False
        When undersampling/oversampling, prefer rows with fewer NaN/inf/zero values
        (based on a simple per-row quality score). Uses columns other than `target`
        and `covariates` for the score.
    grid_balance : list of str, optional
        Subset of `covariates` across which to enforce equal counts jointly with
        the target, within the remaining covariates. For example, with
        covariates=["age", "sex"] and grid_balance=["sex"], the function will:
        - Bin "age" into strata; and for each age bin, enforce equal counts across
          all combinations of (sex x class). This guarantees that per age bin,
          male/female are balanced within each class and across classes.
        If None (default), the function balances classes within each full
        covariate stratum (original behavior).
    require_full_grid : bool, default=False
        If True and `grid_balance` is provided, only strata where every
        (grid_balance x target) combination exists are kept. If False, missing
        combinations are ignored (they cannot be synthesized).

    Returns
    -------
    pd.DataFrame
        A rebalanced DataFrame.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    strategy = strategy.lower()
    if strategy not in {"undersample", "oversample", "auto"}:
        raise ValueError("strategy must be one of {'undersample','oversample','auto'}")

    cls_counts = df[target].value_counts()
    min_count, max_count = int(cls_counts.min()), int(cls_counts.max())

    if strategy == "auto":
        # If undersampling keeps at least half the data, use it; else oversample
        keep_if_under = min_count * cls_counts.size
        strategy = "undersample" if keep_if_under >= len(df) / 2 else "oversample"

    exclude_cols: List[str] = [target]
    if covariates:
        exclude_cols.extend(covariates)

    if not covariates:
        # Global random sampling by class
        if strategy == "undersample":
            size = {cls: min_count for cls in cls_counts.index}
            return _sample_per_class(
                df,
                target,
                size,
                random_state,
                replace=False,
                prefer_clean=prefer_clean_rows,
                exclude_cols=exclude_cols,
            )
        else:
            size = {cls: max_count for cls in cls_counts.index}
            return _sample_per_class(
                df,
                target,
                size,
                random_state,
                replace=True,
                prefer_clean=prefer_clean_rows,
                exclude_cols=exclude_cols,
            )

    # Covariate-aware balancing: stratify by covariates then equalize
    for cov in covariates:
        if cov not in df.columns:
            raise ValueError(f"Covariate '{cov}' not found in DataFrame")

    strata = _make_strata(df, covariates=covariates, n_bins=n_bins, binning=binning)
    tmp = df.copy()
    tmp["__strata__"] = strata

    # If `grid_balance` is requested, we equalize across the cross-product of
    # (grid_balance variables x target) WITHIN each base stratum defined by the
    # remaining covariates. This satisfies constraints like: within each age bin,
    # Male/Female are equal within and across classes.
    balanced_parts: List[pd.DataFrame] = []
    if grid_balance:
        # Sanity: ensure grid_balance subset of covariates
        missing = [g for g in grid_balance if g not in covariates]
        if missing:
            raise ValueError(
                f"grid_balance contains columns not in covariates: {missing}"
            )

        base_cov = [c for c in covariates if c not in grid_balance]
        # Build base strata; if empty, use a single global stratum
        if base_cov:
            base_strata = _make_strata(df, covariates=base_cov, n_bins=n_bins, binning=binning)
        else:
            base_strata = pd.Series(["__ALL__"] * len(df), index=df.index, dtype="category")

        tmp2 = df.copy()
        tmp2["__base__"] = base_strata
        # Build label columns for each grid variable (bin numeric, pass-through categoricals)
        grid_label_cols: List[str] = []
        for gcol in grid_balance:
            glabel = _make_strata(df, covariates=[gcol], n_bins=n_bins, binning=binning)
            colname = f"__grid__{gcol}__"
            tmp2[colname] = glabel
            grid_label_cols.append(colname)

        # Global class list and target per-class cap (min class size)
        classes_all = list(cls_counts.index)
        T = min_count

        # Desired allocation per (base, grid, class)
        desired: Dict[Tuple[str, str, Any], int] = {}
        available: Dict[Tuple[str, str, Any], int] = {}
        good_pairs: List[Tuple[str, str]] = []  # (base, grid)
        grids_per_base: Dict[str, List[str]] = {}

        # First pass: compute equalizable core m_b per base over grids that have all classes
        for base_val, g in tmp2.groupby("__base__"):
            # Build grid key for this base subset
            if grid_label_cols:
                grid_key = g[grid_label_cols].astype(str).agg("|".join, axis=1)
            else:
                grid_key = pd.Series(["__ALL__"] * len(g), index=g.index)
            g = g.assign(__gridkey__=grid_key)

            # counts per (grid, class)
            counts = g.groupby(["__gridkey__", target]).size().unstack(fill_value=0)
            if counts.empty:
                continue

            # Determine grids to use for equal core
            if require_full_grid:
                # grids where all classes are present (strict)
                mask_full = (counts > 0).all(axis=1)
                grids_here = counts.index[mask_full].tolist()
                if not grids_here:
                    continue
                # additionally require that all grids present in this base satisfy presence
                if len(grids_here) != counts.shape[0]:
                    # not a full grid -> skip base
                    continue
            else:
                # use only grids where all classes are present; allow skipping others
                grids_here = counts.index[(counts > 0).all(axis=1)].tolist()
                if not grids_here:
                    # cannot equalize in this base, but we may still use it in later phases
                    grids_here = []

            if grids_here:
                # equalizable amount per grid is min across classes for that grid
                per_grid_min = counts.loc[grids_here].min(axis=1)
                m_b = int(per_grid_min.min()) if not per_grid_min.empty else 0
            else:
                m_b = 0

            # Record availability for all observed (grid, class)
            for gr in counts.index.tolist():
                for cl in counts.columns.tolist():
                    available[(str(base_val), str(gr), cl)] = int(counts.loc[gr, cl])

            grids_per_base[str(base_val)] = [str(gr) for gr in counts.index.tolist()]

            if m_b > 0 and grids_here:
                for gr in grids_here:
                    good_pairs.append((str(base_val), str(gr)))
                    for cl in counts.columns.tolist():
                        desired[(str(base_val), str(gr), cl)] = desired.get((str(base_val), str(gr), cl), 0) + m_b

        # Equal core contribution per class
        core_per_class = 0
        # sum m_b across good pairs equals sum over pairs of per-grid min; but we stored per pair desired as m_b
        # Compute core total per class from desired map
        for key, val in desired.items():
            # same across classes by construction; accumulate per class via one class only
            # We'll compute after symmetric extras by summing over class later; here leave as zero
            pass

        # Compute current per-class allocation
        def _current_total_for_class(cl: Any) -> int:
            return sum(v for (b, gr, c), v in desired.items() if c == cl)

        # Remaining quota per class to reach T
        remaining_per_class: Dict[Any, int] = {cl: max(T - _current_total_for_class(cl), 0) for cl in classes_all}

        # Step B: symmetric extras per (base, grid) adding +1 for every class together
        # Compute residual symmetric capacity per pair (min over classes of available - desired)
        pairs = sorted(set([(b, gr) for (b, gr, c) in available.keys()]))
        # But we prefer pairs that are in good_pairs first
        order = good_pairs + [p for p in pairs if p not in good_pairs]

        # When allocating symmetric batches, the limiting class defines how many full batches we can add
        def _pair_symmetric_capacity(b: str, gr: str) -> int:
            caps = []
            for cl in classes_all:
                caps.append(available.get((b, gr, cl), 0) - desired.get((b, gr, cl), 0))
            return max(min(caps), 0) if caps else 0

        # Make as many symmetric rounds as possible up to the remaining quota for the smallest class
        # Since all classes receive the same increments, use any class to bound iterations
        smallest_class = classes_all[int(np.argmin([cls_counts[cl] for cl in classes_all]))]
        R = remaining_per_class[smallest_class]
        if strategy == "undersample" and R > 0:
            num_pairs = len(order)
            # multi-pass round-robin with progress check
            while R > 0 and num_pairs > 0:
                made_progress = False
                for (b, gr) in order:
                    if R <= 0:
                        break
                    cap = _pair_symmetric_capacity(b, gr)
                    if cap > 0:
                        for cl in classes_all:
                            desired[(b, gr, cl)] = desired.get((b, gr, cl), 0) + 1
                            remaining_per_class[cl] = max(remaining_per_class[cl] - 1, 0)
                        R -= 1
                        made_progress = True
                if not made_progress:
                    break

        # Step C: fill remaining per class independently (keeps class totals equal to T)
        if strategy == "undersample":
            for cl in classes_all:
                r = remaining_per_class[cl]
                if r <= 0:
                    continue
                # list of pairs with remaining capacity for this class
                pair_caps = [
                    (b, gr, available.get((b, gr, cl), 0) - desired.get((b, gr, cl), 0))
                    for (b, gr) in order
                ]
                # filter positive capacity
                pair_caps = [(b, gr, cap) for (b, gr, cap) in pair_caps if cap > 0]
                if not pair_caps:
                    continue
                num = len(pair_caps)
                # multi-pass with progress check
                while r > 0 and num > 0:
                    made_progress = False
                    for i in range(num):
                        if r <= 0:
                            break
                        b, gr, cap = pair_caps[i]
                        if cap > 0:
                            desired[(b, gr, cl)] = desired.get((b, gr, cl), 0) + 1
                            cap -= 1
                            r -= 1
                            pair_caps[i] = (b, gr, cap)
                            made_progress = True
                    if not made_progress:
                        break

        # Now sample according to desired allocation (without replacement)
        # Build grid keys again per base and call sampler per (base, grid)
        for base_val, g in tmp2.groupby("__base__"):
            if grid_label_cols:
                grid_key = g[grid_label_cols].astype(str).agg("|".join, axis=1)
            else:
                grid_key = pd.Series(["__ALL__"] * len(g), index=g.index)
            g = g.assign(__gridkey__=grid_key)
            for gr, sub in g.groupby("__gridkey__"):
                sizes = {cl: desired.get((str(base_val), str(gr), cl), 0) for cl in classes_all}
                if sum(sizes.values()) == 0:
                    continue
                # Remove classes with zero to avoid sampling errors
                sizes = {cl: n for cl, n in sizes.items() if n > 0}
                if not sizes:
                    continue
                balanced_parts.append(
                    _sample_per_class(
                        sub,
                        target,
                        sizes,
                        random_state,
                        replace=False,
                        prefer_clean=prefer_clean_rows,
                        exclude_cols=exclude_cols,
                    )
                )
    else:
        # Original behavior: for each covariate stratum, balance classes
        for stratum, g in tmp.groupby("__strata__"):
            strat_counts = g[target].value_counts()
            if len(strat_counts) <= 1:
                # Nothing to balance if only one class present; keep as-is for undersample
                if strategy == "undersample":
                    # skip strata that do not have at least 2 classes to avoid bias
                    continue
                else:
                    # For oversample: duplicate within the single-class stratum to match max inside this stratum
                    max_in_stratum = int(strat_counts.max())
                    size = {cls: max_in_stratum for cls in strat_counts.index}
                    balanced_parts.append(_sample_per_class(g, target, size, random_state, replace=True))
                    continue

            if strategy == "undersample":
                m = int(strat_counts.min())
                size = {cls: m for cls in strat_counts.index}
                balanced_parts.append(
                    _sample_per_class(
                        g,
                        target,
                        size,
                        random_state,
                        replace=False,
                        prefer_clean=prefer_clean_rows,
                        exclude_cols=exclude_cols,
                    )
                )
            else:  # oversample
                M = int(strat_counts.max())
                size = {cls: M for cls in strat_counts.index}
                balanced_parts.append(
                    _sample_per_class(
                        g,
                        target,
                        size,
                        random_state,
                        replace=True,
                        prefer_clean=prefer_clean_rows,
                        exclude_cols=exclude_cols,
                    )
                )

    if not balanced_parts:
        # Fallback to global strategy if strata were all single-class and undersampling was chosen
        if strategy == "undersample":
            size = {cls: min_count for cls in cls_counts.index}
            return _sample_per_class(
                df,
                target,
                size,
                random_state,
                replace=False,
                prefer_clean=prefer_clean_rows,
                exclude_cols=exclude_cols,
            )
        else:
            size = {cls: max_count for cls in cls_counts.index}
            return _sample_per_class(
                df,
                target,
                size,
                random_state,
                replace=True,
                prefer_clean=prefer_clean_rows,
                exclude_cols=exclude_cols,
            )

    # Drop helper columns and shuffle to remove potential ordering by strata/classes
    out = pd.concat(balanced_parts)
    out = out.drop(columns=[c for c in ["__strata__", "__base__", "__gridkey__"] if c in out.columns])  # type: ignore[arg-type]
    return out.sample(frac=1.0, random_state=random_state)


__all__ = ["balance_dataset"]
