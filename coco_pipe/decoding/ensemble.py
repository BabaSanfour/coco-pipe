"""
Ensembling and Out-of-Fold Stacking
===================================

Higher-level helpers that combine predictions across decoding experiments,
supporting partial unit overlap and missing input blocks without imputation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from coco_pipe.io import DataContainer

from ._diagnostics import proba_columns, resolve_column
from ._splitters import get_cv_splitter
from .configs import (
    BaseEstimatorConfig,
    CVConfig,
    ExperimentConfig,
    HistGradientBoostingClassifierConfig,
)
from .experiment import Experiment
from .registry import get_estimator_cls, resolve_estimator_spec
from .stats import evaluate_binary_scores

__all__ = [
    "ensemble_predictions",
    "stack_oof_predictions",
]

_DEFAULT_UNIT_COLS = ("Group", "Subject", "SampleID")


def ensemble_predictions(
    fits: Mapping[str, Any],
    *,
    unit_col: str | None = None,
    score_col: str | None = None,
    evaluate: bool = False,
    n_permutations: int = 2000,
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, float]]:
    """Rank-average out-of-fold predictions across fits with partial unit overlap.

    Within each fit, predictions are averaged per unit and converted to
    percentile ranks in ``(0, 1]`` so fits with different calibrations weigh
    equally. Across fits, each unit's ensemble score is the mean of its ranks
    over the fits that scored it.

    Parameters
    ----------
    fits : mapping of str to ExperimentResult or pandas.DataFrame
        Named decoding results or prediction frames to combine.
    unit_col : str, optional
        Column identifying the independent unit. When ``None``, resolves the
        standard prediction schema in order: ``Group`` (if present and
        non-null), ``Subject`` (if present and non-null), else ``SampleID``.
        Callers with custom DataFrames should pass ``unit_col`` explicitly.
    score_col : str, optional
        Explicit score column. When ``None``, uses the highest-indexed
        ``y_proba_*`` column, falling back to ``y_score`` or ``score``.
    evaluate : bool, default=False
        If ``True``, also return binary evaluation metrics from
        :func:`~coco_pipe.decoding.stats.evaluate_binary_scores`.
    n_permutations : int, default=2000
        Number of label permutations when ``evaluate=True``.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples when ``evaluate=True``.
    random_state : int, default=0
        Random seed used when ``evaluate=True``.

    Returns
    -------
    ensemble : pandas.DataFrame
        DataFrame with columns ``unit``, ``y``, ``score``, and ``n_fits``.
    summary : dict of str to float, optional
        Returned alongside *ensemble* when ``evaluate=True``.

    Raises
    ------
    ValueError
        If *fits* is empty, required columns are missing or ambiguous, or a
        unit has conflicting labels within or across fits.
    """
    if not fits:
        raise ValueError("fits must not be empty.")

    ranked = []
    for name, item in fits.items():
        frame = (
            item.get_predictions()
            if hasattr(item, "get_predictions")
            else pd.DataFrame(item)
        )
        if frame.empty:
            continue

        if unit_col is not None:
            u_col = resolve_column(frame, unit_col)
        else:
            u_col = next(
                (
                    col
                    for col in _DEFAULT_UNIT_COLS
                    if col in frame.columns and frame[col].notna().any()
                ),
                None,
            )
            if u_col is None:
                raise ValueError(
                    f"No default unit column ({', '.join(_DEFAULT_UNIT_COLS)}) with "
                    f"values found in fit '{name}'. Pass unit_col=... explicitly. "
                    f"Available: {list(frame.columns)}"
                )

        if score_col is not None:
            s_col = resolve_column(frame, score_col)
        else:
            pcols = proba_columns(frame.columns)
            s_col = (
                pcols[-1]
                if pcols
                else resolve_column(frame, ("y_score", "score"), required=False)
            )
            if s_col is None:
                raise ValueError(
                    f"Fit '{name}' holds no class probabilities (y_proba_*), "
                    "y_score, or score column."
                )

        y_col = resolve_column(frame, ("y_true", "y"), required=False)
        grouped = frame.groupby(u_col, sort=False)
        if y_col is not None:
            n_lbl = grouped[y_col].nunique(dropna=True)
            if (n_lbl > 1).any():
                bad = list(n_lbl[n_lbl > 1].index)
                raise ValueError(
                    f"Units with two labels within fit '{name}': {bad[:5]}"
                )
            y_vals = grouped[y_col].first().to_numpy()
        else:
            y_vals = np.full(grouped.ngroups, np.nan)

        scores = grouped[s_col].mean()
        unit_df = pd.DataFrame(
            {
                "unit": scores.index,
                "y": y_vals,
                "score": scores.to_numpy(dtype=float),
            }
        )
        unit_df = unit_df[unit_df["score"].notna()].copy()
        if unit_df.empty:
            continue
        unit_df["rank"] = unit_df["score"].rank(pct=True)
        unit_df["fit"] = name
        ranked.append(unit_df)

    if not ranked:
        raise ValueError("No valid predictions found across fits.")

    stacked = pd.concat(ranked, ignore_index=True)
    labels = stacked.groupby("unit")["y"].nunique(dropna=True)
    if (labels > 1).any():
        raise ValueError(
            f"Units with two labels across fits: {list(labels[labels > 1].index)}"
        )
    out = (
        stacked.groupby("unit", sort=True)
        .agg(y=("y", "first"), score=("rank", "mean"), n_fits=("fit", "nunique"))
        .reset_index()
    )
    if evaluate:
        labeled = out[out["y"].notna()]
        summary = evaluate_binary_scores(
            labeled["y"].to_numpy(),
            labeled["score"].to_numpy(),
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        return out, summary
    return out


def stack_oof_predictions(
    inputs: Mapping[str, tuple[DataContainer | pd.DataFrame, ExperimentConfig]],
    y: pd.Series,
    folds: pd.Series,
    *,
    inner_cv: CVConfig | None = None,
    combiner: BaseEstimatorConfig | BaseEstimator | None = None,
    random_state: int = 0,
) -> pd.Series:
    """Nested out-of-fold stacking over inputs with missing blocks kept as NaN.

    For each outer fold, each input produces inner-CV out-of-fold predictions on
    the training units that have it and refit predictions on the test units
    that have it, via :class:`~coco_pipe.decoding.Experiment` with predefined
    folds. Units missing an input stay ``NaN`` (no imputation) and are combined
    by *combiner*.

    Parameters
    ----------
    inputs : mapping of str to (DataContainer or DataFrame, ExperimentConfig)
        Named feature inputs and their decoding experiment configs. When a
        :class:`~coco_pipe.io.DataContainer` has ``coords["feature_block"]``,
        per-block preprocessing is picked up automatically.
    y : pandas.Series
        Binary target indexed by unit ID.
    folds : pandas.Series
        Outer fold assignment indexed by unit ID.
    inner_cv : CVConfig, optional
        Inner cross-validation configuration used to generate meta-training
        predictions inside each outer training split. Defaults to stratified
        5-fold CV (automatically reduced if a class has fewer samples).
    combiner : BaseEstimatorConfig or BaseEstimator, optional
        Meta-classifier config (resolved through coco-pipe's estimator registry)
        or scikit-learn estimator with ``predict_proba`` and native NaN
        support. Defaults to a regularized
        :class:`~coco_pipe.decoding.configs.HistGradientBoostingClassifierConfig`.
    random_state : int, default=0
        Random seed for default inner splits and the default combiner.

    Returns
    -------
    scores : pandas.Series
        Out-of-fold positive-class probabilities aligned to ``y.index``.

    Raises
    ------
    ValueError
        If *inputs* is empty or a ``DataContainer`` input lacks ``ids``.
    """
    if not inputs:
        raise ValueError("inputs must not be empty.")

    if isinstance(combiner, BaseEstimator):
        base_combiner = clone(combiner)
    else:
        combiner_cfg = (
            combiner
            if combiner is not None
            else HistGradientBoostingClassifierConfig(
                max_depth=2,
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=random_state,
            )
        )
        spec = resolve_estimator_spec(combiner_cfg)
        est_cls = get_estimator_cls(spec.name)
        params = (
            dict(combiner_cfg.params)
            if hasattr(combiner_cfg, "params")
            else combiner_cfg.model_dump(exclude={"method", "kind"})
        )
        base_combiner = est_cls(**params)

    cv_cfg = (
        inner_cv
        if inner_cv is not None
        else CVConfig(
            strategy="stratified",
            n_splits=5,
            shuffle=True,
            random_state=random_state + 1,
        )
    )

    input_indices: dict[str, pd.Index] = {}
    for name, (x_in, _) in inputs.items():
        if isinstance(x_in, DataContainer):
            if x_in.ids is None:
                raise ValueError(f"DataContainer input '{name}' must define ids.")
            input_indices[name] = pd.Index(x_in.ids)
        else:
            input_indices[name] = x_in.index

    def _fit_and_predict(
        name: str,
        x_in: DataContainer | pd.DataFrame,
        cfg: ExperimentConfig,
        train_units: pd.Index,
        test_units: pd.Index,
        inner_fold: pd.Series | None,
    ) -> pd.Series:
        have = input_indices[name]
        tr = train_units.intersection(have)
        te = test_units.intersection(have)
        if inner_fold is not None:
            if len(tr) == 0:
                return pd.Series(dtype=float)
            ids = tr
            fold_vals = inner_fold.loc[tr].to_numpy()
            keep_ids = tr
        else:
            if len(tr) == 0 or len(te) == 0:
                return pd.Series(dtype=float)
            ids = tr.append(te)
            fold_vals = np.concatenate(
                [np.full(len(tr), -1, dtype=int), np.zeros(len(te), dtype=int)]
            )
            keep_ids = te

        if isinstance(x_in, DataContainer):
            pos = pd.Index(x_in.ids).get_indexer(ids)
            sub = x_in.isel(obs=pos)
            sub = sub.replace(coords={**sub.coords, "fold": fold_vals})
        else:
            sub = DataContainer(
                X=x_in.loc[ids].to_numpy(dtype=float),
                dims=("obs", "feature"),
                coords={"fold": fold_vals},
                ids=ids.to_numpy(),
            )

        run_cfg = cfg.model_copy(
            update={
                "cv": CVConfig(strategy="predefined", fold_col="fold"),
                "metrics": ["accuracy"],
                "n_jobs": 1,
                "verbose": False,
            }
        )
        res = Experiment(run_cfg).run(
            sub,
            y.loc[ids].to_numpy(),
            groups=ids.to_numpy(),
            sample_ids=ids.to_numpy(),
        )
        errors = [r["error"] for r in res.raw.values() if "error" in r]
        if errors:
            raise RuntimeError(f"Input '{name}' failed to fit: {errors[0]}")
        preds = res.get_predictions()
        pcols = proba_columns(preds.columns)
        s_col = pcols[-1] if pcols else "y_score"
        series = pd.Series(
            preds[s_col].to_numpy(dtype=float),
            index=preds["SampleID"].to_numpy(),
        )
        return series.reindex(keep_ids)

    out = pd.Series(np.nan, index=y.index, dtype=float)
    for k in sorted(folds.unique()):
        train_units = y.index[folds != k]
        test_units = y.index[folds == k]
        y_train = y.loc[train_units]
        inner = pd.Series(-1, index=train_units, dtype=int)
        splitter = get_cv_splitter(cv_cfg, y=y_train, task="classification")
        for j, (_, rows) in enumerate(splitter.split(np.zeros(len(y_train)), y_train)):
            inner.iloc[rows] = j
        meta_train = pd.DataFrame(index=train_units, columns=list(inputs), dtype=float)
        meta_test = pd.DataFrame(index=test_units, columns=list(inputs), dtype=float)
        for name, (x_in, cfg) in inputs.items():
            oof = _fit_and_predict(name, x_in, cfg, train_units, test_units, inner)
            if not oof.empty:
                meta_train.loc[oof.index, name] = oof
            pred = _fit_and_predict(name, x_in, cfg, train_units, test_units, None)
            if not pred.empty:
                meta_test.loc[pred.index, name] = pred
        fitted = clone(base_combiner).fit(meta_train, y_train)
        out.loc[test_units] = fitted.predict_proba(meta_test)[:, 1]
    return out
