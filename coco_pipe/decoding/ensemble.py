"""
Ensembling and Out-of-Fold Stacking
===================================

Higher-level helpers that combine predictions across decoding experiments,
supporting partial unit overlap and missing input blocks without imputation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from .configs import CVConfig
from .experiment import Experiment
from .stats import aggregate_predictions_for_inference, evaluate_binary_scores

__all__ = [
    "ensemble_predictions",
    "stack_oof_predictions",
]


def _resolve_col(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first matching column in *frame* (exact, then case-insensitive)."""
    lower = {str(col).casefold(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.casefold() in lower:
            return lower[candidate.casefold()]
    return None


def _sorted_proba_columns(columns: Sequence[Any]) -> list[str]:
    """Return ``y_proba_*`` columns sorted by their trailing integer class index."""
    return sorted(
        [col for col in columns if str(col).startswith("y_proba_")],
        key=lambda col: int(str(col).rsplit("_", 1)[-1]),
    )


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

    Within each fit, predictions are aggregated per unit (using
    :func:`~coco_pipe.decoding.stats.aggregate_predictions_for_inference` when
    class probabilities and targets are present) and converted to percentile
    ranks in ``(0, 1]`` so fits with different calibrations weigh equally.
    Across fits, each unit's ensemble score is the mean of its ranks over the
    fits that scored it.

    Parameters
    ----------
    fits : mapping of str to ExperimentResult or pandas.DataFrame
        Named decoding results or prediction frames to combine.
    unit_col : str, optional
        Column identifying the independent unit. When ``None``, common unit
        column names (``unit``, ``child``, ``group_id``, ``Group``,
        ``Subject``, ``SampleID``) are tried in order.
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
        If *fits* is empty, required columns are missing, or a unit has
        conflicting labels within or across fits.
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
        u_col = (
            _resolve_col(frame, [unit_col])
            if unit_col is not None
            else _resolve_col(
                frame, ("unit", "child", "group_id", "Group", "Subject", "SampleID")
            )
        )
        if u_col is None:
            raise ValueError(
                f"Unit column not found in fit '{name}'. "
                f"Available: {list(frame.columns)}"
            )
        y_col = _resolve_col(frame, ("y_true", "y"))
        proba_cols = _sorted_proba_columns(frame.columns)
        if score_col is not None:
            s_col = _resolve_col(frame, [score_col])
            if s_col is None:
                raise ValueError(f"score_col '{score_col}' not found in fit '{name}'.")
            use_agg_helper = False
        elif proba_cols:
            s_col = proba_cols[-1]
            use_agg_helper = y_col == "y_true" and frame["y_true"].notna().all()
        else:
            s_col = _resolve_col(frame, ("y_score", "score"))
            if s_col is None:
                raise ValueError(
                    f"Fit '{name}' holds no class probabilities (y_proba_*), "
                    "y_score, or score column."
                )
            use_agg_helper = False

        if use_agg_helper:
            agg = aggregate_predictions_for_inference(
                frame,
                metric="roc_auc",
                task="classification",
                unit_of_inference="custom",
                custom_unit_column=u_col,
                custom_aggregation="mean",
            )
            unit_df = pd.DataFrame(
                {
                    "unit": agg["InferentialUnitID"],
                    "y": agg["y_true"],
                    "score": agg[s_col].astype(float),
                }
            )
        else:
            grouped = frame.groupby(u_col, sort=False)
            if y_col is not None:
                n_lbl = grouped[y_col].nunique(dropna=True)
                if (n_lbl > 1).any():
                    bad = list(n_lbl[n_lbl > 1].index)
                    raise ValueError(
                        f"Units with two labels within fit '{name}': {bad[:5]}"
                    )
                y_vals = grouped[y_col].first()
            else:
                y_vals = pd.Series(np.nan, index=grouped.size().index)
            unit_df = pd.DataFrame(
                {
                    "unit": y_vals.index,
                    "y": y_vals.to_numpy(),
                    "score": grouped[s_col].mean().to_numpy(dtype=float),
                }
            )
        unit_df = unit_df[unit_df["score"].notna()].copy()
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
    inputs: Mapping[str, Any],
    y: pd.Series,
    folds: pd.Series,
    *,
    inner_splits: int = 5,
    combiner: Any | None = None,
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
    inputs : mapping of str to tuple or mapping
        Each value is either ``(X_frame, ExperimentConfig)`` or a mapping with
        keys ``"X"``, ``"config"``, and optional ``"feature_blocks"``.
    y : pandas.Series
        Binary target indexed by unit ID.
    folds : pandas.Series
        Outer fold assignment indexed by unit ID.
    inner_splits : int, default=5
        Maximum number of inner stratified folds used to generate meta-training
        predictions inside each outer training split.
    combiner : estimator, optional
        Scikit-learn compatible binary classifier with ``predict_proba`` and
        native NaN support. Defaults to a regularized
        :class:`sklearn.ensemble.HistGradientBoostingClassifier`.
    random_state : int, default=0
        Random seed for inner splits and the default combiner.

    Returns
    -------
    scores : pandas.Series
        Out-of-fold positive-class probabilities aligned to ``y.index``.
    """
    base_combiner = (
        combiner
        if combiner is not None
        else HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        )
    )

    def _unpack(spec: Any) -> tuple[pd.DataFrame, Any, Any]:
        if isinstance(spec, Mapping):
            return spec["X"], spec["config"], spec.get("feature_blocks")
        x_df, cfg = spec
        return x_df, cfg, None

    def _fit_and_predict(
        spec: Any,
        train_units: pd.Index,
        test_units: pd.Index,
        inner_fold: pd.Series | None,
    ) -> pd.Series:
        x_df, cfg, blocks = _unpack(spec)
        have = x_df.index
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
        run_cfg = cfg.model_copy(
            update={
                "cv": CVConfig(strategy="predefined", fold_col="fold"),
                "metrics": ["accuracy"],
                "n_jobs": 1,
                "verbose": False,
            }
        )
        res = Experiment(run_cfg).run(
            x_df.loc[ids].to_numpy(),
            y.loc[ids].to_numpy(),
            groups=ids.to_numpy(),
            sample_ids=ids.to_numpy(),
            sample_metadata=pd.DataFrame(
                {"Subject": ids.to_numpy(), "Session": "01", "fold": fold_vals}
            ),
            feature_blocks=blocks,
        )
        preds = res.get_predictions()
        proba_cols = _sorted_proba_columns(preds.columns)
        score_col = proba_cols[-1] if proba_cols else "y_score"
        series = pd.Series(
            preds[score_col].to_numpy(dtype=float),
            index=preds["SampleID"].to_numpy(),
        )
        return series.reindex(keep_ids)

    out = pd.Series(np.nan, index=y.index, dtype=float)
    for k in sorted(folds.unique()):
        train_units = y.index[folds != k]
        test_units = y.index[folds == k]
        y_train = y.loc[train_units]
        inner = pd.Series(-1, index=train_units, dtype=int)
        n_inner = min(inner_splits, int(y_train.value_counts().min()))
        splitter = StratifiedKFold(n_inner, shuffle=True, random_state=random_state + 1)
        for j, (_, rows) in enumerate(splitter.split(np.zeros(len(y_train)), y_train)):
            inner.iloc[rows] = j
        meta_train = pd.DataFrame(index=train_units, columns=list(inputs), dtype=float)
        meta_test = pd.DataFrame(index=test_units, columns=list(inputs), dtype=float)
        for name, spec in inputs.items():
            oof = _fit_and_predict(spec, train_units, test_units, inner)
            if not oof.empty:
                meta_train.loc[oof.index, name] = oof
            pred = _fit_and_predict(spec, train_units, test_units, None)
            if not pred.empty:
                meta_test.loc[pred.index, name] = pred
        fitted = clone(base_combiner).fit(meta_train, y_train)
        out.loc[test_units] = fitted.predict_proba(meta_test)[:, 1]
    return out
