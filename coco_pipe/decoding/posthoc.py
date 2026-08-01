"""Post-hoc scoring of a saved decoding Result.

Re-scores the per-fold predictions stored by ``Result.save()`` to produce BOTH
epoch-level and subject-level metrics (incl. ``balanced_accuracy_optimal``) from a
single CV run. This complements ``CVConfig.subject_level_metrics``, which is a
per-run switch (one level per run): here both levels come from one run's stored
predictions. Pure re-scoring, so numbers match the CV loop (per fold, then averaged).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._metrics import _balanced_accuracy_optimal_score


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, proba1: np.ndarray) -> dict:
    """Classification metrics from one set of predictions (probability of class 1)."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    )
    two_class = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba1)) if two_class else float("nan"),
        "balanced_accuracy_optimal": (
            _balanced_accuracy_optimal_score(y_true, proba1) if two_class else float("nan")),
    }


def aggregate_subject(y_true: np.ndarray, proba1: np.ndarray, groups: np.ndarray):
    """Mean probability per subject -> one prediction per subject (threshold 0.5)."""
    uniq = np.unique(groups)
    sy = np.array([int(round(float(y_true[groups == g].mean()))) for g in uniq])
    sp = np.array([float(proba1[groups == g].mean()) for g in uniq])
    return sy, (sp >= 0.5).astype(int), sp


def posthoc_metrics_from_result(json_path, analysis_level: str = "epoch_level"):
    """Write a ``<name>_posthoc_metrics.json`` sidecar next to a saved Result JSON,
    holding epoch- and subject-level metrics (incl. balanced_accuracy_optimal),
    computed per fold then averaged. Returns ``(summary, sidecar_path)``."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    summary = {}
    for model, node in data.get("results", {}).items():
        per_level = {"epoch_level": [], "subject_level": []}
        for fold in node.get("predictions", []):
            yt = np.asarray(fold["y_true"])
            yp = np.asarray(fold["y_pred"])
            proba = np.asarray(fold["y_proba"])
            p1 = proba[:, 1] if proba.ndim == 2 else proba
            per_level["epoch_level"].append(score_predictions(yt, yp, p1))
            grp = fold.get("group")
            if grp is not None:
                grp = np.asarray(grp)
                if grp.size == yt.size and len(np.unique(grp)) < yt.size:
                    sy, spred, sp = aggregate_subject(yt, p1, grp)
                    per_level["subject_level"].append(score_predictions(sy, spred, sp))
        out = {}
        for lvl, folds in per_level.items():
            if not folds:
                continue
            out[lvl] = {
                k: {
                    "mean": float(np.nanmean([f[k] for f in folds])),
                    "std": float(np.nanstd([f[k] for f in folds])),
                }
                for k in folds[0]
            }
        summary[model] = out
    sidecar = json_path.with_name(json_path.stem + "_posthoc_metrics.json")
    sidecar.write_text(
        json.dumps({"analysis_level": analysis_level, "metrics": summary}, indent=2)
    )
    return summary, sidecar
