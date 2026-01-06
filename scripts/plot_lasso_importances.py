#!/usr/bin/env python3
"""
Plot top-N feature importances for a Lasso-regularized Logistic Regression run,
and annotate accuracy and the number of zeroed features on the plot.

Inputs
------
- Aggregated pickle from scripts/run_ml.py (results/<global_experiment_id>.pkl)

Behavior
--------
- Select an analysis (by --analysis-id or first found) and a model (default 'Logistic Regression').
- Extract per-feature importances from results['feature_importances'].
- Rank by absolute importance by default (configurable) and plot top-N as a bar chart.
- Count zeroed features (all fold importances ~ 0 within tolerance) and include in title
  along with mean accuracy.

Example
-------
python scripts/plot_lasso_importances.py \
  --results results/toy_ml_config.pkl \
  --analysis-id classification_baseline \
  --model "Logistic Regression" \
  --metric accuracy \
  --top-n 20 \
  --abs \
  --save results/lasso_importances.png
"""

import argparse
import json
import os
from typing import Dict, Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from coco_pipe.viz import plot_bar, plot_scatter2d
from coco_pipe.io import load as load_data


def pick_analysis(all_results: Dict[str, Dict[str, dict]], analysis_id: Optional[str]) -> str:
    if analysis_id:
        if analysis_id not in all_results:
            raise KeyError(f"Analysis id '{analysis_id}' not found in results.")
        return analysis_id
    # fallback: first key
    return next(iter(all_results))


def pick_model(results_per_model: Dict[str, dict], preferred: Optional[Sequence[str]] = None) -> str:
    preferred = preferred or ("Logistic Regression",)
    for m in preferred:
        if m in results_per_model:
            return m
    return next(iter(results_per_model))


def _greekify(text: str) -> str:
    rep = {"alpha": "α", "beta": "β", "gamma": "γ", "theta": "θ", "delta": "δ"}
    for k, v in rep.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text


def make_label_map_keep_sensor(cols: Sequence[str]) -> Dict[str, str]:
    """Build compact labels per column, preserving the sensor name.

    Expected column format: 'feature-<FEATURE>.spaces-<SENSOR>'.
    Keeps '<SENSOR>' and abbreviates '<FEATURE>'.
    """
    abbrev = {
        "BandRatiosFromAverageFooof": "BR Fooof",
        "BandRatiosFromAverageSpectrum": "BR Spec",
        "RelativeBandPowerFromAverageFooof": "RBP Fooof",
        "RelativeBandPowerFromAverageSpectrum": "RBP Spec",
        "higuchiFd": "Higuchi FD",
        "katzFd": "Katz FD",
        "petrosianFd": "Petrosian FD",
        "hjorthComplexity": "Hjorth Complexity",
        "hjorthMobility": "Hjorth Mobility",
        "numZerocross": "ZeroCross",
        "svdEntropy": "SVD Entropy",
        "spectralEntropy": "Spectral Entropy",
        "sampleEntropy": "Sample Entropy",
        "permEntropy": "Perm Entropy",
        "entropyMultiscale": "MSE",
        "fooofExponent": "FOOOF Exp",
        "foofOffset": "FOOOF Off",
        "fooofOffset": "FOOOF Off",
    }

    out: Dict[str, str] = {}
    for raw in cols:
        s = raw
        if s.startswith("feature-"):
            s = s[len("feature-") :]
        # Extract sensor from trailing '.spaces-<SENSOR>'
        sensor = None
        m_sensor = re.search(r"\.spaces-([A-Za-z0-9]+)$", s)
        if m_sensor:
            sensor = m_sensor.group(1)
            s = s[: m_sensor.start()]  # strip the sensor suffix
        # Abbreviate feature part
        m_pair = re.search(r"bands_pairs-\((.+)\)", s)
        if m_pair:
            pair = m_pair.group(1).replace("'", "").replace(" ", "").replace(",", "/")
            pair = _greekify(pair)
            head = s[: m_pair.start()].rstrip(".")
            for k, v in abbrev.items():
                head = head.replace(k, v)
            feat_label = f"{head} {pair}".strip()
        else:
            feat_label = s
            for k, v in abbrev.items():
                feat_label = feat_label.replace(k, v)
            feat_label = _greekify(feat_label)
            feat_label = feat_label.replace("MeanEpochs", "")
            feat_label = re.sub(r"[_.-]+$", "", feat_label).strip()

        out[raw] = f"{sensor or ''} — {feat_label}".strip(" —")
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot L1-LR feature importances (top-N) with zeroed count and accuracy.")
    parser.add_argument("--results", required=True, help="Path to aggregated results pickle (from run_ml.py)")
    parser.add_argument("--analysis-id", default=None, help="Analysis id key from the aggregated results dict")
    parser.add_argument("--model", default=None, help="Model name (default: try 'Logistic Regression' else first)")
    parser.add_argument("--metric", default="accuracy", help="Metric name to show in title (default: accuracy)")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N features to plot (default: 20)")
    parser.add_argument("--abs", dest="use_abs", action="store_true", help="Rank by absolute importance (default)")
    parser.add_argument("--no-abs", dest="use_abs", action="store_false", help="Rank by signed importance")
    parser.set_defaults(use_abs=True)
    parser.add_argument("--zero-threshold", type=float, default=1e-12, help="Threshold to treat coefficients as zero across folds")
    parser.add_argument("--label-map", default=None, help="Optional JSON mapping from feature name to display label")
    parser.add_argument("--xlabel", default=None, help="X-axis label (default: 'Coefficient magnitude' if --abs else 'Coefficient')")
    parser.add_argument("--save", default=None, help="Path to save the figure (optional)")
    # For scatter plots
    parser.add_argument("--data", required=False, help="Path to original dataset (CSV/TSV/Excel) for scatter plots")
    parser.add_argument("--target", required=False, help="Target column name in the dataset")
    parser.add_argument("--sep", dest="csv_sep", default=None, help="CSV separator if needed (auto by extension otherwise)")
    parser.add_argument("--sheet", dest="sheet_name", default=None, help="Excel sheet name if applicable")
    parser.add_argument("--scatter-dir", default=None, help="Directory to save scatter plots (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive windows; save only")

    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if not os.path.exists(args.results):
        raise FileNotFoundError(args.results)

    all_results: Dict[str, Dict[str, dict]] = pd.read_pickle(args.results)
    aid = pick_analysis(all_results, args.analysis_id)
    res_per_model = all_results[aid]
    model_name = pick_model(res_per_model, preferred=(args.model,) if args.model else None)
    res = res_per_model[model_name]

    # Accuracy (or desired metric)
    metrics = res.get("metric_scores", {})
    metric_name = args.metric if args.metric in metrics else (next(iter(metrics)) if metrics else None)
    acc_mean = float(metrics[metric_name]["mean"]) if metric_name and isinstance(metrics[metric_name], dict) else None

    # Feature importances
    fi = res.get("feature_importances", {})
    if not fi:
        raise RuntimeError("No feature_importances found in results for the selected model.")

    # Build series of importance (weighted_mean if present else mean)
    values = {}
    zeros = 0
    for fname, stats in fi.items():
        imp = stats.get("weighted_mean", stats.get("mean", 0.0))
        # zero detection using fold_importances
        folds = np.asarray(stats.get("fold_importances", []), dtype=float)
        if folds.size > 0 and np.all(np.abs(folds) <= args.zero_threshold):
            zeros += 1
        values[fname] = float(imp)

    s = pd.Series(values)
    rank_vals = s.abs() if args.use_abs else s
    # Take top-N indices by ranking
    top_idx = rank_vals.sort_values(ascending=False).head(args.top_n).index
    s_top = s.loc[top_idx]
    rank_top = (s_top.abs() if args.use_abs else s_top).sort_values(ascending=False)
    s_top = s.loc[rank_top.index]

    # Optional label mapping (JSON) merged with auto-compact labels
    auto_map = make_label_map_keep_sensor(s_top.index.tolist())
    label_map: Mapping[str, str] = dict(auto_map)
    if args.label_map:
        with open(args.label_map, "r") as f:
            label_map.update(json.load(f))

    xlabel = args.xlabel
    if xlabel is None:
        xlabel = "Coefficient magnitude" if args.use_abs else "Coefficient"

    title_parts = [f"{model_name}"]
    if acc_mean is not None:
        title_parts.append(f"{metric_name.capitalize()}: {acc_mean:.3f}")
    title_parts.append(f"Zeroed features: {zeros}")
    title = " — ".join(title_parts)

    fig, ax = plot_bar(
        s_top,
        labels=s_top.index.tolist(),
        label_map=label_map,
        top_n=None,  # already trimmed
        ascending=False,
        orientation="vertical",
        title=title,
        xlabel=xlabel,
        cmap="magma",
        figsize=(10, 4)
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # Scatter plots for top importances if data is available
    if args.data and args.target:
        os.makedirs(args.scatter_dir or os.path.dirname(args.save or "") or ".", exist_ok=True)
        df = load_data("tabular", args.data, sheet_name=args.sheet_name, sep=args.csv_sep)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Expected a DataFrame from data loader.")

        if args.target not in df.columns:
            raise KeyError(f"Target column '{args.target}' not found in dataset")

        # Determine top positive and negative features by signed mean coefficients
        s_signed = pd.Series({k: v.get("mean", 0.0) for k, v in fi.items()})
        # Filter to columns present in df
        s_signed = s_signed[[c for c in s_signed.index if c in df.columns]]
        pos_feats = s_signed.sort_values(ascending=False).head(2).index.tolist()
        neg_feats = s_signed.sort_values(ascending=True).head(2).index.tolist()
        # Best positive and best negative
        best_pos = pos_feats[0] if pos_feats else None
        best_neg = neg_feats[0] if neg_feats else None

        y = df[args.target]

        # Helper to plot a pair if valid
        def _scatter_pair(fx: Optional[str], fy: Optional[str], name: str):
            if not fx or not fy:
                return
            if fx not in df.columns or fy not in df.columns:
                return
            out_path = None
            if args.scatter_dir:
                out_path = os.path.join(args.scatter_dir, f"scatter_{name}.png")
            title = f"{model_name} – {name} (top features)\n{metric_name.capitalize()}: {acc_mean:.3f}" if acc_mean is not None else f"{model_name} – {name} (top features)"
            fig_s, ax_s = plot_scatter2d(
                df[fx].values,
                df[fy].values,
                labels=y.values,
                title=title,
                xlabel=fx,
                ylabel=fy,
                save=out_path,
            )
            if not args.no_show and not args.save:
                plt.show()
            plt.close(fig_s)

        # (i) top two positive
        if len(pos_feats) >= 2:
            _scatter_pair(pos_feats[0], pos_feats[1], "top2_positive")
        # (ii) top two negative
        if len(neg_feats) >= 2:
            _scatter_pair(neg_feats[0], neg_feats[1], "top2_negative")
        # (iii) top positive and top negative
        if best_pos and best_neg:
            _scatter_pair(best_pos, best_neg, "top_pos_neg")
    else:
        # If not provided, inform how to enable scatter
        if not args.no_show:
            print("Skipping scatter plots: provide --data and --target to enable.")


if __name__ == "__main__":
    main()
