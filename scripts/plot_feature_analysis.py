#!/usr/bin/env python3
"""
Plot feature-level results (analysis unit = feature) from an aggregated coco_pipe run.

For each computed feature (one analysis per feature using all sensors as inputs), this script:
  - Plots a topomap of sensor importances (one topomap per computed feature).
  - Aggregates the per-feature accuracy (or chosen metric) into a bar plot.

Inputs
------
- Aggregated pickle from scripts/run_ml.py (results/<global_experiment_id>.pkl)
- Sensor coordinates via either:
    * --coords CSV/TSV/JSON file with columns name,x,y (case-insensitive), or
    * --use-mne and --montage to generate from an MNE standard montage

Outputs
-------
- One topomap image per computed feature.
- A bar plot summarizing per-feature metric (e.g., accuracy).

Example
-------
python scripts/plot_feature_analysis.py \
  --results results/toy_ml_config.pkl \
  --use-mne --montage standard_1020 \
  --model "Random Forest" \
  --metric accuracy \
  --out-dir results/feature_plots
"""

import argparse
import json
import os
import re
from typing import Dict, Tuple, Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coco_pipe.viz import plot_topomap, plot_bar


def load_coords(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".json", ".jsn")):
        with open(path, "r") as f:
            raw = json.load(f)
        rows = []
        for name, val in raw.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                x, y = float(val[0]), float(val[1])
            elif isinstance(val, dict) and "x" in val and "y" in val:
                x, y = float(val["x"]), float(val["y"])
            else:
                raise ValueError(f"Invalid JSON coord entry for {name}: {val}")
            rows.append((name, x, y))
        return pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")

    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    name_col = next((cols[c] for c in ("name", "sensor", "channel", "id") if c in cols), None)
    if name_col is None:
        name_col = df.columns[0]
    x_col = next((cols[c] for c in ("x", "xs", "xpos", "x_coord") if c in cols), None)
    y_col = next((cols[c] for c in ("y", "ys", "ypos", "y_coord") if c in cols), None)
    if x_col is None or y_col is None:
        if df.shape[1] >= 3:
            x_col = x_col or df.columns[1]
            y_col = y_col or df.columns[2]
        else:
            raise ValueError("Coordinates file must include numeric x,y columns.")
    cdf = df[[name_col, x_col, y_col]].copy()
    cdf.columns = ["name", "x", "y"]
    cdf = cdf.set_index("name")
    cdf["x"] = pd.to_numeric(cdf["x"], errors="coerce")
    cdf["y"] = pd.to_numeric(cdf["y"], errors="coerce")
    return cdf.dropna()


def generate_coords_from_mne(montage: str = "standard_1020",
                             restrict_to: Optional[Sequence[str]] = None) -> pd.DataFrame:
    try:
        import mne  # type: ignore
    except Exception as e:
        raise ImportError("This feature requires 'mne'. Install it via 'pip install mne'.") from e
    std_montage = mne.channels.make_standard_montage(montage)
    pos = std_montage.get_positions()
    ch_pos = pos.get('ch_pos', {})
    rows = []
    names = list(restrict_to) if restrict_to else list(ch_pos.keys())
    for name in names:
        key = name
        if key not in ch_pos:
            if name.upper() in ch_pos:
                key = name.upper()
            elif name.capitalize() in ch_pos:
                key = name.capitalize()
            else:
                continue
        xyz = ch_pos[key]
        rows.append((name, float(xyz[0]), float(xyz[1])))
    if not rows:
        rows = [(n, float(v[0]), float(v[1])) for n, v in ch_pos.items()]
    return pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")


def pick_model(results_per_model: Dict[str, dict], preferred: Optional[Sequence[str]] = None) -> str:
    preferred = preferred or ("Logistic Regression",)
    for m in preferred:
        if m in results_per_model:
            return m
    return next(iter(results_per_model))


def sensor_from_column(col: str, sensors: Sequence[str], sep: str, reverse: bool) -> Optional[str]:
    sensors_set = set(sensors)
    if sep in col:
        left, right = col.split(sep, 1)
        cand = right if reverse else left
        if cand in sensors_set:
            return cand
        up = cand.upper()
        cap = cand.capitalize()
        if up in sensors_set:
            return up
        if cap in sensors_set:
            return cap
    # fallback: look for a sensor token within the column name
    for s in sensors:
        if re.search(rf"\b{re.escape(s)}\b", col, flags=re.IGNORECASE):
            return s
    return None


def feature_from_columns(cols: Sequence[str], sep: str, reverse: bool) -> Optional[str]:
    tokens = []
    for c in cols:
        if sep not in c:
            continue
        left, right = c.split(sep, 1)
        tokens.append(left if reverse else right)
    if not tokens:
        return None
    # return most common token
    return pd.Series(tokens).value_counts().idxmax()


def main():
    parser = argparse.ArgumentParser(description="Plot per-feature topomaps of sensor importances and bar plot of per-feature metric.")
    parser.add_argument("--results", required=True, help="Path to aggregated results pickle (from run_ml.py)")
    parser.add_argument("--coords", required=False, help="Path to sensor coordinates (CSV/TSV/JSON)")
    parser.add_argument("--use-mne", action="store_true", help="Generate sensor coordinates from MNE standard montage")
    parser.add_argument("--montage", default="standard_1020", help="MNE montage name (default: standard_1020)")
    parser.add_argument("--model", default=None, help="Model name to use (default: try 'Logistic Regression' else first)")
    parser.add_argument("--metric", default="accuracy", help="Metric to plot per feature (default: accuracy)")
    parser.add_argument("--sep", default="_", help="Separator between unit and feature in column names (default: _)")
    parser.add_argument("--reverse", action="store_true", help="If set, interpret columns as <feature><sep><unit>")
    parser.add_argument("--out-dir", default="results/feature_analysis_plots", help="Directory to save plots")
    parser.add_argument("--label-map", default=None, help="Optional JSON mapping from feature name to display label")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive windows; save only")

    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if not os.path.exists(args.results):
        raise FileNotFoundError(args.results)

    os.makedirs(args.out_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, dict]] = pd.read_pickle(args.results)

    # Attempt to infer sensor names from columns later; for MNE restriction, we can pass None
    if args.use_mne or not args.coords:
        coords_df = generate_coords_from_mne(args.montage)
    else:
        if not os.path.exists(args.coords):
            raise FileNotFoundError(args.coords)
        coords_df = load_coords(args.coords)

    sensors = coords_df.index.tolist()

    # Optional feature label mapping
    feature_label_map: Mapping[str, str] = {}
    if args.label_map:
        with open(args.label_map, "r") as f:
            feature_label_map = json.load(f)

    # Collect per-feature metric values and produce topomaps
    feature_metric: Dict[str, float] = {}

    for analysis_id, results_per_model in all_results.items():
        model_name = pick_model(results_per_model, preferred=(args.model,) if args.model else None)
        res = results_per_model[model_name]

        # metric
        metrics = res.get("metric_scores", {})
        metric_name = args.metric if args.metric in metrics else (next(iter(metrics)) if metrics else None)
        if metric_name is None:
            continue

        # importances
        fi = res.get("feature_importances", {})
        if not fi:
            continue

        col_names = list(fi.keys())
        # Derive computed feature name from columns
        feat_name = feature_from_columns(col_names, sep=args.sep, reverse=args.reverse) or str(analysis_id)

        # sensor importances: prefer weighted_mean, else mean
        sensor_imp: Dict[str, float] = {}
        for col, stats in fi.items():
            sname = sensor_from_column(col, sensors, args.sep, args.reverse)
            if not sname:
                continue
            val = stats.get("weighted_mean")
            if val is None:
                val = stats.get("mean")
            if val is None:
                continue
            sensor_imp[sname] = float(val)

        if not sensor_imp:
            continue

        # Plot topomap for this computed feature
        disp_name = feature_label_map.get(feat_name, feat_name)
        fig, ax = plot_topomap(
            sensor_imp,
            coords_df[["x", "y"]],
            title=f"{disp_name} – Sensor Importances ({model_name})",
            cbar_label="Importance",
            sensors="markers",
            cmap="magma",
        )
        out_path = os.path.join(args.out_dir, f"topomap_{re.sub(r'[^A-Za-z0-9_.-]+','_', disp_name)}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        # Store metric
        mean_val = float(metrics[metric_name]["mean"]) if isinstance(metrics[metric_name], dict) else float(metrics[metric_name])
        feature_metric[disp_name] = mean_val

    if not feature_metric:
        raise RuntimeError("No per-feature metrics collected; check results structure and options.")

    # Bar plot of per-feature metric
    # Sort descending for visibility
    s = pd.Series(feature_metric).sort_values(ascending=False)
    fig2, ax2 = plot_bar(
        s,
        orientation="horizontal",
        title=f"Per-Feature {args.metric.capitalize()} ({args.model or 'auto'})",
        xlabel=args.metric.capitalize(),
        cmap="viridis",
    )
    fig2.savefig(os.path.join(args.out_dir, f"features_{args.metric}_bar.png"), dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    main()

