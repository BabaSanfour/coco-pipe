#!/usr/bin/env python3
"""
Plot sensor-level results from an aggregated coco_pipe run.

This script expects that you ran scripts/run_ml.py and produced an aggregated
pickle at results/<global_experiment_id>.pkl containing a dict:
    { analysis_id -> { model_name -> result_dict } }

Assumptions for this visualization:
  - The analysis unit is sensor (1 model per sensor using all features).
  - Each analysis_id encodes the sensor name (e.g., "..._Fz_...", "sensor-F3", etc.).
  - You provide a coordinates file with 2D positions for each sensor.

Outputs:
  - Topomap of sensor accuracies (or chosen metric) across sensors.
  - Bar plot of feature importances (with error bars if available) for the best sensor.

Usage example:
  python scripts/plot_sensor_analysis.py \
      --results results/toy_ml_config.pkl \
      --coords  coords/sensor_coords.csv \
      --model "Logistic Regression" \
      --metric accuracy \
      --save-topo results/topomap.png \
      --save-bar  results/best_sensor_features.png
"""

import argparse
import json
import os
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from coco_pipe.viz import plot_bar, plot_topomap


def load_coords(path: str) -> pd.DataFrame:
    """Load sensor coordinates from CSV/TSV/TXT/JSON into a DataFrame with index=name and columns ['x','y']."""
    lower = path.lower()
    if lower.endswith((".json", ".jsn")):
        with open(path, "r") as f:
            data = json.load(f)
        # Expect {name: [x,y]} or {name: {"x": x, "y": y}}
        rows = []
        for name, val in data.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                x, y = float(val[0]), float(val[1])
            elif isinstance(val, dict) and "x" in val and "y" in val:
                x, y = float(val["x"]), float(val["y"])
            else:
                raise ValueError(f"Invalid JSON coord entry for {name}: {val}")
            rows.append((name, x, y))
        df = pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")
        return df

    # CSV/TSV/TXT: detect separator
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}

    # Find name column
    name_col = None
    for cand in ("name", "sensor", "channel", "id"):
        if cand in cols:
            name_col = cols[cand]
            break
    if name_col is None:
        # if exactly three columns, assume first is name
        if df.shape[1] >= 3:
            name_col = df.columns[0]
        else:
            raise ValueError(
                "Coordinates file must include a sensor name column (e.g., 'name')."
            )

    # Find x/y columns
    x_col = None
    y_col = None
    for cand in ("x", "xs", "xpos", "x_coord"):
        if cand in cols:
            x_col = cols[cand]
            break
    for cand in ("y", "ys", "ypos", "y_coord"):
        if cand in cols:
            y_col = cols[cand]
            break
    if x_col is None or y_col is None:
        # Try second/third columns
        if df.shape[1] >= 3:
            x_col = x_col or df.columns[1]
            y_col = y_col or df.columns[2]
        else:
            raise ValueError("Coordinates file must include numeric x,y columns.")

    cdf = df[[name_col, x_col, y_col]].copy()
    cdf.columns = ["name", "x", "y"]
    cdf = cdf.set_index("name")
    # coerce to float
    cdf["x"] = pd.to_numeric(cdf["x"], errors="coerce")
    cdf["y"] = pd.to_numeric(cdf["y"], errors="coerce")
    return cdf.dropna()


def generate_coords_from_mne(
    montage: str = "standard_1020", restrict_to: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Generate 2D sensor coordinates from an MNE template montage.

    Parameters
    ----------
    montage : str
        Name of the standard montage to use (e.g., 'standard_1020', 'standard_1005', 'biosemi64').
    restrict_to : list of str, optional
        If provided, only return coordinates for these channel names.

    Returns
    -------
    DataFrame
        Index = sensor names, columns ['x','y'] with 2D coordinates derived from montage.

    Notes
    -----
    - This function requires the optional dependency 'mne'. If not installed,
      an ImportError is raised with guidance.
    - We use the x,y components from the montage's 3D positions as a top-view projection.
    """
    try:
        import mne  # type: ignore
    except Exception as e:
        raise ImportError(
            "This feature requires 'mne'. Install it via 'pip install mne'."
        ) from e

    std_montage = mne.channels.make_standard_montage(montage)
    pos = std_montage.get_positions()
    ch_pos = pos.get("ch_pos", {})
    rows = []
    if restrict_to is None:
        names = list(ch_pos.keys())
    else:
        names = list(restrict_to)
    for name in names:
        if name not in ch_pos:
            # Try relaxed matching (upper/capitalize)
            alt = None
            if name.upper() in ch_pos:
                alt = name.upper()
            elif name.capitalize() in ch_pos:
                alt = name.capitalize()
            if alt is None:
                continue
            xyz = ch_pos[alt]
            rows.append((name, float(xyz[0]), float(xyz[1])))
        else:
            xyz = ch_pos[name]
            rows.append((name, float(xyz[0]), float(xyz[1])))
    if not rows:
        # Fall back to all channels in montage
        rows = [(n, float(v[0]), float(v[1])) for n, v in ch_pos.items()]
    df = pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")
    return df


def pick_model(
    results_per_model: Dict[str, dict], preferred: Optional[Sequence[str]] = None
) -> str:
    preferred = preferred or ("Logistic Regression",)
    for m in preferred:
        if m in results_per_model:
            return m
    return next(iter(results_per_model))


def analysis_to_sensor(analysis_id: str, known_sensors: Sequence[str]) -> Optional[str]:
    toks = analysis_id.replace("-", "_").replace(" ", "_").split("_")
    known = set(known_sensors)
    for t in toks:
        if t in known:
            return t
        if t.upper() in known:
            return t.upper()
        c = t.capitalize()
        if c in known:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot sensor accuracies topomap and best sensor feature importances."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to aggregated results pickle (from run_ml.py)",
    )
    parser.add_argument(
        "--coords", required=False, help="Path to sensor coordinates (CSV/TSV/JSON)"
    )
    parser.add_argument(
        "--use-mne",
        action="store_true",
        help="Generate sensor coordinates from MNE standard montage",
    )
    parser.add_argument(
        "--montage",
        default="standard_1020",
        help="MNE montage name (default: standard_1020)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (default: try 'Logistic Regression' else first)",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Metric to plot for sensors (default: accuracy)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Top-N features in bar plot (default: 20)"
    )
    parser.add_argument(
        "--save-topo", default=None, help="Path to save topomap image (optional)"
    )
    parser.add_argument(
        "--save-bar", default=None, help="Path to save barplot image (optional)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows; save only",
    )

    args = parser.parse_args()

    # Non-interactive backend if --no-show and saving
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    if not os.path.exists(args.results):
        raise FileNotFoundError(args.results)
    all_results: Dict[str, Dict[str, dict]] = pd.read_pickle(args.results)

    # Determine sensor names from results (by parsing IDs) to optionally restrict MNE coords
    # Quick heuristic: collect all tokens from analysis ids that look like EEG names (letters+digits+optional z)
    candidate_tokens = set()
    for aid in all_results.keys():
        toks = aid.replace("-", "_").replace(" ", "_").split("_")
        for t in toks:
            if (
                len(t) <= 5
                and any(c.isalpha() for c in t)
                and any(c.isdigit() for c in t)
            ):
                candidate_tokens.add(t)

    if args.use_mne or not args.coords:
        coords_df = generate_coords_from_mne(
            args.montage,
            restrict_to=sorted(candidate_tokens) if candidate_tokens else None,
        )
    else:
        if not os.path.exists(args.coords):
            raise FileNotFoundError(args.coords)
        coords_df = load_coords(args.coords)

    sensor_names = coords_df.index.tolist()

    sensor_acc: Dict[str, float] = {}
    model_name_used: Optional[str] = None
    metric_name = args.metric

    for analysis_id, results_per_model in all_results.items():
        sensor = analysis_to_sensor(analysis_id, sensor_names)
        if not sensor:
            continue
        model_name = pick_model(
            results_per_model, preferred=(args.model,) if args.model else None
        )
        metrics = results_per_model[model_name].get("metric_scores", {})
        if metric_name not in metrics:
            # fallback to first metric if requested not found
            if metrics:
                metric_name = next(iter(metrics.keys()))
            else:
                continue
        mean_val = (
            float(metrics[metric_name]["mean"])
            if isinstance(metrics[metric_name], dict)
            else float(metrics[metric_name])
        )
        sensor_acc[sensor] = mean_val
        model_name_used = model_name

    if not sensor_acc:
        raise RuntimeError(
            "No sensor accuracies found. Ensure analysis IDs include sensor names and coords match."
        )

    # Topomap
    fig1, ax1 = plot_topomap(
        sensor_acc,
        coords_df[["x", "y"]],
        title=f"Sensor {metric_name.capitalize()} ({model_name_used})",
        cbar_label=metric_name.capitalize(),
        sensors="markers",
        cmap="viridis",
        symmetric=False,
    )
    if args.save_topo:
        fig1.savefig(args.save_topo, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig1)

    # Best sensor feature importances
    best_sensor = max(sensor_acc, key=sensor_acc.get)
    # find the analysis id for this sensor
    best_analysis_id = next(
        a for a in all_results if analysis_to_sensor(a, sensor_names) == best_sensor
    )
    best_model_results = all_results[best_analysis_id][model_name_used]
    fi_dict = best_model_results.get("feature_importances", {})

    if not fi_dict:
        print(
            f"No feature_importances available for {best_sensor} / {model_name_used}."
        )
        return

    imp_mean = pd.Series(
        {k: v.get("mean", 0.0) for k, v in fi_dict.items()}
    ).sort_values(ascending=False)
    imp_std = pd.Series({k: v.get("std", 0.0) for k, v in fi_dict.items()}).reindex(
        imp_mean.index
    )

    fig2, ax2 = plot_bar(
        imp_mean,
        errors=imp_std,
        orientation="horizontal",
        top_n=args.top_n,
        title=f"{best_sensor} Feature Importance ({model_name_used})",
        xlabel="Importance",
        cmap="magma",
    )
    if args.save_bar:
        fig2.savefig(args.save_bar, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    main()
