#!/usr/bin/env python3
"""
Numeric agreement comparison for overlapping `coco-pipe.descriptors` and
`mne-features` outputs.

This script complements the broader provenance/benchmark harness by comparing
aligned overlapping features value-by-value on the same deterministic synthetic
signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "scripts" / "outputs" / "numeric_agreement_vs_mne_features"
)

EXACT_SCALAR_MAPPINGS = [
    ("sample_entropy", "samp_entropy"),
    ("approx_entropy", "app_entropy"),
    ("svd_entropy", "svd_entropy"),
    ("higuchi_fd", "higuchi_fd"),
    ("katz_fd", "katz_fd"),
    ("hjorth_mobility", "hjorth_mobility"),
    ("hjorth_complexity", "hjorth_complexity"),
    ("zero_crossings", "zero_crossings"),
    ("kurtosis", "kurtosis"),
    ("rms", "rms"),
]

APPROX_SCALAR_MAPPINGS = [
    ("spectral_entropy", "spect_entropy"),
    ("hurst_exponent", "hurst_exp"),
]

BAND_INDEX_TO_NAME = {
    0: "delta",
    1: "theta",
    2: "alpha",
    3: "beta",
    4: "gamma",
}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _default_environment(temp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(temp_root / "home")
    env["MPLCONFIGDIR"] = str(temp_root / "mpl")
    env["PYTHONNOUSERSITE"] = "1"
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _flatten_mne_columns(columns: list[Any]) -> list[str]:
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, (list, tuple)) and len(column) == 2:
            flattened.append(f"{column[0]}__{column[1]}")
        else:
            flattened.append(str(column))
    return flattened


def _run_subprocess_json(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip())


def _build_input_npz(
    *, output_path: Path, n_epochs: int, n_channels: int, n_times: int, sfreq: float
) -> None:
    from scripts.compare_descriptors_vs_mne_features import _build_signal_array

    X = _build_signal_array(n_epochs=n_epochs, n_channels=n_channels, n_times=n_times)
    channel_names = np.asarray(
        [f"C{idx:02d}" for idx in range(n_channels)], dtype=object
    )
    np.savez_compressed(
        output_path,
        X=X,
        sfreq=float(sfreq),
        channel_names=channel_names,
    )


def _extract_coco(input_npz: Path, output_npz: Path) -> None:
    from coco_pipe.descriptors import DescriptorPipeline

    archive = np.load(input_npz, allow_pickle=True)
    X = archive["X"]
    sfreq = float(archive["sfreq"])
    channel_names = [str(name) for name in archive["channel_names"].tolist()]

    from scripts.compare_descriptors_vs_mne_features import _coco_config_for_subset

    pipe = DescriptorPipeline(_coco_config_for_subset("overlap_combined"))
    result = pipe.extract(X=X, sfreq=sfreq, channel_names=channel_names)
    np.savez_compressed(
        output_npz,
        X=result["X"],
        descriptor_names=np.asarray(result["descriptor_names"], dtype=object),
        channel_names=np.asarray(channel_names, dtype=object),
    )


def _handle_internal_mne_extract(payload_path: Path) -> int:
    from mne_features.feature_extraction import extract_features

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    input_npz = Path(payload["input_npz"])
    output_npz = Path(payload["output_npz"])

    archive = np.load(input_npz, allow_pickle=True)
    X = archive["X"]
    sfreq = float(archive["sfreq"])
    channel_names = [str(name) for name in archive["channel_names"].tolist()]

    from scripts.compare_descriptors_vs_mne_features import _mne_feature_subset

    selected_funcs, funcs_params = _mne_feature_subset("overlap_combined")
    df = extract_features(
        X,
        sfreq=sfreq,
        selected_funcs=selected_funcs,
        funcs_params=funcs_params,
        n_jobs=1,
        ch_names=channel_names,
        return_as_df=True,
    )
    np.savez_compressed(
        output_npz,
        X=df.to_numpy(),
        descriptor_names=np.asarray(
            _flatten_mne_columns(df.columns.tolist()), dtype=object
        ),
        channel_names=np.asarray(channel_names, dtype=object),
    )
    print(json.dumps({"status": "ok", "output_npz": str(output_npz)}))
    return 0


def _load_feature_matrix(npz_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    archive = np.load(npz_path, allow_pickle=True)
    X = archive["X"]
    descriptor_names = [str(name) for name in archive["descriptor_names"].tolist()]
    channel_names = [str(name) for name in archive["channel_names"].tolist()]
    return X, descriptor_names, channel_names


def _compare_vectors(coco_values: np.ndarray, mne_values: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(coco_values) & np.isfinite(mne_values)
    coco_masked = coco_values[mask]
    mne_masked = mne_values[mask]
    n_values = int(mask.sum())
    if n_values == 0:
        return {
            "n_values": 0,
            "mean_coco": None,
            "mean_mne": None,
            "std_coco": None,
            "std_mne": None,
            "pearson_r": None,
            "mae": None,
            "rmse": None,
            "max_abs_error": None,
        }

    diff = coco_masked - mne_masked
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs_error = float(np.max(np.abs(diff)))
    if n_values > 1 and np.std(coco_masked) > 0 and np.std(mne_masked) > 0:
        pearson_r = float(np.corrcoef(coco_masked, mne_masked)[0, 1])
    else:
        pearson_r = math.nan
    return {
        "n_values": n_values,
        "mean_coco": float(np.mean(coco_masked)),
        "mean_mne": float(np.mean(mne_masked)),
        "std_coco": float(np.std(coco_masked)),
        "std_mne": float(np.std(mne_masked)),
        "pearson_r": pearson_r,
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }


def _compare_scalar_features(
    *,
    coco_X: np.ndarray,
    coco_names: list[str],
    mne_X: np.ndarray,
    mne_names: list[str],
    channel_names: list[str],
    mappings: list[tuple[str, str]],
    relationship: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coco_feature, mne_feature in mappings:
        coco_indices = [
            coco_names.index(f"complexity_{coco_feature}_ch-{channel}")
            for channel in channel_names
        ]
        mne_indices = [
            mne_names.index(f"{mne_feature}__{channel}") for channel in channel_names
        ]
        coco_values = coco_X[:, coco_indices].reshape(-1)
        mne_values = mne_X[:, mne_indices].reshape(-1)
        row = {
            "relationship": relationship,
            "coco_feature": coco_feature,
            "mne_feature": mne_feature,
            "n_epochs": int(coco_X.shape[0]),
            "n_channels": len(channel_names),
        }
        row.update(_compare_vectors(coco_values, mne_values))
        rows.append(row)
    return rows


def _compare_band_powers(
    *,
    coco_X: np.ndarray,
    coco_names: list[str],
    mne_X: np.ndarray,
    mne_names: list[str],
    channel_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band_index, band_name in BAND_INDEX_TO_NAME.items():
        coco_indices = [
            coco_names.index(f"band_abs_{band_name}_ch-{channel}")
            for channel in channel_names
        ]
        mne_indices = [
            mne_names.index(f"pow_freq_bands__{channel}_band{band_index}")
            for channel in channel_names
        ]
        coco_values = coco_X[:, coco_indices].reshape(-1)
        mne_values = mne_X[:, mne_indices].reshape(-1)
        row = {
            "relationship": "approx_overlap",
            "coco_feature": f"absolute_power[{band_name}]",
            "mne_feature": f"pow_freq_bands[band{band_index}]",
            "n_epochs": int(coco_X.shape[0]),
            "n_channels": len(channel_names),
        }
        row.update(_compare_vectors(coco_values, mne_values))
        rows.append(row)
    return rows


def render_numeric_summary(rows: list[dict[str, Any]]) -> str:
    exact_rows = [row for row in rows if row["relationship"] == "exact_overlap"]
    approx_rows = [row for row in rows if row["relationship"] == "approx_overlap"]

    def _line(row: dict[str, Any]) -> str:
        return (
            f"- `{row['coco_feature']}` ↔ `{row['mne_feature']}`: "
            f"r={row['pearson_r']:.6f}, mae={row['mae']:.6g}, rmse={row['rmse']:.6g}, "
            f"max_abs_error={row['max_abs_error']:.6g}"
        )

    lines = [
        "# Numeric agreement summary",
        "",
        "## Exact-overlap features",
        *[_line(row) for row in exact_rows],
        "",
        "## Approximate-overlap features",
        *[_line(row) for row in approx_rows],
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    from scripts.compare_descriptors_vs_mne_features import (
        DEFAULT_COCO_PYTHON,
        DEFAULT_MNE_FEATURES_PYTHON,
    )

    parser = argparse.ArgumentParser(
        description="Numeric agreement comparison for overlapping "
        "coco-pipe and mne-features outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where numeric comparison artifacts will be written.",
    )
    parser.add_argument(
        "--coco-python",
        type=Path,
        default=DEFAULT_COCO_PYTHON,
        help="Python executable for the coco-pipe environment.",
    )
    parser.add_argument(
        "--mne-features-python",
        type=Path,
        default=DEFAULT_MNE_FEATURES_PYTHON,
        help="Python executable for the mne-features environment.",
    )
    parser.add_argument("--n-epochs", type=int, default=64)
    parser.add_argument("--n-channels", type=int, default=19)
    parser.add_argument("--n-times", type=int, default=512)
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--internal-mne-extract", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.internal_mne_extract:
        return _handle_internal_mne_extract(args.internal_mne_extract)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="numeric_agreement_") as tmp_dir:
        temp_root = Path(tmp_dir)
        input_npz = temp_root / "input.npz"
        coco_npz = temp_root / "coco_overlap.npz"
        mne_npz = temp_root / "mne_overlap.npz"

        _build_input_npz(
            output_path=input_npz,
            n_epochs=args.n_epochs,
            n_channels=args.n_channels,
            n_times=args.n_times,
            sfreq=args.sfreq,
        )
        _extract_coco(input_npz, coco_npz)

        payload_path = temp_root / "mne_payload.json"
        _write_json(
            {"input_npz": str(input_npz), "output_npz": str(mne_npz)}, payload_path
        )
        env = _default_environment(temp_root)
        _run_subprocess_json(
            [
                str(args.mne_features_python),
                str(Path(__file__).resolve()),
                "--internal-mne-extract",
                str(payload_path),
            ],
            env=env,
        )

        coco_X, coco_names, channel_names = _load_feature_matrix(coco_npz)
        mne_X, mne_names, _ = _load_feature_matrix(mne_npz)

        rows = []
        rows.extend(
            _compare_scalar_features(
                coco_X=coco_X,
                coco_names=coco_names,
                mne_X=mne_X,
                mne_names=mne_names,
                channel_names=channel_names,
                mappings=EXACT_SCALAR_MAPPINGS,
                relationship="exact_overlap",
            )
        )
        rows.extend(
            _compare_scalar_features(
                coco_X=coco_X,
                coco_names=coco_names,
                mne_X=mne_X,
                mne_names=mne_names,
                channel_names=channel_names,
                mappings=APPROX_SCALAR_MAPPINGS,
                relationship="approx_overlap",
            )
        )
        rows.extend(
            _compare_band_powers(
                coco_X=coco_X,
                coco_names=coco_names,
                mne_X=mne_X,
                mne_names=mne_names,
                channel_names=channel_names,
            )
        )

        _write_csv(rows, output_dir / "numeric_agreement.csv")
        _write_json(rows, output_dir / "numeric_agreement.json")
        (output_dir / "numeric_agreement_summary.md").write_text(
            render_numeric_summary(rows),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
