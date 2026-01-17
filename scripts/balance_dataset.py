#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# coco_pipe imports

# balance_dataset is now a method of DataContainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("balance_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Balance imbalanced target classes with optional covariate-aware sampling."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV/TSV/Excel file"
    )
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument(
        "--output", "-o", required=True, help="Path to write balanced CSV"
    )
    parser.add_argument(
        "--strategy",
        choices=["undersample", "oversample", "auto"],
        default="undersample",
        help="Sampling strategy",
    )
    parser.add_argument(
        "--covariates",
        "-c",
        nargs="*",
        default=None,
        help="Covariates to preserve distribution across (e.g., age sex)",
    )
    parser.add_argument(
        "--grid-balance",
        nargs="*",
        default=None,
        help="Subset of covariates for grid equalization.",
    )
    parser.add_argument(
        "--require-full-grid", action="store_true", help="Require full grid presence."
    )
    parser.add_argument("--qbins", type=int, default=5, help="Number of quantile bins")
    parser.add_argument(
        "--binning",
        choices=["quantile", "uniform"],
        default="quantile",
        help="Binning method",
    )
    parser.add_argument("--sep", default=None, help="Separator override")
    parser.add_argument("--sheet", default=None, help="Excel sheet name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--prefer-clean",
        action="store_true",
        help="Prioritize rows with fewer NaN/inf/0 values",
    )
    args = parser.parse_args()

    # 1. Load DataContainer
    # Note: tabular load puts target in y if known, but here we specify target in CLI.
    # If generic load is used without target_col, target is a column in X (or feature coord).
    # load() uses TabularDataset defaults which might not know target_col yet unless passed.
    # load() signature in generic entry point needs checking.
    # Actually, simpler to use TabularDataset directly if we need specific kwargs like target_col.
    # But for a script, let's stick to generic load and assume target is in columns.

    # However, TabularDataset puts everything in X/coords unless target_col is specified.
    # If target is in X, we balance by 'feature' name.

    # We'll use TabularDataset explicitly to ensure we handle target correctly
    from coco_pipe.io.dataset import TabularDataset

    logger.info(f"Loading {args.input}...")
    ds_loader = TabularDataset(
        path=args.input,
        target_col=args.target,  # Extract target to y
        sep=args.sep if args.sep else "\t",
        sheet_name=args.sheet if args.sheet else 0,
    )
    container = ds_loader.load()

    logger.info("Initial Shape: %s", container.shape)
    if container.y is not None:
        uniq, counts = np.unique(container.y, return_counts=True)
        logger.info(
            "Class distribution for '%s': %s", args.target, dict(zip(uniq, counts))
        )
    else:
        raise RuntimeError(f"Target column '{args.target}' not found or not extracted.")

    if args.covariates:
        logger.info("Using covariates: %s", args.covariates)

    # 2. Balance
    balanced_container = container.balance(
        target="y",  # Since we extracted it
        strategy=args.strategy,
        covariates=args.covariates,
        random_state=args.seed,
        n_bins=args.qbins,
        binning=args.binning,
        prefer_clean_rows=args.prefer_clean,
        grid_balance=args.grid_balance,
        require_full_grid=args.require_full_grid,
    )

    # 3. Save / Export
    # Reconstruct DataFrame
    # X columns
    feats = balanced_container.coords.get("feature", [])
    if len(feats) != balanced_container.X.shape[1]:
        feats = [f"feat_{i}" for i in range(balanced_container.X.shape[1])]

    df_out = pd.DataFrame(balanced_container.X, columns=feats)

    # Add target
    df_out[args.target] = balanced_container.y

    # Add other coords (covariates) if they match obs length
    for k, v in balanced_container.coords.items():
        if k != "feature" and len(v) == len(df_out):
            # Be careful not to overwrite target if name conflict (though target is usually separate in container)
            if k != args.target:
                df_out[k] = v

    logger.info("Balanced Shape: %s", df_out.shape)
    if balanced_container.y is not None:
        uniq, counts = np.unique(balanced_container.y, return_counts=True)
        logger.info("Balanced counts: %s", dict(zip(uniq, counts)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine save format
    if out_path.suffix == ".csv":
        df_out.to_csv(out_path, index=False)
    elif out_path.suffix in [".xlsx", ".xls"]:
        df_out.to_excel(out_path, index=False)
    else:
        df_out.to_csv(out_path, sep="\t", index=False)

    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
