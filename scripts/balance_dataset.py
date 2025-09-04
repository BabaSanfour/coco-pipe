#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd

from coco_pipe.io import load, balance_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("balance_dataset")


def main():
    parser = argparse.ArgumentParser(description="Balance imbalanced target classes with optional covariate-aware sampling.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV/TSV/Excel file")
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument("--output", "-o", required=True, help="Path to write balanced CSV")
    parser.add_argument("--strategy", choices=["undersample", "oversample", "auto"], default="undersample", help="Sampling strategy")
    parser.add_argument("--covariates", "-c", nargs="*", default=None, help="Covariates to preserve distribution across (e.g., age sex)")
    parser.add_argument(
        "--grid-balance",
        nargs="*",
        default=None,
        help=(
            "Subset of covariates across which to equalize counts jointly with the target "
            "within the remaining covariates. For example: --covariates age sex --grid-balance sex "
            "will, within each age bin, enforce equal counts across (sex x class)."
        ),
    )
    parser.add_argument(
        "--require-full-grid",
        action="store_true",
        help=(
            "With --grid-balance, only keep strata where all (grid x class) combinations exist. "
            "Otherwise, missing combinations are ignored."
        ),
    )
    parser.add_argument("--qbins", type=int, default=5, help="Number of quantile bins for numeric covariates")
    parser.add_argument("--binning", choices=["quantile", "uniform"], default="quantile", help="Binning method for numeric covariates")
    parser.add_argument("--sep", default=None, help="Separator override for CSV/TSV (auto by extension)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (if using .xlsx/.xls)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prefer-clean", action="store_true", help="Prioritize rows with fewer NaN/inf/0 values when sampling")
    args = parser.parse_args()

    # Load dataframe (uses coco_pipe.io.load for consistency)
    df = load(
        type="tabular",
        data_path=args.input,
        sheet_name=args.sheet,
        sep=args.sep,
    )
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Expected a DataFrame from load().")

    logger.info("Initial class distribution for '%s':\n%s", args.target, df[args.target].value_counts())
    if args.covariates:
        logger.info("Using covariates for stratification: %s", args.covariates)

    balanced = balance_dataset(
        df=df,
        target=args.target,
        strategy=args.strategy,
        covariates=args.covariates,
        n_bins=args.qbins,
        binning=args.binning,
        random_state=args.seed,
        prefer_clean_rows=args.prefer_clean,
        grid_balance=args.grid_balance,
        require_full_grid=args.require_full_grid,
    )

    logger.info("Balanced class distribution for '%s':\n%s", args.target, balanced[args.target].value_counts())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out_path, index=False)
    logger.info("Saved balanced dataset to %s (rows: %d, cols: %d)", out_path, len(balanced), len(balanced.columns))


if __name__ == "__main__":
    main()
