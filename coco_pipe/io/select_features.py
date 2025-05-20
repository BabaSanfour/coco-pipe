"""
coco_pipe/io/select_features.py
----------------
Select features from a DataFrame.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict

def select_features(
    df: pd.DataFrame,
    target_columns: Union[str, List[str]],
    covariates: Optional[List[str]] = None,
    spatial_units: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
    feature_names: Union[str, List[str]] = "all",
    sep: str = "_",
    reverse: bool = False,
    row_filter: Optional[Union[dict, List[dict]]] = None,
) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Slice a DataFrame into X (features) and y (target) according to:

      - covariates: columns like age, sex, clinical scores, etc.
      - spatial_units: sensors/regions or groups thereof, e.g. "left_frontal", "right_frontal", "T1", "C3"
      - feature_names: e.g. 'alpha', 'beta', etc.
      - sep: separator between spatial unit and feature name (default '_')
      - reverse: if True, expects naming '<feature>{sep}<spatial_unit>' instead of '<spatial_unit>{sep}<feature>'
      - row_filter: {"column": col, "values": [...]} or list of such dicts

    Assumes brain-feature columns are named accordingly.

    Parameters
    ----------
        :df: pd.DataFrame, DataFrame containing the data to select features from
        :target_columns: Union[str, List[str]], Target variable(s) to predict
        :covariates: Optional[List[str]], Covariates to include in the model, e.g. age, sex, clinical scores, behavioral metrics
        :spatial_units: Optional[Union[str, List[str], Dict[str, List[str]]]], List of brain regions, sensors, or groups of regions/sensors, e.g. ["left_frontal", "right_frontal", "T1", "C3"]
        :feature_names: Union[str, List[str]], Names of features computed, e.g. ["alpha", "beta", "gamma", "entropy", "slope"]
        :sep: str, Separator between spatial unit and feature name (default '_')
        :reverse: bool, If True, expects naming '<feature>{sep}<spatial_unit>' instead of '<spatial_unit>{sep}<feature>'
        :row_filter: Optional[dict], {"column": ..., "values": [...]} to subset rows, e.g. {"column": "subject", "values": ["sub-01", "sub-02"]}
    Returns
    -------
        X: pd.DataFrame of selected features
        y: pd.Series or DataFrame of target(s)
    """
    # Map lowercase to actual column names for case-insensitive matching
    col_map = {col.lower(): col for col in df.columns}

    # 1) optional row-filtering
    if row_filter:
        filters = row_filter if isinstance(row_filter, list) else [row_filter]
        for filt in filters:
            col_l = filt["column"].lower()
            if col_l not in col_map:
                raise ValueError(f"Row filter column '{col_l}' not found in DataFrame")
            actual_col = col_map[col_l]
            vals = filt.get("values")
            op = filt.get("operator")
            if op == ">":
                df = df[df[actual_col] > vals]
            elif op == "<":
                df = df[df[actual_col] < vals]
            elif op == ">=":
                df = df[df[actual_col] >= vals]
            elif op == "<=":
                df = df[df[actual_col] <= vals]
            else:
                df = df[df[actual_col].isin(vals)]

    parts: List[pd.DataFrame] = []

    # 2) covariates
    if covariates:
        missing_covs = [cov for cov in covariates if cov.lower() not in col_map]
        if missing_covs:
            raise ValueError(f"Requested covariates not found in DataFrame: {missing_covs}")
        matched_covs = [col_map[cov.lower()] for cov in covariates]
        parts.append(df[matched_covs])

    all_cols = df.columns.tolist()

    # 3) spatial_units normalization
    if spatial_units == "all":
        # determine from column names
        spatial_units = sorted({
            cols.split(sep)[1] if reverse else cols.split(sep)[0]
            for cols in all_cols if sep in cols
        })
    elif spatial_units is None:
        spatial_units = []
    elif isinstance(spatial_units, dict):
        spatial_units = list(spatial_units.keys())

    # 4) feature_names normalization
    if feature_names == "all":
        feature_names = sorted({
            cols.split(sep)[0] if reverse else cols.split(sep)[1]
            for cols in all_cols if sep in cols
        })
    elif isinstance(feature_names, str):
        feature_names = [feature_names]

    # 5) collect feature columns
    if reverse:
        requested = [f"{feat}{sep}{su}".lower() for su in spatial_units for feat in feature_names]
    else:
        requested = [f"{su}{sep}{feat}".lower() for su in spatial_units for feat in feature_names]

    missing_cols = [col for col in requested if col not in col_map]
    if missing_cols:
        raise ValueError(f"Requested feature columns not found in DataFrame: {missing_cols}")

    feature_cols = [col_map[col] for col in requested]
    if feature_cols:
        parts.append(df[feature_cols])

    # 6) ensure features exist
    if not parts:
        raise ValueError(
            "No features selected: both covariates and spatial_units yielded no columns."
        )

    X = pd.concat(parts, axis=1)

    # 7) select target(s)
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    missing_targets = [t for t in target_columns if t.lower() not in col_map]
    if missing_targets:
        raise ValueError(f"Target columns not found in DataFrame: {missing_targets}")
    matched_targets = [col_map[t.lower()] for t in target_columns]
    y = df[matched_targets[0]] if len(matched_targets) == 1 else df[matched_targets]
    return X, y