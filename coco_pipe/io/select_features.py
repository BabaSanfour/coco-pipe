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
    row_filter: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Slice a DataFrame into X (features) and y (target) according to:

      - covariates: columns like age, sex, clinical scores, etc.
      - spatial_units: sensors/regions or groups thereof, e.g. "left_frontal", "right_frontal", "T1", "C3"
      - feature_names: e.g. 'alpha', 'beta', etc.
      - row_filter: {"column": col, "values": [...]}

    Assumes brain-feature columns are named "<spatial_unit>_<feature_name>".

    Parameters
    ----------
        :df: pd.DataFrame, DataFrame containing the data to select features from
        :target_columns: Union[str, List[str]], Target variable(s) to predict
        :covariates: Optional[List[str]], Covariates to include in the model, e.g. age, sex, clinical scores, behavioral metrics
        :spatial_units: Optional[Union[str, List[str], Dict[str, List[str]]]], List of brain regions, sensors, or groups of regions/sensors, e.g. ["left_frontal", "right_frontal", "T1", "C3"]
        :feature_names: Union[str, List[str]], Names of features computed, e.g. ["alpha", "beta", "gamma", "entropy", "slope"]
        :row_filter: Optional[dict], {"column": ..., "values": [...]} to subset rows, e.g. {"column": "subject", "values": ["sub-01", "sub-02"]}

    Returns
    -------
        :X: pd.DataFrame, DataFrame of selected features
        :y: Union[pd.Series, pd.DataFrame], Target variable(s) to predict
    """
    col_map = {col.lower(): col for col in df.columns}

    # 1) optional row-filtering
    # Handle multiple row filters
    if row_filter:
        # Convert to list if single filter provided
        filters = row_filter if isinstance(row_filter, list) else [row_filter]
        
        # Apply each filter sequentially
        for filt in filters:
            col = filt["column"].lower()
            vals = filt["values"] 
            operator = filt.get("operator")
            
            if col not in col_map:
                raise ValueError(f"Row filter column '{col}' not found in DataFrame")
                
            if operator == ">":
                df = df[df[col_map[col]] > vals]
            elif operator == "<":
                df = df[df[col_map[col]] < vals]
            elif operator == ">=":
                df = df[df[col_map[col]] >= vals]
            elif operator == "<=":
                df = df[df[col_map[col]] <= vals]
            else:
                df = df[df[col_map[col]].isin(vals)]

    parts: List[pd.DataFrame] = []

    # 2) covariates
    if covariates:
        # Check for missing covariates
        missing_covs = [cov for cov in covariates if cov.lower() not in col_map]
        if missing_covs:
            raise ValueError(f"Requested covariates not found in DataFrame: {missing_covs}")
        # Match covariates case-insensitively
        matched_covs = [col_map[cov.lower()] for cov in covariates]
        parts.append(df[matched_covs])

    all_cols = df.columns.tolist()

    # 3) spatial_units normalization
    if spatial_units == "all":
        # every unique prefix before the underscore
        spatial_units = sorted({c.split("_")[0] for c in all_cols if "_" in c})
    elif spatial_units is None:
        spatial_units = []
    elif isinstance(spatial_units, dict):
        # user gave named groups; flatten to list of keys
        spatial_units = list(spatial_units.keys())
    # else assume list of spatial_units

    # 4) feature_names normalization
    if feature_names == "all":
        feature_names = sorted({c.split("_")[1] for c in all_cols if "_" in c})
    elif isinstance(feature_names, str):
        feature_names = [feature_names]
    # else assume list of feature_names

    # 5) collect "<spatial>_<feature>" columns using case-insensitive matching
    requested_cols = [f"{s}_{f}".lower() for s in spatial_units for f in feature_names]
    missing_cols = [col for col in requested_cols if col not in col_map]
    if missing_cols:
        raise ValueError(f"Requested feature columns not found in DataFrame: {missing_cols}")
        
    spatial_unit_cols = [col_map[col] for col in requested_cols]
    if spatial_unit_cols:
        parts.append(df[spatial_unit_cols])

    # 6) ensure we have at least something
    if not parts:
        raise ValueError(
            "No features selected: both covariates and spatial_units yielded no columns."
        )

    X = pd.concat(parts, axis=1)

    # 7) select target(s) with case-insensitive matching
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    
    # Check for missing target columns
    missing_targets = [t for t in target_columns if t.lower() not in col_map]
    if missing_targets:
        raise ValueError(f"Target columns not found in DataFrame: {missing_targets}")
        
    matched_targets = [col_map[t.lower()] for t in target_columns]
    y = df[matched_targets[0] if len(matched_targets) == 1 else matched_targets]
    return X, y
