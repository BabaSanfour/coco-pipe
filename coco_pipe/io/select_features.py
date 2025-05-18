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

      - covariates: extra columns like age, sex, clinical scores
      - spatial_units: sensors/regions or groups thereof
      - feature_names: e.g. 'alpha', 'beta', etc.
      - row_filter: {"column": col, "values": [...]}

    Assumes brain-feature columns are named "<spatial_unit>_<feature_name>".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to select features from
    target_columns : Union[str, List[str]]
        Target variable(s) to predict
    covariates : Optional[List[str]]
        Covariates to include in the model, e.g. age, sex, clinical scores, behavioral metrics
    spatial_units : Optional[Union[str, List[str], Dict[str, List[str]]]]
        List of brain regions, sensors, or groups of regions/sensors, e.g. ["left_frontal", "right_frontal", "T1", "C3"]
    feature_names : Union[str, List[str]]
        Names of features computed, e.g. ["alpha", "beta", "gamma", "entropy", "slope"]
    row_filter : Optional[dict]
        {"column": ..., "values": [...]} to subset rows, e.g. {"column": "subject", "values": ["sub-01", "sub-02"]}

    Returns
    -------
    X : DataFrame of selected features
    y : Series (or DataFrame if multiple targets)
    """
    # 1) optional row-filtering
    if row_filter:
        col, vals = row_filter["column"], row_filter["values"]
        df = df[df[col].isin(vals)]

    parts: List[pd.DataFrame] = []

    # 2) covariates
    if covariates:
        parts.append(df[covariates])

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

    # 5) collect "<spatial>_<feature>" columns
    spatial_unit_cols = [
        f"{s}_{f}"
        for s in spatial_units
        for f in feature_names
        if f"{s}_{f}" in all_cols
    ]
    if spatial_unit_cols:
        parts.append(df[spatial_unit_cols])

    # 6) ensure we have at least something
    if not parts:
        raise ValueError(
            "No features selected: both covariates and spatial_units yielded no columns."
        )

    X = pd.concat(parts, axis=1)

    # 7) select target(s)
    y = df[target_columns]
    return X, y
