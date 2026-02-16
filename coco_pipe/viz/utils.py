"""
Visualization Utilities
=======================

Shared utility functions for static (matplotlib) and interactive (plotly) visualizations.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def is_categorical(labels: Union[np.ndarray, list], max_unique_numeric: int = 20) -> bool:
    """
    Check if labels are categorical or continuous.
    
    Parameters
    ----------
    labels : array-like
        Array of labels.
    max_unique_numeric : int, default=20
        Maximum number of unique values for numeric data to be considered categorical.
        
    Returns
    -------
    bool
        True if categorical (string, bool, or few unique numeric values).
    """
    if labels is None:
        return False
        
    arr = np.array(labels)
    # Check for string/object/bool types
    if arr.dtype.kind in ("U", "S", "O", "b"):
        return True

    # If numeric, check unique count to detect discrete categories (e.g., classes 0, 1, 2)
    try:
        # Avoid issues with NaNs in unique count
        valid_mask = ~pd.isna(arr)
        if hasattr(valid_mask, "to_numpy"):
            valid_mask = valid_mask.to_numpy()
            
        n_unique = len(np.unique(arr[valid_mask]))
        if n_unique < max_unique_numeric:
            return True
    except Exception:
        pass
        
    return False


def filter_metrics(scores: Dict[str, Any]) -> Dict[str, float]:
    """
    Filter metrics dictionary to keep only scalar values suitable for plotting.
    
    Excludes 'n_iter_', 'n_components', and non-numeric values.
    
    Parameters
    ----------
    scores : dict
        Dictionary of metric names and values.
        
    Returns
    -------
    dict
        Filtered dictionary containing only scalar float/int values.
    """
    if not scores:
        return {}
        
    exclude_keys = {"n_iter_", "n_components"}
    
    filtered = {}
    for k, v in scores.items():
        if k in exclude_keys:
            continue
            
        # Check if value is scalar numeric
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            filtered[k] = float(v)
            
    return filtered


def prepare_dataframe(
    embedding: np.ndarray, 
    labels: Optional[np.ndarray] = None, 
    meta: Optional[Dict[str, Any]] = None,
    dimensions: int = 2
) -> pd.DataFrame:
    """
    Prepare a DataFrame for plotting from embedding and metadata.
    
    Parameters
    ----------
    embedding : np.ndarray
        (N, D) array of embedding coordinates.
    labels : np.ndarray, optional
        Primary labels for coloring.
    meta : dict, optional
        Additional metadata columns.
    dimensions : int
        Number of dimensions to include (2 or 3).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'x', 'y' (and 'z'), 'Default Label', and metadata columns.
    """
    n_points = embedding.shape[0]
    
    data = {
        "x": embedding[:, 0],
        "y": embedding[:, 1]
    }
    
    if dimensions == 3 and embedding.shape[1] > 2:
        data["z"] = embedding[:, 2]
        
    if labels is not None:
        # Convert to string if categorical for clearer discrete coloring in Plotly
        if is_categorical(labels):
            data["Default Label"] = np.array(labels).astype(str)
        else:
            data["Default Label"] = labels
            
    if meta:
        for k, v in meta.items():
            # Only add if length matches
            if hasattr(v, "__len__") and len(v) == n_points:
                data[k] = v
                
    return pd.DataFrame(data)
