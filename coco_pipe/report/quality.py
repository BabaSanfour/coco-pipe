"""
Data Quality Checks
===================

Functions for performing automated quality checks on data containers.
These checks are used by the Report engine to flag potential issues.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List, Union
import numpy as np
import pandas as pd

@dataclass
class CheckResult:
    """
    Result of a data quality check.
    
    Attributes
    ----------
    check_name : str
        Name of the check (e.g., "Missing Values").
    status : str
        "OK", "WARN", or "FAIL".
    message : str
        Human-readable description of the issue.
    severity : int
        0 (Info) to 10 (Critical).
    metric_name : str, optional
        Name of the metric evaluated (e.g., "missing_pct").
    metric_value : float, optional
        Value of the metric.
        
    Examples
    --------
    >>> res = CheckResult("Missingness", "FAIL", "Too many NaNs", 9)
    >>> res.is_issue
    True
    """
    check_name: str
    status: str
    message: str
    severity: int
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    
    @property
    def is_issue(self) -> bool:
        """Return True if status is WARN or FAIL."""
        return self.status in ["WARN", "FAIL"]

def check_missingness(df: Union[pd.DataFrame, np.ndarray], threshold_warn: float = 0.01, threshold_fail: float = 0.20) -> CheckResult:
    """
    Check for missing values (NaNs).
    
    Parameters
    ----------
    df : DataFrame or ndarray
        The data to check.
    threshold_warn : float
        Ratio of NaNs to trigger a warning. Default 0.01 (1%).
    threshold_fail : float
        Ratio of NaNs to trigger a failure. Default 0.20 (20%).
        
    Returns
    -------
    CheckResult
        Quality check result.
        
    Examples
    --------
    >>> data = np.array([1, 2, np.nan, 4])
    >>> check_missingness(data, threshold_warn=0.1)
    CheckResult(check_name='Missingness', status='FAIL', ...)
    """
    if isinstance(df, np.ndarray):
        missing_count = np.isnan(df).sum()
        total = df.size
    else:
        missing_count = df.isna().sum().sum()
        total = df.size
        
    if total == 0:
         return CheckResult("Missingness", "WARN", "Dataset is empty.", 5)

    pct = missing_count / total
    
    if pct > threshold_fail:
        return CheckResult("Missingness", "FAIL", f"Critical missing data ({pct:.1%}).", 9, "missing_ratio", pct)
    elif pct > threshold_warn:
        return CheckResult("Missingness", "WARN", f"High missing data ({pct:.1%}).", 4, "missing_ratio", pct)
    
    return CheckResult("Missingness", "OK", "Missing data within limits.", 0, "missing_ratio", pct)

def check_constant_columns(df: Union[pd.DataFrame, np.ndarray]) -> List[CheckResult]:
    """
    Check for columns/features with zero variance.
    
    Parameters
    ----------
    df : DataFrame or ndarray
        The data to check.
        
    Returns
    -------
    List[CheckResult]
        List of findings. Empty if no constant columns found.
        
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1,1,1], 'b': [1,2,3]})
    >>> check_constant_columns(df)
    [CheckResult(check_name='Constant Features', ...)]
    """
    results = []
    
    if isinstance(df, np.ndarray):
        # Allow checking columns of 2D array
        if df.ndim != 2: return []
        # Check std dev along axis 0
        stds = np.nanstd(df, axis=0)
        constant_indices = np.where(stds == 0)[0]
        if len(constant_indices) > 0:
            results.append(CheckResult("Constant Features", "WARN", f"Found {len(constant_indices)} constant features (zero variance).", 3))
    else:
        # Pandas
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0: return []
        
        # Check std == 0
        stds = df[numeric_cols].std()
        constant_cols = stds[stds == 0].index.tolist()
        
        if len(constant_cols) > 0:
             msg = f"Found {len(constant_cols)} constant columns: {', '.join(constant_cols[:3])}{'...' if len(constant_cols)>3 else ''}."
             results.append(CheckResult("Constant Features", "WARN", msg, 3))
             
    return results

def check_outliers_zscore(df: Union[pd.DataFrame, np.ndarray], sigma: float = 5.0) -> Optional[CheckResult]:
    """
    Check for extreme values (> sigma).
    Uses a simple global Z-score approach.
    
    Parameters
    ----------
    df : DataFrame or ndarray
        Data to check.
    sigma : float
        Z-score threshold. Default 5.0.
        
    Returns
    -------
    Optional[CheckResult]
        CheckResult if outliers found, else None.
    """
    # Simple global check for efficiency on report
    try:
        if isinstance(df, pd.DataFrame):
            vals = df.select_dtypes(include=np.number).values
        else:
            vals = df
            
        if vals.size == 0: return None
        
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        
        if std == 0: return None
        
        z_scores = np.abs((vals - mean) / std)
        max_z = np.nanmax(z_scores)
        
        if max_z > sigma:
            return CheckResult("Outliers", "WARN", f"Extreme values detected (Max Z-Score: {max_z:.1f}).", 4, "max_z", max_z)
            
    except Exception:
        pass
    
    return None

def check_flatline(signal: np.ndarray, threshold: float = 1e-10) -> CheckResult:
    """
    Check if signal is effectively dead (flatline).
    
    Parameters
    ----------
    signal : ndarray
        1D signal array or flattened data.
    threshold : float
        Standard deviation threshold. Default 1e-10.
        
    Returns
    -------
    CheckResult
        Result indicating if signal is flat.
    """
    try:
        std = np.nanstd(signal)
        if std < threshold:
            return CheckResult("Signal Quality", "FAIL", "Signal is flatlined (Zero or near-zero variance).", 8, "std_dev", std)
    except Exception:
        pass
    return CheckResult("Signal Quality", "OK", "Signal variance OK.", 0)
