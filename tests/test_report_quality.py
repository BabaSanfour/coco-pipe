import numpy as np
import pandas as pd

from coco_pipe.report.core import Section
from coco_pipe.report.quality import (
    CheckResult,
    check_constant_columns,
    check_flatline,
    check_missingness,
    check_outliers_zscore,
)


def test_check_missingness():
    # 1. NumPy Array
    data = np.random.randn(100)
    data[:10] = np.nan
    res = check_missingness(data, threshold_warn=0.01)
    assert res.status == "WARN"
    assert res.metric_value == 0.1

    # Empty data
    res_empty = check_missingness(np.array([]))
    assert res_empty.status == "WARN"
    assert "empty" in res_empty.message

    # 2. Pandas DataFrame
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, 6]})
    res_df = check_missingness(df, threshold_warn=0.1)
    assert res_df.status == "WARN"
    assert np.isclose(res_df.metric_value, 1 / 6)


def test_check_constant_columns():
    # 1. Pandas
    df = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": ["x", "y", "z"]})
    results = check_constant_columns(df)
    assert len(results) == 1
    assert "A" in results[0].message

    # No numeric cols
    assert check_constant_columns(pd.DataFrame({"A": ["x", "y"]})) == []

    # 2. NumPy 2D
    arr = np.array([[1, 2], [1, 3], [1, 4]])
    results_arr = check_constant_columns(arr)
    assert len(results_arr) == 1
    assert "1 constant features" in results_arr[0].message

    # NumPy 1D (unsupported for constant cols check)
    assert check_constant_columns(np.array([1, 1, 1])) == []


def test_check_outliers_zscore():
    # 1. NumPy
    data = np.array([1, 1.1, 1.2, 1.0, 1.1, 1000.0])  # Clear outlier
    res = check_outliers_zscore(data, sigma=2.0)
    assert res is not None
    assert res.status == "WARN"
    assert "Extreme values" in res.message

    # Empty
    assert check_outliers_zscore(np.array([])) is None

    # Zero std
    assert check_outliers_zscore(np.array([1, 1, 1])) is None

    # 2. Pandas
    df = pd.DataFrame({"A": [1, 1.1, 1.2, 1000.0], "B": [0, 0, 0, 0]})
    res_df = check_outliers_zscore(df, sigma=2.0)
    assert res_df is not None
    assert res_df.status == "WARN"


def test_check_flatline():
    # Flatline
    sig = np.zeros(100)
    res = check_flatline(sig)
    assert res.status == "FAIL"

    # OK
    sig_ok = np.random.randn(100)
    res_ok = check_flatline(sig_ok)
    assert res_ok.status == "OK"

    # Error path (e.g. wrong type)
    res_err = check_flatline("not an array")
    assert res_err.status == "OK"  # Fallback


def test_section_status_update():
    sec = Section("Quality Test")
    assert sec.status == "OK"

    # Add WARN
    res_warn = CheckResult("Test", "WARN", "Warning", 5)
    sec.add_finding(res_warn)
    assert sec.status == "WARN"

    # Add FAIL (should upgrade)
    res_fail = CheckResult("Test", "FAIL", "Failure", 9)
    sec.add_finding(res_fail)
    assert sec.status == "FAIL"

    # Add WARN (should stay FAIL)
    sec.add_finding(res_warn)
    assert sec.status == "FAIL"


def test_findings_serialization():
    sec = Section("Serialization")
    sec.add_finding(CheckResult("Test", "WARN", "Msg", 5))

    html = sec.render()
    # Check if finding message is present in HTML (via Jinja loop)
    assert "Msg" in html
    assert "⚠️" in html
