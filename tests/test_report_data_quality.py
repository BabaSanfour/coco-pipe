import numpy as np
import pandas as pd

from coco_pipe.report.data_quality import (
    CheckResult,
    _numeric_values,
    check_constant_columns,
    check_flatline,
    check_missingness,
    check_outliers_zscore,
)


def test_check_result_is_issue():
    ok = CheckResult("test", "OK", "msg", 0)
    warn = CheckResult("test", "WARN", "msg", 5)
    fail = CheckResult("test", "FAIL", "msg", 10)
    assert not ok.is_issue
    assert warn.is_issue
    assert fail.is_issue


def test_numeric_values_dataframe():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [3.5, np.nan]})
    arr = _numeric_values(df)
    assert arr.shape == (2, 2)
    assert np.isnan(arr[1, 1])


def test_numeric_values_ndarray():
    arr = np.array([1, 2, 3])
    res = _numeric_values(arr)
    assert res.dtype == float
    assert res.tolist() == [1.0, 2.0, 3.0]

    # Mixed object array with strings
    obj_arr = np.array(["a", "b", "c"], dtype=object)
    res_obj = _numeric_values(obj_arr)
    assert res_obj.size == 0


def test_check_missingness_empty():
    res = check_missingness(np.array([]))
    assert res.status == "WARN"
    assert "empty" in res.message


def test_check_missingness_ndarray():
    arr = np.array([1, np.nan, 3, np.nan])
    res = check_missingness(arr, threshold_warn=0.1, threshold_fail=0.4)
    assert res.status == "FAIL"
    assert res.metric_value == 0.5


def test_check_missingness_dataframe():
    df = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    res = check_missingness(df, threshold_warn=0.2, threshold_fail=0.5)
    assert res.status == "WARN"  # 2/6 = 33% > 20%
    assert np.isclose(res.metric_value, 1 / 3)


def test_check_constant_columns_empty():
    assert check_constant_columns(np.array([])) == []
    assert check_constant_columns(pd.DataFrame()) == []


def test_check_constant_columns_ndarray():
    # 2D array
    arr = np.array([[1, 2, 3], [1, 4, 3], [1, 5, 3]])
    res = check_constant_columns(arr)
    assert len(res) == 1
    assert "2 constant features" in res[0].message


def test_check_constant_columns_dataframe():
    df = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": ["x", "x", "x"]})
    res = check_constant_columns(df)
    assert len(res) == 1
    assert "constant columns: A" in res[0].message


def test_check_outliers_zscore_empty_or_nan():
    assert check_outliers_zscore(np.array([])) is None
    assert check_outliers_zscore(np.array([np.nan, np.nan])) is None


def test_check_outliers_zscore_no_variance():
    assert check_outliers_zscore(np.array([5, 5, 5])) is None


def test_check_outliers_zscore_detection():
    arr = np.array([1.0] * 50 + [100.0])
    res = check_outliers_zscore(arr, sigma=3.0)
    assert res is not None
    assert res.status == "WARN"
    assert res.metric_value > 3.0


def test_check_flatline_empty():
    res = check_flatline(np.array([]))
    assert res.status == "WARN"

    res_non_num = check_flatline(np.array(["a", "b"]))
    assert res_non_num.status == "OK"  # Not numeric, handled gracefully


def test_check_flatline_detection():
    # Standard dev is zero
    arr = np.array([5, 5, 5])
    res = check_flatline(arr, threshold=1e-5)
    assert res.status == "FAIL"
    assert res.metric_value == 0

    arr2 = np.array([1, 2, 3])
    res2 = check_flatline(arr2, threshold=1e-5)
    assert res2.status == "OK"


def test_check_constant_columns_strings_only():
    df = pd.DataFrame({"A": ["x", "y"], "B": ["a", "b"]})
    res = check_constant_columns(df)
    assert res == []

    arr = np.array([["a", "b"], ["c", "d"]], dtype=object)
    res_arr = check_constant_columns(arr)
    assert res_arr == []


def test_check_constant_columns_many():
    df = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": [3, 3], "D": [4, 4], "E": [5, 6]})
    res = check_constant_columns(df)
    assert len(res) == 1
    assert "..." in res[0].message


def test_check_constant_columns_none():
    df = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
    assert check_constant_columns(df) == []


def test_check_outliers_zscore_all_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    assert check_outliers_zscore(arr) is None
