import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from coco_pipe.io.tabular import load_tabular

def create_csv(tmp_path, df, sep=',', header=True, index=False):
    path = tmp_path / f"data{sep if sep != ',' else ''}.csv"
    df.to_csv(path, sep=sep, header=header, index=index)
    return path


def create_tsv(tmp_path, df, header=True, index=False):
    path = tmp_path / "data.tsv"
    df.to_csv(path, sep='\t', header=header, index=index)
    return path


def create_excel(tmp_path, df_dict):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path) as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def test_csv_load_no_target(tmp_path):
    df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = load_tabular(path)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_tsv_auto_detect(tmp_path):
    df = pd.DataFrame({'x':[7,8], 'y':[9,10]})
    path = create_tsv(tmp_path, df)
    result = load_tabular(path)
    pd.testing.assert_frame_equal(result, df)


def test_custom_sep(tmp_path):
    df = pd.DataFrame({'col1':[1,2], 'col2':[3,4]})
    path = tmp_path / "data.csv"
    df.to_csv(path, sep=';', index=False)
    result = load_tabular(path, sep=';')
    pd.testing.assert_frame_equal(result, df)


def test_excel_single_sheet(tmp_path):
    df = pd.DataFrame({'p':[0,1], 'q':[2,3]})
    path = create_excel(tmp_path, {'Sheet1': df})
    result = load_tabular(path)
    pd.testing.assert_frame_equal(result, df)


def test_excel_specific_sheet(tmp_path):
    df1 = pd.DataFrame({'m':[5,6], 'n':[7,8]})
    df2 = pd.DataFrame({'m':[9,10], 'n':[11,12]})
    path = create_excel(tmp_path, {'one': df1, 'two': df2})
    result1 = load_tabular(path, sheet_name='one')
    result2 = load_tabular(path, sheet_name='two')
    pd.testing.assert_frame_equal(result1, df1)
    pd.testing.assert_frame_equal(result2, df2)


def test_index_col_int(tmp_path):
    df = pd.DataFrame({'i':[1,2],'j':[3,4]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    # treat first column as index
    result = load_tabular(path, index_col=0)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == df['i'].tolist()
    assert 'i' not in result.columns


def test_header_none(tmp_path):
    arr = np.array([[10,20],[30,40]])
    df = pd.DataFrame(arr)
    path = tmp_path / "data.csv"
    df.to_csv(path, header=False, index=False)
    result = load_tabular(path, header=None)
    # columns should be integers
    assert list(result.columns) == [0,1]
    pd.testing.assert_frame_equal(result, df)


def test_target_cols_str(tmp_path):
    df = pd.DataFrame({'f1':[1,2],'target':[0,1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    X, y = load_tabular(path, target_cols='target')
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)
    assert 'target' not in X.columns
    pd.testing.assert_series_equal(y['target'], pd.Series([0,1], name='target'))


def test_target_cols_list(tmp_path):
    df = pd.DataFrame({'t1':[1,2],'t2':[3,4],'f':[5,6]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    X, y = load_tabular(path, target_cols=['t1','t2'])
    assert set(y.columns) == {'t1','t2'}
    assert 'f' in X.columns


def test_missing_target_raises(tmp_path):
    df = pd.DataFrame({'a':[1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Target column\(s\) not found in data: \['missing'\]"):
        load_tabular(path, target_cols='missing')
