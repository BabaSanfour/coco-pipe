import json

import pandas as pd

from coco_pipe.report._utils import _config_element, _table_from_mapping
from coco_pipe.report.elements import CodeBlockElement, TableElement


def test_table_from_mapping():
    mapping = {"a": 1, "b": "string", 42: "number_key"}
    element = _table_from_mapping(mapping, title="Test Table")

    assert isinstance(element, TableElement)
    assert element.title == "Test Table"

    df = element.data
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Key", "Value"]
    assert len(df) == 3

    # Keys should be converted to strings
    keys = df["Key"].tolist()
    assert keys == ["a", "b", "42"]

    values = df["Value"].tolist()
    assert values == [1, "string", "number_key"]


def test_table_from_mapping_empty():
    element = _table_from_mapping({}, title="Empty")
    df = element.data
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Key", "Value"]
    assert len(df) == 0


def test_config_element_flat():
    config = {"param1": 10, "param2": "value"}
    element = _config_element(config, title="Flat Config")

    assert isinstance(element, TableElement)
    assert element.title == "Flat Config"
    assert len(element.data) == 2


def test_config_element_nested_dict():
    config = {"param1": 10, "nested": {"a": 1}}
    element = _config_element(config, title="Nested Dict")

    assert isinstance(element, CodeBlockElement)
    assert element.title == "Nested Dict"
    assert element.language == "json"

    parsed = json.loads(element._code)
    assert parsed == config


def test_config_element_nested_list():
    config = {"param1": 10, "nested": [1, 2, 3]}
    element = _config_element(config, title="Nested List")

    assert isinstance(element, CodeBlockElement)

    parsed = json.loads(element._code)
    assert parsed == config


def test_config_element_nested_set():
    config = {"param1": 10, "nested": {1, 2, 3}}
    element = _config_element(config, title="Nested Set")

    assert isinstance(element, CodeBlockElement)

    parsed = json.loads(element._code)
    # The set is converted to string using default=str
    assert parsed["param1"] == 10
    assert isinstance(parsed["nested"], str)
    assert "{" in parsed["nested"]


def test_config_element_string_is_not_nested():
    config = {"param1": "this is a string, which is technically a sequence"}
    element = _config_element(config, title="String Config")

    # It should remain a TableElement, not fall back to JSON
    assert isinstance(element, TableElement)
    assert element.data["Key"].iloc[0] == "param1"
    assert (
        element.data["Value"].iloc[0]
        == "this is a string, which is technically a sequence"
    )


def test_report_lazy_getattr():
    import pytest

    import coco_pipe.report as report

    # 1. Valid lazy getattr
    assert report.Report is not None
    assert report.Section is not None

    # 2. Invalid attribute raises AttributeError
    with pytest.raises(AttributeError, match="has no attribute InvalidAttr"):
        _ = report.InvalidAttr
