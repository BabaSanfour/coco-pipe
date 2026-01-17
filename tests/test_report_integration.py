"""
Integration Tests for Report Factory Functions
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io.structures import DataContainer
from coco_pipe.report import (
    from_bids,
    from_container,
    from_embeddings,
    from_reductions,
    from_tabular,
)

# --- Mocks & Fixtures ---


@pytest.fixture
def mock_container():
    return DataContainer(
        X=np.zeros((10, 5)),
        dims=("obs", "feature"),
        ids=np.arange(10).astype(str),
        coords={"feature": ["f1", "f2", "f3", "f4", "f5"]},
    )


@pytest.fixture
def mock_reducer():
    red = MagicMock()
    # Important: Set to real arrays, otherwise Plotly validation fails when it sees a MagicMock
    red.embedding_ = np.random.randn(10, 2)
    red.loss_history_ = [10, 5, 2]
    red.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
    return red
    # red.__class__.__name__ works automatically for MagicMock if spec'd,
    # but let's just let the API handle the mock name.
    return red


# --- Tests ---


def test_from_container(mock_container):
    rep = from_container(mock_container, title="Test Report")
    html = rep.render()

    assert "Test Report" in html
    assert "Data Overview" in html
    assert "Data Overview" in html
    assert "Dimensions" in html
    assert "Raw Data Inspector" in html
    assert "🔍" in html


@patch("coco_pipe.report.api.BIDSDataset")
def test_from_bids(MockBIDSDataset, mock_container, tmp_path):
    # Setup Mock
    instance = MockBIDSDataset.return_value
    instance.load.return_value = mock_container

    # Call
    rep = from_bids(root=tmp_path, task="rest")

    # Verify
    MockBIDSDataset.assert_called_with(root=tmp_path, task="rest")
    instance.load.assert_called_once()

    html = rep.render()
    assert "BIDS Report: rest" in html
    html = rep.render()
    assert "BIDS Report: rest" in html
    assert "Data Overview" in html

    assert "source" in html
    assert "BIDS" in html
    assert "BIDS" in html
    # assert f"{tmp_path}" in html # Flaky on Windows due to backslashes
    assert f"{tmp_path.name}" in html


def test_from_tabular(tmp_path):
    # We can actually run this one with a real file since it's easy
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False, sep="\t")

    rep = from_tabular(p)
    html = rep.render()

    assert f"Tabular Report: {p.name}" in html
    assert "Data Overview" in html


@patch("coco_pipe.report.api.EmbeddingDataset")
def test_from_embeddings(MockEmbDataset, mock_container, tmp_path):
    instance = MockEmbDataset.return_value
    instance.load.return_value = mock_container

    rep = from_embeddings(tmp_path)

    MockEmbDataset.assert_called_with(path=tmp_path)
    html = rep.render()
    assert "Embedding Report" in html


def test_from_reductions(mock_reducer, mock_container):
    reductions = [mock_reducer, mock_reducer]  # 2 reducers

    rep = from_reductions(reductions, container=mock_container)
    html = rep.render()

    assert "DimReduction Comparison" in html
    assert "Data Overview" in html
    # Should have 2 reduction sections
    assert html.count("📉") >= 2
    # Should have embedding plots
    assert html.count("lazy-plot") >= 2
