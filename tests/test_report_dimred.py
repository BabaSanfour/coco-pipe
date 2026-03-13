"""
Tests for Dim-Red Components
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from coco_pipe.report.core import PlotlyElement, Report
from coco_pipe.viz.plotly_utils import plot_embedding_interactive, plot_metric_details


class MockReducer:
    """Mock DimReduction object."""

    def __init__(self):
        self.loss_history_ = [10, 5, 2, 1]
        self.explained_variance_ratio_ = np.array([0.5, 0.3, 0.1])
        self.capabilities = {
            "supported_diagnostics": [
                "loss_history_",
                "explained_variance_ratio_",
            ],
            "supported_metadata": [],
        }

    def get_diagnostics(self):
        return {
            "loss_history_": self.loss_history_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
        }

    def get_quality_metadata(self):
        return {}

    def get_summary(self):
        return {
            "method": "MockReducer",
            "metrics": {},
            "metric_records": [],
            "quality_metadata": self.get_quality_metadata(),
            "diagnostics": self.get_diagnostics(),
            "interpretation": {},
            "interpretation_records": [],
            "capabilities": self.capabilities,
        }


def test_plotly_element_rendering():
    fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
    el = PlotlyElement(fig)
    html = el.render()

    # Check for lazy loading structure
    assert "lazy-plot" in html
    assert "data-figure" in html
    assert "&quot;data&quot;" in html  # JSON encoded


def test_plot_embedding_interactive_logic():
    emb = np.random.randn(50, 2)
    labels = np.random.randint(0, 2, 50)
    metadata = {"Class": ["X"] * 25 + ["Y"] * 25, "Score": np.random.rand(50)}

    # 1. Basic call (backward compatibility)
    fig_basic = plot_embedding_interactive(emb, labels=labels, title="Basic")
    assert isinstance(fig_basic, go.Figure)
    assert fig_basic.layout.updatemenus == ()  # No dropdowns

    # 2. Advanced call (with metadata -> Dropdowns)
    fig_adv = plot_embedding_interactive(
        emb, labels=labels, metadata=metadata, title="Adv"
    )
    assert isinstance(fig_adv, go.Figure)
    assert len(fig_adv.layout.updatemenus) > 0  # Should have dropdowns


def test_plot_metric_details():
    df = pd.DataFrame(
        {
            "Method": ["PCA", "UMAP"],
            "Trustworthiness": [0.8, 0.9],
            "Continuity": [0.7, 0.95],
        }
    ).set_index("Method")

    fig = plot_metric_details(df)
    assert isinstance(fig, go.Figure)
    # 2 methods = 2 groups of bars (traces) or 2 traces depending on impl
    # Our impl adds One Trace per Method
    assert len(fig.data) == 2


def test_report_add_reduction_logic():
    rep = Report("DimRed Test")
    reducer = MockReducer()
    embedding = np.random.randn(100, 2)
    metadata = {
        "Group": ["A"] * 50 + ["B"] * 50,
        "Value": np.random.rand(100),
    }

    rep.add_reduction(reducer, name="MockPCA", X_emb=embedding, metadata=metadata)

    html = rep.render()

    assert "MockPCA" in html
    assert "📉" in html

    assert html.count("lazy-plot") >= 3


def test_report_add_comparison():
    rep = Report("Comparison Test")

    metrics_df = pd.DataFrame(
        {
            "Method": ["PCA", "ISO"],
            "Trustworthiness": [0.8, 0.6],
            "Continuity": [0.7, 0.5],
        }
    ).set_index("Method")

    rep.add_comparison(metrics_df)

    html = rep.render()

    # Check for section title and icon
    assert "Method Comparison" in html
    assert "📊" in html
    # Check for plots (Radar + Bar)
    # Check for Table title
    assert "Quality Metrics" in html


def test_report_add_reduction_requires_summary_contract():
    """Report.add_reduction should require the explicit reducer summary contract."""
    rep = Report("Safe Access Test")

    class BrokenReducer:
        def __init__(self):
            self.embedding_ = np.random.randn(10, 2)

        @property
        def loss_history_(self):
            raise RuntimeError("Broken loss history")

    reducer = BrokenReducer()
    with pytest.raises(TypeError, match="must implement get_summary"):
        rep.add_reduction(reducer, name="BrokenMethod")
