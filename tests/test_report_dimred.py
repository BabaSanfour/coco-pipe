"""
Tests for Phase 3: Dim-Red Components
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from coco_pipe.report.core import PlotlyElement, Report
from coco_pipe.viz.plotly_utils import (
    plot_embedding_interactive,
    plot_metric_details,
    plot_radar_comparison,
)


class MockReducer:
    """Mock DimReduction object."""

    def __init__(self):
        self.embedding_ = np.random.randn(100, 2)
        self.loss_history_ = [10, 5, 2, 1]
        self.explained_variance_ratio_ = np.array([0.5, 0.3, 0.1])
        # Add metadata for testing dropdowns
        self.metadata_ = {"Group": ["A"] * 50 + ["B"] * 50, "Value": np.random.rand(100)}


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
    meta = {"Class": ["X"] * 25 + ["Y"] * 25, "Score": np.random.rand(50)}

    # 1. Basic call (backward compatibility)
    fig_basic = plot_embedding_interactive(emb, labels=labels, title="Basic")
    assert isinstance(fig_basic, go.Figure)
    assert fig_basic.layout.updatemenus == ()  # No dropdowns

    # 2. Advanced call (with metadata -> Dropdowns)
    fig_adv = plot_embedding_interactive(emb, labels=labels, meta=meta, title="Adv")
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

    rep.add_reduction(reducer, name="MockPCA")

    html = rep.render()

    assert "MockPCA" in html
    assert "📉" in html

    # With metadata now passed, we expect the embedding plot to work
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

