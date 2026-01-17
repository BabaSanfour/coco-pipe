"""
Tests for Phase 3: Dim-Red Components
"""
import pytest
import numpy as np
import plotly.graph_objects as go
from coco_pipe.report.core import Report, PlotlyElement
from coco_pipe.viz.plotly_utils import plot_embedding_interactive

class MockReducer:
    """Mock DimReduction object."""
    def __init__(self):
        self.embedding_ = np.random.randn(100, 2)
        self.loss_history_ = [10, 5, 2, 1]
        self.explained_variance_ratio_ = np.array([0.5, 0.3, 0.1])

def test_plotly_element_rendering():
    fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
    el = PlotlyElement(fig)
    html = el.render()
    
    # Check for lazy loading structure
    assert "lazy-plot" in html
    assert "data-figure" in html
    assert "&quot;data&quot;" in html # JSON encoded
    
def test_plot_embedding_interactive():
    emb = np.random.randn(50, 2)
    labels = np.random.randint(0, 2, 50)
    
    fig = plot_embedding_interactive(emb, labels=labels, title="Test Emb")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Emb"
    # Check trace type (Scattergl or Scatter)
    # default plot_embedding_interactive uses Scatter (webgl mode via express)
    # verify logic runs
    
def test_add_reduction_logic():
    rep = Report("DimRed Test")
    reducer = MockReducer()
    
    rep.add_reduction(reducer, name="MockPCA")
    
    html = rep.render()
    
    assert "MockPCA" in html
    assert "📉" in html
    
    # Needs to find multiple plots (Main + Loss + Scree)
    # We count occurrences of 'lazy-plot'
    assert html.count("lazy-plot") >= 3
