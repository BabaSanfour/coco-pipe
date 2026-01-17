"""
Tests for Phase 2: Data & Basic Viz
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from coco_pipe.report.core import Report, ImageElement, TableElement
from coco_pipe.io import DataContainer

@pytest.fixture
def sample_container():
    """Create a mock DataContainer for testing."""
    X = np.random.randn(10, 5)
    dims = ('obs', 'feature')
    coords = {'feature': ['A', 'B', 'C', 'D', 'E']}
    y = np.random.choice(['Class1', 'Class2'], 10)
    return DataContainer(X=X, dims=dims, coords=coords, y=y)

def test_image_element_matplotlib():
    """Test Base64 encoding of matplotlib figures."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    element = ImageElement(fig, caption="Test Plot")
    html = element.render()
    
    assert "data:image/png;base64" in html
    assert "<img" in html
    assert "Test Plot" in html
    
    plt.close(fig)

def test_table_element():
    """Test rendering of Pandas DataFrame."""
    df = pd.DataFrame({'Col1': [1, 2], 'Col2': ['A', 'B']})
    element = TableElement(df, title="My Table")
    html = element.render()
    
    assert "<table" in html
    assert "Col1" in html
    assert "A" in html
    assert "My Table" in html
    assert "border" in html # Tailwind class

def test_add_container(sample_container):
    """Test standard DataContainer summary generation."""
    rep = Report("Data Test")
    rep.add_container(sample_container)
    
    html = rep.render()
    
    # Check Section Title
    assert "Data Overview" in html
    assert "💾" in html
    
    # Check Dimension Table present
    assert "Dimensions" in html
    assert "feature" in html
    assert "10" in html # obs count
    
    # Check Plot generated (Class Distribution since y is present)
    assert "Target label distribution." in html
    assert "data:image/png;base64" in html

def test_add_figure_shortcut():
    """Test report.add_figure() shortcut."""
    rep = Report("Fig Test")
    fig, ax = plt.subplots()
    rep.add_figure(fig, caption="Shortcut")
    
    html = rep.render()
    assert "Shortcut" in html
    plt.close(fig)
