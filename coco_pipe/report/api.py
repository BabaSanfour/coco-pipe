"""
High-level API for generating Reports from various sources.
"""

from typing import List, Optional, Any, Union
from pathlib import Path

from coco_pipe.io.structures import DataContainer
from coco_pipe.io.dataset import BIDSDataset, TabularDataset, EmbeddingDataset
from coco_pipe.report.core import Report

def from_container(container: DataContainer, title: str = "Analysis Report") -> Report:
    """
    Create a standard report from a DataContainer.
    
    Parameters
    ----------
    container : DataContainer
        The data to summarize.
    title : str
        Report title.
        
    Returns
    -------
    Report
        A Report object with a "Data Overview" section added.
        
    Examples
    --------
    >>> ds = TabularDataset("data.csv")
    >>> container = ds.load()
    >>> report = from_container(container)
    >>> report.save("report.html")
    """
    report = Report(title=title)
    report.add_container(container)
    return report

def from_bids(root: Union[str, Path], task: Optional[str] = None, **kwargs) -> Report:
    """
    Auto-generate a report from a BIDS dataset.
    
    Parameters
    ----------
    root : str or Path
        BIDS root directory.
    task : str, optional
        Task name.
    **kwargs
        Additional arguments passed to BIDSDataset (e.g., session, subjects).
        
    Returns
    -------
    Report
    """
    ds = BIDSDataset(root=root, task=task, **kwargs)
    container = ds.load()
    title = f"BIDS Report: {task}" if task else "BIDS Dataset Report"
    return from_container(container, title=title)

def from_tabular(path: Union[str, Path], **kwargs) -> Report:
    """
    Auto-generate a report from a tabular file (CSV/Excel).
    
    Parameters
    ----------
    path : str or Path
        Path to file.
    **kwargs
        Additional arguments passed to TabularDataset (e.g., target_col, clean).
        
    Returns
    -------
    Report
    """
    ds = TabularDataset(path=path, **kwargs)
    container = ds.load()
    return from_container(container, title=f"Tabular Report: {Path(path).name}")

def from_embeddings(path: Union[str, Path], **kwargs) -> Report:
    """
    Auto-generate a report from a directory of embeddings.
    
    Parameters
    ----------
    path : str or Path
        Directory containing embedding files.
    **kwargs
        Additional arguments passed to EmbeddingDataset.
        
    Returns
    -------
    Report
    """
    ds = EmbeddingDataset(path=path, **kwargs)
    container = ds.load()
    return from_container(container, title=f"Embedding Report: {Path(path).name}")

def from_reductions(
    reductions: List[Any], 
    container: Optional[DataContainer] = None, 
    title: str = "DimReduction Comparison"
) -> Report:
    """
    Create a comparative report from multiple dimensionality reduction results.
    
    Parameters
    ----------
    reductions : List[Any]
        List of fitted reduction objects (must have `embedding_`).
    container : DataContainer, optional
        Original data container to include in "Data Overview".
    title : str
        Report title.
        
    Returns
    -------
    Report
        Report with Data Overview (if valid) and one section per reduction.
    """
    report = Report(title=title)
    
    if container:
        report.add_container(container)
        
    for i, red in enumerate(reductions):
        # Try to guess a name
        name = None
        if hasattr(red, 'name'): 
            name = red.name
        elif hasattr(red, '__class__'):
            name = red.__class__.__name__
        else:
            name = f"Method {i+1}"            
        report.add_reduction(red, name=name)
        
    return report
