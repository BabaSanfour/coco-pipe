"""
High-level API for generating Reports from various sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from coco_pipe.io.structures import DataContainer

    from .core import Report


def from_container(
    container: "DataContainer",
    title: str = "Analysis Report",
    config: Optional[Dict] = None,
    raw_preview: bool = True,
) -> Report:
    """
    Create a standard report from a DataContainer.

    Parameters
    ----------
    container : DataContainer
        The data to summarize.
    title : str
        Report title.
    config : Dict, optional
        Configuration/provenance info.
    raw_preview : bool
        If True, adds an interactive raw data scroller. Default True.

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
    from .core import Report

    report = Report(title=title, config=config)
    report.add_container(container)

    if raw_preview:
        report.add_raw_preview(container)

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
        A Report object with a "Data Overview" section added.

    Examples
    --------
    >>> report = from_bids("/path/to/bids")
    >>> report.save("report.html")
    """
    from coco_pipe.io.dataset import BIDSDataset

    ds = BIDSDataset(root=root, task=task, **kwargs)
    container = ds.load()
    title = f"BIDS Report: {task}" if task else "BIDS Dataset Report"

    # Auto-Provenance
    # We construct a dict that matches ReportConfig structure slightly better
    # or just pass as run_params
    config = {
        "title": title,
        "run_params": {"source": "BIDS", "root": str(root), "task": task, **kwargs},
    }

    return from_container(container, title=title, config=config)


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
    from coco_pipe.io.dataset import TabularDataset

    ds = TabularDataset(path=path, **kwargs)
    container = ds.load()

    # Auto-Provenance
    config = {
        "title": f"Tabular Report: {Path(path).name}",
        "run_params": {"source": "Tabular", "path": str(path), **kwargs},
    }

    return from_container(
        container, title=f"Tabular Report: {Path(path).name}", config=config
    )


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

    Examples
    --------
    >>> report = from_embeddings("/path/to/embeddings")
    >>> report.save("report.html")
    """
    from coco_pipe.io.dataset import EmbeddingDataset

    ds = EmbeddingDataset(path=path, **kwargs)
    container = ds.load()

    # Auto-Provenance
    config = {
        "title": f"Embedding Report: {Path(path).name}",
        "run_params": {"source": "Embeddings", "path": str(path), **kwargs},
    }

    return from_container(
        container, title=f"Embedding Report: {Path(path).name}", config=config
    )


def from_reductions(
    reductions: List[Any],
    container: Optional["DataContainer"] = None,
    embeddings: Optional[List[np.ndarray]] = None,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    times: Optional[np.ndarray] = None,
    title: str = "DimReduction Comparison",
    config: Optional[Dict] = None,
) -> Report:
    """
    Create a comparative report from multiple dimensionality reduction results.

    Parameters
    ----------
    reductions : List[Any]
        List of scored reduction objects implementing ``get_summary()``.
    container : DataContainer, optional
        Original data container to include in "Data Overview".
    embeddings : list of np.ndarray, optional
        Explicit embedding payloads aligned with ``reductions``.
    labels : np.ndarray, optional
        Optional labels aligned with each embedding.
    metadata : dict, optional
        Optional column-oriented metadata aligned with 2D embeddings.
    times : np.ndarray, optional
        Optional time axis aligned with 3D trajectory embeddings.
    title : str
        Report title.

    Returns
    -------
    Report
        Report with Data Overview (if valid) and one section per reduction.

    Notes
    -----
    Reduction summaries no longer carry cached embedding payloads. Pass
    ``embeddings`` explicitly when the report should render embedding or
    trajectory plots.

    Examples
    --------
    >>> report = from_reductions([pca, tsne], embeddings=[pca_emb, tsne_emb])
    >>> report.save("report.html")
    """
    from .core import Report

    report = Report(title=title, config=config)

    if container:
        report.add_container(container)

    if embeddings is not None and len(embeddings) != len(reductions):
        raise ValueError("`embeddings` must align with `reductions`.")

    for i, red in enumerate(reductions):
        # Try to guess a name
        name = None

        # Priority 1: method attribute
        if hasattr(red, "method"):
            candidate = getattr(red, "method")
            if isinstance(candidate, str):
                name = candidate

        # Priority 2: Class Name
        if name is None and hasattr(red, "__class__"):
            name = red.__class__.__name__

        # Fallback
        if name is None or not isinstance(name, str):
            name = f"Method {i + 1}"

        report.add_reduction(
            red,
            name=name,
            X_emb=None if embeddings is None else embeddings[i],
            labels=labels,
            metadata=metadata,
            times=times,
        )

    return report
