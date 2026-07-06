"""
High-level API for generating Reports from various sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from coco_pipe.io.quality import QCResult
    from coco_pipe.io.structures import DataContainer

    from .core import Report

SectionSelection = list[str] | Literal["compact", "default", "full"]


def from_container(
    container: DataContainer,
    title: str = "Analysis Report",
    config: dict[str, Any] | None = None,
    raw_preview: bool = True,
    theme: str = "paper",
    asset_urls: dict[str, str] | None = None,
    output_path: str | Path | None = None,
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
    theme : str
        Report theme preset.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    output_path : path-like, optional
        If given, save the rendered report to this path.

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

    report = Report(title=title, config=config, theme=theme, asset_urls=asset_urls)
    report.add_container(container)

    if raw_preview:
        report.add_raw_preview(container)
    if output_path is not None:
        report.save(output_path)

    return report


def from_bids(
    root: str | Path,
    task: str | None = None,
    theme: str = "paper",
    raw_preview: bool = True,
    asset_urls: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    **kwargs,
) -> Report:
    """
    Auto-generate a report from a BIDS dataset.

    Parameters
    ----------
    root : str or Path
        BIDS root directory.
    task : str, optional
        Task name.
    theme : str
        Report theme preset.
    raw_preview : bool
        If True, adds an interactive raw data scroller.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    output_path : path-like, optional
        If given, save the rendered report to this path.
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
    config = {
        "title": title,
        "run_params": {"source": "BIDS", "root": str(root), "task": task, **kwargs},
    }

    return from_container(
        container,
        title=title,
        config=config,
        raw_preview=raw_preview,
        theme=theme,
        asset_urls=asset_urls,
        output_path=output_path,
    )


def from_tabular(
    path: str | Path,
    theme: str = "paper",
    raw_preview: bool = True,
    asset_urls: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    **kwargs,
) -> Report:
    """
    Auto-generate a report from a tabular file (CSV/Excel).

    Parameters
    ----------
    path : str or Path
        Path to file.
    theme : str
        Report theme preset.
    raw_preview : bool
        If True, adds an interactive raw data scroller.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    output_path : path-like, optional
        If given, save the rendered report to this path.
    **kwargs
        Additional arguments passed to TabularDataset (e.g., target_col, clean).

    Returns
    -------
    Report
    """
    from coco_pipe.io.dataset import TabularDataset

    ds = TabularDataset(path=path, **kwargs)
    container = ds.load()
    title = f"Tabular Report: {Path(path).name}"
    config = {
        "title": title,
        "run_params": {"source": "Tabular", "path": str(path), **kwargs},
    }

    return from_container(
        container,
        title=title,
        config=config,
        raw_preview=raw_preview,
        theme=theme,
        asset_urls=asset_urls,
        output_path=output_path,
    )


def from_embeddings(
    path: str | Path,
    theme: str = "paper",
    raw_preview: bool = True,
    asset_urls: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    **kwargs,
) -> Report:
    """
    Auto-generate a report from a directory of embeddings.

    Parameters
    ----------
    path : str or Path
        Directory containing embedding files.
    theme : str
        Report theme preset.
    raw_preview : bool
        If True, adds an interactive raw data scroller.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    output_path : path-like, optional
        If given, save the rendered report to this path.
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
    title = f"Embedding Report: {Path(path).name}"
    config = {
        "title": title,
        "run_params": {"source": "Embeddings", "path": str(path), **kwargs},
    }

    return from_container(
        container,
        title=title,
        config=config,
        raw_preview=raw_preview,
        theme=theme,
        asset_urls=asset_urls,
        output_path=output_path,
    )


def from_reductions(
    reductions: list[Any],
    container: DataContainer | None = None,
    embeddings: list[np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    times: np.ndarray | None = None,
    title: str = "DimReduction Comparison",
    config: dict[str, Any] | None = None,
    sections: SectionSelection = "default",
    theme: str = "paper",
    raw_preview: bool = False,
    asset_urls: dict[str, str] | None = None,
    qc_result: QCResult | None = None,
    output_path: str | Path | None = None,
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
    config : dict, optional
        Extra configuration metadata stored in the report header.
    sections : list of str or ``"default"``
        Ordered list of reduction sections to include.
    theme : str
        Report theme preset.
    raw_preview : bool
        If True and *container* is provided, add an interactive raw data scroller.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    qc_result : QCResult, optional
        Structured QC drop log rendered as a standard report section.
    output_path : path-like, optional
        If given, save the rendered report to this path.

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
    from .dim_reduction import make_reduction_report

    factory_output_path = None if container is not None else output_path
    report = make_reduction_report(
        reductions,
        embeddings=embeddings,
        labels=labels,
        metadata=metadata,
        times=times,
        sections=sections,
        theme=theme,
        title=title,
        config=config,
        asset_urls=asset_urls,
        qc_result=qc_result,
        output_path=factory_output_path,
    )
    if container is not None:
        report.add_container(container)
        if raw_preview:
            report.add_raw_preview(container)
        if output_path is not None:
            report.save(output_path)
    return report


def from_experiment_result(
    result: Any,
    *,
    feature_metadata=None,
    info=None,
    coords=None,
    sections: SectionSelection = "default",
    interactive: bool = False,
    theme: str = "paper",
    title: str = "Decoding Report",
    config: dict | None = None,
    asset_urls: dict[str, str] | str | None = None,
    qc_result: QCResult | None = None,
    output_path: str | Path | None = None,
    verbose: bool | None = None,
    on_error: Literal["raise", "warn", "placeholder"] = "warn",
    section_options=None,
) -> Report:
    """Build a decoding report from an ``~coco_pipe.decoding.result.ExperimentResult``.

    Parameters
    ----------
    result : Any
        Decoding result object (e.g. ``~coco_pipe.decoding.result.ExperimentResult``).
    feature_metadata : pd.DataFrame, optional
        Feature-level metadata for sensor map sections.
    info : mne.Info, optional
        MNE Info for topomap rendering.
    coords : array-like or DataFrame, optional
        Sensor coordinates used when MNE Info is unavailable.
    sections : list of str or ``{"compact", "default", "full"}``
        Ordered section keys or a named report preset.
    interactive : bool
        If True, chart-like sections render interactive Plotly figures; topomap
        and sensor-map sections remain static Matplotlib images.
    theme : str
        Matplotlib theme preset (``"paper"`` | ``"notebook"`` | ``"poster"``).
    title : str
        Report title.
    config : dict, optional
        Extra configuration metadata stored in the report header.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    qc_result : QCResult, optional
        Structured QC drop log rendered as a standard report section.
    output_path : path-like, optional
        If given, save the rendered report to this path.
    verbose : bool, optional
        Override whether raw tables are rendered inline.
    on_error : {"raise", "warn", "placeholder"}
        Policy for unexpected section-builder failures.
    section_options : mapping, optional
        Per-section keyword overrides keyed by section name.

    Returns
    -------
    Report
        Fully populated decoding report.

    See Also
    --------
    coco_pipe.report.decoding.make_decoding_result_report : Lower-level factory.
    merge_reports : Combine multiple reports for cross-run comparison.

    Examples
    --------
    >>> report = from_experiment_result(result, title="EEG Decoding")
    >>> report.save("decoding.html")
    """
    from .decoding import make_decoding_result_report

    report = make_decoding_result_report(
        result,
        feature_metadata=feature_metadata,
        info=info,
        coords=coords,
        sections=sections,
        interactive=interactive,
        theme=theme,
        title=title,
        config=config,
        asset_urls=asset_urls,
        qc_result=qc_result,
        output_path=output_path,
        verbose=verbose,
        on_error=on_error,
        section_options=section_options,
    )
    return report


def from_experiment_results(
    items,
    *,
    by,
    comparisons="default",
    per_result="compact",
    nest=True,
    feature_metadata=None,
    info=None,
    coords=None,
    interactive=False,
    title="Decoding Comparison",
    config=None,
    asset_urls=None,
    qc_result=None,
    output_path=None,
    on_error="warn",
) -> Report:
    """Build one report from many labelled
    ``~coco_pipe.decoding.result.ExperimentResult`` objects or paths."""
    from .decoding_sweep import make_experiment_results_report

    return make_experiment_results_report(
        items,
        by=by,
        comparisons=comparisons,
        per_result=per_result,
        nest=nest,
        feature_metadata=feature_metadata,
        info=info,
        coords=coords,
        interactive=interactive,
        title=title,
        config=config,
        asset_urls=asset_urls,
        qc_result=qc_result,
        output_path=output_path,
        on_error=on_error,
    )


def from_decoding_sweep(records, **kwargs) -> Report:
    """Build the classical decoding sweep report from per-unit records.

    Thin façade over
    :func:`coco_pipe.report.decoding.make_decoding_report`; every keyword
    argument is forwarded unchanged (see that function for the full parameter
    set, e.g. ``title``, ``scope_order``, ``feature_metadata``, ``output_path``).

    Parameters
    ----------
    records
        Per-unit classical decoding rows (one mapping per analysis unit).

    Returns
    -------
    Report

    See Also
    --------
    from_foundation_sweep : The foundation-model counterpart.
    from_experiment_results : Compare a handful of loaded result objects.
    """
    from .decoding import make_decoding_report

    return make_decoding_report(records, **kwargs)


def from_foundation_sweep(records, **kwargs) -> Report:
    """Build the foundation-model decoding sweep report from per-model records.

    Thin façade over
    :func:`coco_pipe.report.foundation.make_foundation_decoding_report`; every
    keyword argument is forwarded unchanged (e.g. ``capability_records``,
    ``group_by``, ``per_result_sections``, ``output_path``).

    Parameters
    ----------
    records
        Per-model foundation decoding rows.

    Returns
    -------
    Report

    See Also
    --------
    from_decoding_sweep : The classical counterpart.
    """
    from .foundation import make_foundation_decoding_report

    return make_foundation_decoding_report(records, **kwargs)


def from_head_to_head(comparison_frame, **kwargs) -> Report:
    """Build the head-to-head comparison report from a tagged comparison frame.

    Thin façade over
    :func:`coco_pipe.report.decoding_sweep.make_head_to_head_report`; every
    keyword argument is forwarded unchanged. ``baseline_family`` is required by
    the underlying builder.

    Parameters
    ----------
    comparison_frame
        A frame carrying ``comparison_family`` plus per-run metrics (e.g. built
        with :func:`coco_pipe.report.collect_comparison_runs`).

    Returns
    -------
    Report
    """
    from .decoding_sweep import make_head_to_head_report

    return make_head_to_head_report(comparison_frame, **kwargs)


def merge_reports(*reports: Report, title: str = "Comparison Report") -> Report:
    """Merge multiple reports into a single comparison report.

    Each source report's sections are copied, prefixed with its own title, and
    appended to the merged output in the order supplied.

    Parameters
    ----------
    *reports : Report
        Two or more reports to merge.
    title : str
        Title for the merged report.

    Returns
    -------
    Report
        A new report whose sections interleave the source reports' content.

    Raises
    ------
    ValueError
        If fewer than two reports are supplied.

    See Also
    --------
    from_experiment_result : Build a single decoding report.
    from_reductions : Build a single reduction report.

    Examples
    --------
    >>> r1 = make_decoding_result_report(result_cohort_a)
    >>> r2 = make_decoding_result_report(result_cohort_b)
    >>> merged = merge_reports(r1, r2, title="Cross-Cohort Comparison")
    >>> merged.save("comparison.html")
    """
    from copy import deepcopy

    from .core import Report, _slugify

    if len(reports) < 2:
        raise ValueError("merge_reports() requires at least two Report objects.")

    merged = Report(title=title)
    for source in reports:
        prefix = getattr(source, "title", "Report") or "Report"
        for section in getattr(source, "children", []):
            copied = deepcopy(section)
            original_title = getattr(section, "title", "")
            copied.title = f"{prefix} — {original_title}" if original_title else prefix
            copied.id = _slugify(copied.title)
            merged.add_section(copied)
    return merged
