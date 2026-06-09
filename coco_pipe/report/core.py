"""
Core Reporting Classes
======================

Defines :class:`Section` and :class:`Report` — the high-level containers that
assemble :mod:`coco_pipe.report.elements` into a single-file HTML report.

The element primitives themselves live in :mod:`coco_pipe.report.elements`
and are re-exported from this module for backward compatibility.
"""

import base64
import gzip
import html
import json
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from coco_pipe.io.quality import (
    CheckResult,
    check_constant_columns,
    check_flatline,
    check_missingness,
    check_outliers_zscore,
)
from coco_pipe.utils import get_environment_info

from ._engine import render_template
from .config import ProvenanceConfig, ReportConfig
from .elements import (
    ColumnsElement,
    ContainerElement,
    Element,
    HtmlElement,
    ImageElement,
    PlotlyElement,
    StatCardElement,
    TableElement,
)

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """Return a stable HTML id slug for a human-readable label."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "section"


def _replace_non_finite(value: Any) -> Any:
    """Recursively replace ``NaN``/``±Inf`` floats with ``None``.

    The browser uses ``JSON.parse`` to read the gzip-compressed report
    payload. Python's ``json.dumps(allow_nan=True)`` default emits
    bare ``NaN`` and ``Infinity`` tokens, which JSON.parse rejects with
    ``SyntaxError: Unexpected token 'N'``. The error breaks the whole
    payload, so *every* figure / table in the report fails to render
    — not just the one carrying the NaN.

    We walk dicts, lists, tuples, and numpy/pandas containers, swap each
    non-finite scalar for ``None`` (which serialises to ``null``), and
    leave everything else untouched.
    """
    if isinstance(value, dict):
        return {k: _replace_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cleaned = [_replace_non_finite(v) for v in value]
        return cleaned if isinstance(value, list) else tuple(cleaned)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            return np.where(np.isfinite(value), value, None).tolist()
        return value.tolist()
    if isinstance(value, (float, np.floating)):
        return value if np.isfinite(value) else None
    return value


class Section(ContainerElement):
    """
    A logical section of the report.

    Parameters
    ----------
    title : str
        The section title.
    icon : str, optional
        Short label or SVG markup displayed next to the title.
    tags : List[str], optional
        Tags for filtering.
    status : str, optional
        Status string ("OK", "WARN", "FAIL"). Default "OK".
    code : str, optional
        Source code snippet to reproduce this section.
    description : str, optional
        Short secondary text shown under the section title.
    metadata : dict, optional
        Compact key/value facts rendered above the section body, such as
        method, inputs, or outputs.

    Examples
    --------
    >>> sec = Section("Results", status="OK")
    >>> sec.add_element(plotly_element)
    >>> rep.add_section(sec)
    """

    def __init__(
        self,
        title: str,
        icon: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: str = "OK",
        code: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.title = title
        self.icon = icon
        self.tags = tags if tags else []
        self.status = status
        self.code = code
        self.description = description
        self.metadata = metadata or {}
        self.findings: List[Dict] = []  # List of serialized CheckResults

        # Generated ID (slugify)
        self.id = _slugify(self.title)

    def add_finding(self, result: CheckResult) -> None:
        """Add a quality finding and automatically update status."""
        self.findings.append(result.__dict__)  # Store as dict for JSON serialization

        # Upgrade status logic
        if result.status == "FAIL":
            self.status = "FAIL"
        elif result.status == "WARN" and self.status != "FAIL":
            self.status = "WARN"

    def add_columns(
        self,
        elements: List[Element],
        cols: Optional[int] = None,
        gap: str = "gap-4",
    ) -> "Section":
        """
        Add multiple elements side by side in a CSS grid row.

        Parameters
        ----------
        elements : list of Element
            Child elements to place in columns.
        cols : int, optional
            Number of columns. Inferred from ``len(elements)`` when omitted.
        gap : str, optional
            Tailwind gap class. Default ``"gap-4"``.

        Returns
        -------
        Section
            The section instance for fluent chaining.
        """
        self.add_element(ColumnsElement(elements, cols=cols, gap=gap))
        return self

    def render(self) -> str:
        content = self.render_children()
        return render_template(
            "section.html",
            title=self.title,
            icon=self.icon,
            content=content,
            id=self.id,
            tags=json.dumps(self.tags),
            status=self.status,
            code=self.code,
            description=self.description,
            metadata=self.metadata,
            findings=self.findings,  # Pass list of dicts for Jinja iteration
        )


class Report(ContainerElement):
    """
    The main report container.

    Parameters
    ----------
    title : str
        The report title.
    config : Union[Dict, ReportConfig], optional
        Configuration dictionary or ReportConfig object used for the run.
    asset_urls : dict or ``"inline"``, optional
        Controls how the Plotly / Tailwind / pako JS bundles are loaded:

        - ``None`` (default): load from CDN.
        - ``dict``: override one or more URLs (e.g. point at self-hosted copies).
        - ``"inline"``: download (and cache on disk) the bundles, then inline
          them in ``<script>`` tags so the rendered HTML is fully self-contained
          and openable offline. See :mod:`coco_pipe.report._assets`.
    """

    _DEFAULT_ASSET_URLS = {
        "plotly": "https://cdn.plot.ly/plotly-2.27.0.min.js",
        "tailwind": "https://cdn.tailwindcss.com",
        "pako": "https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js",
    }

    def __init__(
        self,
        title: str = "CoCo Analysis Report",
        config: Optional[Union[Dict, ReportConfig]] = None,
        theme: str = "paper",
        asset_urls: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.theme = theme
        self._apply_theme(theme)
        self.asset_urls, self.asset_mode = self._resolve_assets(asset_urls)
        self.config = self._resolve_config(config, title)
        self.title = self.config.title
        if self.config.provenance is None:
            self.config.provenance = ProvenanceConfig(**get_environment_info())
        self.metadata = self.config.provenance.model_dump()

    @staticmethod
    def _apply_theme(theme: str) -> None:
        """Apply the matplotlib + Plotly theme used by report figures."""
        try:
            from coco_pipe.viz.theme import set_coco_theme

            set_coco_theme(theme)
        except Exception as exc:
            logger.debug("Could not apply report theme %s: %s", theme, exc)

        try:
            from coco_pipe.viz.interactive._utils import _register_coco_template

            _register_coco_template()
        except Exception as exc:
            logger.debug("Could not register coco Plotly template: %s", exc)

    @classmethod
    def _resolve_assets(
        cls, asset_urls: Optional[Union[Dict[str, str], str]]
    ) -> tuple[Dict[str, str], str]:
        """
        Resolve the JS-asset slots into ``(urls_or_contents, mode)``.

        - ``None``       → CDN defaults (``mode="cdn"``).
        - ``dict``       → CDN defaults merged with user overrides (``mode="custom"``).
        - ``"inline"``   → download (with caching) the bundles and inline them
                           in ``<script>`` tags (``mode="inline"``).
        """
        from ._assets import INLINE_SENTINEL, get_vendored_contents

        if asset_urls == INLINE_SENTINEL:
            return get_vendored_contents(), "inline"
        if asset_urls is None:
            return dict(cls._DEFAULT_ASSET_URLS), "cdn"
        if isinstance(asset_urls, dict):
            return {**cls._DEFAULT_ASSET_URLS, **asset_urls}, "custom"
        raise TypeError(
            f"asset_urls must be None, a dict, or {INLINE_SENTINEL!r}; "
            f"got {type(asset_urls).__name__}."
        )

    @staticmethod
    def _resolve_config(
        config: Optional[Union[Dict, ReportConfig]],
        title: str,
    ) -> ReportConfig:
        """Coerce ``config`` to a :class:`ReportConfig`, honoring ``title``."""
        if config is None:
            return ReportConfig(title=title)
        if isinstance(config, ReportConfig):
            return config
        # config is a dict at this point
        # Title in the dict wins over the argument; otherwise inject the arg
        if "title" not in config:
            config = {**config, "title": title}
        try:
            return ReportConfig(**config)
        except Exception:
            # Fallback: treat the dict as a free-form run_params bag
            return ReportConfig(title=config.get("title", title), run_params=config)

    def add_section(self, section: Section) -> "Report":
        """Syntactic sugar for adding a Section."""
        existing_ids = {
            child.id for child in self.children if isinstance(child, Section)
        }
        base_id = section.id or _slugify(section.title)
        candidate = base_id
        suffix = 2
        while candidate in existing_ids:
            candidate = f"{base_id}-{suffix}"
            suffix += 1
        section.id = candidate
        return self.add_element(section)

    def add_figure(self, fig: Any, caption: Optional[str] = None) -> "Report":
        """
        Add a figure (Matplotlib) or Image.
        """
        self.add_element(ImageElement(fig, caption=caption))
        return self

    def add_container(
        self,
        container: Any,
        name: str = "Data Overview",
        show_coords: bool = True,
        show_dist: bool = True,
    ) -> "Report":
        """
        Add a summary section for a DataContainer.
        Automatically runs quality checks (Missingness, Constants).

        Parameters
        ----------
        container : DataContainer
            The data container to summarize.
        name : str
            Title for the section.
        show_coords : bool
            If True, shows the table of coordinates.
        show_dist : bool
            If True, shows the data/class distribution plot.
        """
        try:
            # Create Section
            sec = Section(title=name)

            # Dimensions
            dims_data = [
                {"Dimension": d, "Size": s}
                for d, s in zip(container.dims, container.shape)
            ]
            sec.add_element(TableElement(dims_data, title="Dimensions"))

            # Coordinates Info
            if show_coords and container.coords:
                coords_data = [
                    {"Name": k, "Type": str(np.array(v).dtype), "Count": len(v)}
                    for k, v in container.coords.items()
                ]
                sec.add_element(TableElement(coords_data, title="Coordinates"))

            # 2. Distribution Plot
            if show_dist:
                try:
                    # Quality Checks
                    if container.X is not None:
                        res_missing = check_missingness(container.X)
                        if res_missing.is_issue:
                            sec.add_finding(res_missing)

                        for res in check_constant_columns(container.X):
                            sec.add_finding(res)

                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 3))

                    if container.y is not None:
                        y_series = pd.Series(container.y)
                        y_series.value_counts().plot(kind="bar", ax=ax, color="skyblue")
                        ax.set_title("Class Distribution")
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Count")
                        caption = "Target label distribution."
                    else:
                        data_flat = container.X.flatten()
                        if len(data_flat) > 5000:
                            data_flat = np.random.choice(data_flat, 5000, replace=False)
                        ax.hist(data_flat, bins=30, color="gray", alpha=0.7)
                        ax.set_title("Data Value Distribution (Sampled)")
                        caption = "Histogram of data values."

                    plt.tight_layout()
                    sec.add_element(ImageElement(fig, caption=caption, width="80%"))
                    plt.close(fig)

                except Exception as e:
                    msg = f"Could not generate plot: {e}"
                    html = f"<div class='text-red-500 text-xs'>{msg}</div>"
                    sec.add_element(HtmlElement(html))

            self.add_section(sec)
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to add container info to report: {e}", UserWarning)

        return self

    def add_raw_preview(self, data: Any, name: str = "Raw Data Inspector") -> "Report":
        """
        Add an interactive scroller for raw data.
        Automatically checks for flatlines and outliers.

        Parameters
        ----------
        data : DataContainer or np.ndarray
            The data to visualize.
        name : str
            Section title.
        """
        sec = Section(title=name)

        # Extract array
        X = data
        names = None
        if hasattr(data, "X"):  # DataContainer
            X = data.X

        X_array = np.asarray(X)
        try:
            sample_X = X_array if X_array.size < 10000 else X_array.reshape(-1)[:10000]
            res_flat = check_flatline(sample_X)
            if res_flat.is_issue:
                sec.add_finding(res_flat)

            res_outlier = check_outliers_zscore(sample_X)
            if res_outlier:
                sec.add_finding(res_outlier)
        except Exception as e:
            logger.debug(f"Data quality checks failed: {e}")

        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if X_array.ndim > 2:
            X_array = X_array.reshape(X_array.shape[0] * X_array.shape[1], -1)

        from coco_pipe.viz.interactive.dim_reduction import plot_raw_preview

        fig = plot_raw_preview(X_array, names=names, title=name)
        sec.add_element(PlotlyElement(fig, height="450px"))

        self.add_section(sec)
        return self

    def render(self) -> str:
        """
        Render the full HTML report.

        Walks every element to collect heavy payload data (Plotly figures,
        interactive tables) into a single registry keyed by UUID, then
        compresses and base64-encodes the registry so it can be embedded
        inline in the page.

        Payload format (the contract with ``base.html``)
        -----------------------------------------------
        Each element that wants to store payload assigns itself a UUID
        ``data-id`` and pushes its data into the registry:

            registry[uuid] = {<element-specific JSON>}

        The registry is then serialized as::

            JSON.dumps(registry)
            -> utf-8 bytes
            -> gzip.compress()
            -> base64.b64encode().decode("utf-8")

        and emitted inside ``<script type="application/json"
        id="report-payload">...</script>``. The browser script in
        ``base.html`` reverses this (``atob -> pako.inflate -> JSON.parse``)
        and exposes the result as ``REPORT_DATA``. Each
        ``<div class="lazy-plot" data-id="..."></div>`` then looks up its
        payload in ``REPORT_DATA[dataId]`` on intersection.
        """
        # 1. Collect Payload (Global Data Store)
        data_registry = {}
        self.collect_payload(data_registry)

        # 2. Compress Payload (JSON -> Gzip -> Base64)
        payload_json = json.dumps(_replace_non_finite(data_registry)).encode("utf-8")
        compressed = gzip.compress(payload_json)
        payload_b64 = base64.b64encode(compressed).decode("utf-8")
        plot_count = sum(
            1
            for item in data_registry.values()
            if isinstance(item, dict) and "data" in item and "layout" in item
        )
        table_count = sum(
            1
            for item in data_registry.values()
            if isinstance(item, dict) and "columns" in item and "rows" in item
        )

        # 3. Get content from children (Sections)
        # Note: Children now render with data-id references since collect_payload
        # was called.
        content_html = super().render()

        # Build TOC Structure from Sections
        toc = []
        for child in self.children:
            if isinstance(child, Section):
                toc.append(
                    {
                        "id": child.id,
                        "title": child.title,
                        "icon": child.icon,
                        "status": child.status,
                    }
                )
        warn_count = sum(1 for item in toc if item["status"] == "WARN")
        fail_count = sum(1 for item in toc if item["status"] == "FAIL")
        report_summary = {
            "sections": len(toc),
            "warnings": warn_count,
            "failures": fail_count,
            "generated": self.timestamp,
            "git_hash": self.metadata.get("git_hash", "Unknown"),
            "python_version": self.metadata.get("python_version", "Unknown"),
            "package_version": self.metadata.get(
                "coco_pipe_version",
                self.metadata.get("versions", {}).get("coco-pipe", "Unknown"),
            ),
            "payload_bytes": len(compressed),
            "payload_items": len(data_registry),
            "plot_count": plot_count,
            "table_count": table_count,
            "asset_mode": self.asset_mode,
        }

        # Wrap in base template
        return render_template(
            "base.html",
            title=self.title,
            content=content_html,
            timestamp=self.timestamp,
            toc=toc,
            metadata=self.metadata,
            config=self.config.model_dump_json(indent=2),
            payload=payload_b64,
            report_summary=report_summary,
            asset_urls=self.asset_urls,
            asset_mode=self.asset_mode,
        )

    def add_summary_card(
        self,
        metrics: Dict[str, Any],
        colors: Optional[List[str]] = None,
    ) -> "Report":
        """
        Add a row of KPI stat-cards above the first section.

        Each key-value pair in *metrics* becomes one :class:`StatCardElement`.
        The card row is inserted at the top of the report children list so it
        appears before any sections regardless of when this method is called.

        Parameters
        ----------
        metrics : dict
            Mapping of label → value to display. Float values are formatted
            with ``:.4g``; all other types are converted to ``str``.
        colors : list of str, optional
            Tailwind color keys for each card in order. Cycles through
            ``["blue", "green", "purple", "yellow", "red"]`` by default.

        Returns
        -------
        Report
            The report instance for fluent chaining.

        Examples
        --------
        >>> report.add_summary_card({"Best Accuracy": 0.842, "Models": 3, "Folds": 5})
        """
        palette = colors or ["blue", "green", "purple", "yellow", "red"]
        cards = [
            StatCardElement(label, value, color=palette[i % len(palette)])
            for i, (label, value) in enumerate(metrics.items())
        ]
        row = ColumnsElement(cards, cols=min(len(cards), 5))
        # Insert at position 0 so it appears above all sections.
        self.children.insert(0, row)
        return self

    def show(self, port: Optional[int] = None) -> None:
        """
        Render the report and open it in the default web browser.

        A temporary HTML file is written to the system temp directory, then
        ``webbrowser.open`` is called. No server is started; the file is
        self-contained and can be opened directly.

        Parameters
        ----------
        port : int, optional
            Ignored. Kept for API compatibility with future server-based
            ``show()`` variants.

        Examples
        --------
        >>> report.show()
        """
        import tempfile
        import webbrowser

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(self.render())
            path = f.name
        webbrowser.open(f"file://{path}")

    def _repr_html_(self) -> str:
        """
        Return an iframe embedding of the rendered report for Jupyter display.

        Jupyter automatically calls this method when a :class:`Report` instance
        is the last expression in a notebook cell, rendering an inline preview.

        Returns
        -------
        str
            HTML ``<iframe>`` wrapping the full report via a ``srcdoc`` attribute.
        """
        rendered = self.render()
        escaped = html.escape(rendered, quote=True)
        return (
            f'<iframe srcdoc="{escaped}" width="100%" height="800px" '
            f'style="border:1px solid #e5e7eb; border-radius:8px;" '
            f"allowfullscreen></iframe>"
        )

    def save(self, filename: str | Path) -> None:
        """
        Render and save the report to a file.

        Parameters
        ----------
        filename : str or Path
            Path to save the HTML file.

        Notes
        -----
        Emits a ``UserWarning`` when the report still depends on three
        external CDN scripts (Plotly, Tailwind, pako). A CDN-backed
        report renders as grey placeholders if opened without a network
        connection (e.g. ``file://`` viewing, offline archives, restrictive
        corporate firewalls, air-gapped machines). Pass
        ``asset_urls="inline"`` to :class:`Report` to bundle the
        JavaScript directly into the HTML.
        """
        if self.asset_mode == "cdn":
            warnings.warn(
                f"Report saved to {filename} references three external "
                "CDN scripts (Plotly, Tailwind, pako). It will appear as "
                "grey placeholder boxes if opened without a network "
                "connection (file:// viewing, offline machines, "
                "restrictive firewalls). Pass `asset_urls='inline'` to "
                "Report(...) for a fully self-contained HTML.",
                UserWarning,
                stacklevel=2,
            )
        full_html = self.render()
        Path(filename).write_text(full_html, encoding="utf-8")


__all__ = [
    "Section",
    "Report",
]
