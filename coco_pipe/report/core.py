"""
Core Reporting Classes
======================

Defines the object hierarchy for the reporting system.
"""

import base64
import gzip
import io
import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from coco_pipe.viz.plotly_utils import (
    plot_embedding_interactive,
    plot_loss_history_interactive,
    plot_raw_preview,
    plot_scree_interactive,
)

from .config import ReportConfig
from .engine import render_template
from .provenance import get_environment_info
from .quality import (
    CheckResult,
    check_constant_columns,
    check_flatline,
    check_missingness,
    check_outliers_zscore,
)


class Element(ABC):
    """
    Abstract base class for all report elements.
    """

    @abstractmethod
    def render(self) -> str:
        """Render the element to HTML."""
        pass

    def collect_payload(self, registry: Dict[str, Any]) -> None:
        """
        Collect data to be stored in the global payload.
        Default implementation does nothing.

        Parameters
        ----------
        registry : Dict[str, Any]
            Global dictionary accumulating data. Keyed by UUID.
        """
        pass


class HtmlElement(Element):
    """
    Wrapper for raw HTML content.

    Parameters
    ----------
    html : str
        The raw HTML string to include.

    Examples
    --------
    >>> elem = HtmlElement("<div>My Custom HTML</div>")
    >>> rep.add_element(elem)
    """

    def __init__(self, html: str):
        self.html = html

    def render(self) -> str:
        return self.html


class ImageElement(Element):
    """
    Embeds an image or matplotlib figure as Base64.

    Parameters
    ----------
    src : str, bytes, Path, or matplotlib.figure.Figure
        The image source.
    caption : str, optional
        Caption text for the figure.
    width : str, optional
        CSS width (e.g., '100%', '600px'). Default '100%'.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> elem = ImageElement(fig, caption="My Plot")
    """

    def __init__(self, src: Any, caption: Optional[str] = None, width: str = "100%"):
        self.src = src
        self.caption = caption
        self.width = width

    def _encode_image(self) -> str:
        """Convert input to base64 string."""
        # Check for Matplotlib Figure
        if hasattr(self.src, "savefig"):
            buf = io.BytesIO()
            self.src.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            data = buf.read()
            return base64.b64encode(data).decode("utf-8")

        # Check for bytes
        if isinstance(self.src, bytes):
            return base64.b64encode(self.src).decode("utf-8")

        # Check for path (str or Path)
        if isinstance(self.src, (str, type(None))):  # type check loose for Path
            pass  # import pathlib below

        import pathlib

        if isinstance(self.src, (str, pathlib.Path)):
            p = pathlib.Path(self.src)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode("utf-8")

        raise ValueError(f"Unsupported image source type: {type(self.src)}")

    def render(self) -> str:
        b64_str = self._encode_image()
        html = f"""
        <figure class="my-6">
            <img src="data:image/png;base64,{b64_str}" style="width: {self.width};" class="rounded shadow-sm mx-auto border border-gray-100">
            {f'<figcaption class="text-center text-sm text-gray-500 mt-2">{self.caption}</figcaption>' if self.caption else ''}
        </figure>
        """
        return html


class PlotlyElement(Element):
    """
    Embeds a Plotly figure using lazy loading and global data usage.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to render.
    height : str, optional
        Height of the plot plot container. Default "500px".

    Examples
    --------
    >>> fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    >>> elem = PlotlyElement(fig)
    """

    def __init__(self, figure: Any, height: str = "500px"):
        self.figure = figure
        self.height = height
        self.registry_id = None

    def collect_payload(self, registry: Dict[str, Any]) -> None:
        """Extract figure data and store in registry."""
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        fig_dict = self.figure.to_dict()
        registry[self.registry_id] = fig_dict

    def render(self) -> str:
        # Instead of dumping JSON, we reference the ID
        if self.registry_id is None:
            return self._render_inline()

        html = f"""
        <div class="my-6">
            <div class="lazy-plot w-full rounded shadow-sm border border-gray-100 bg-gray-50 flex items-center justify-center text-gray-400 animate-pulse"
                 style="height: {self.height};"
                 data-id="{self.registry_id}">
                 <span class="sr-only">Loading Plot...</span>
            </div>
        </div>
        """
        return html

    def _render_inline(self) -> str:
        fig_dict = self.figure.to_dict()
        json_str = json.dumps(fig_dict)
        safe_json = json_str.replace('"', "&quot;")

        return f"""
        <div class="my-6">
            <div class="lazy-plot w-full rounded shadow-sm border border-gray-100 bg-gray-50 flex items-center justify-center text-gray-400 animate-pulse"
                 style="height: {self.height};"
                 data-figure="{safe_json}">
                 <span class="sr-only">Loading Plot...</span>
            </div>
        </div>
        """


class TableElement(Element):
    """
    Renders a Pandas DataFrame or Dict as a styled HTML table.

    Parameters
    ----------
    data : DataFrame, Dict, or List[Dict]
        Data to display.
    title : str, optional
        Title describing the table.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> elem = TableElement(df, title="Metrics")
    """

    def __init__(self, data: Any, title: Optional[str] = None):
        self.data = data
        self.title = title
        self.table_id = f"table-{uuid.uuid4().hex[:8]}"

    def render(self) -> str:
        # Convert to DataFrame
        if isinstance(self.data, pd.DataFrame):
            df = self.data
        else:
            df = pd.DataFrame(self.data)

        # Basic Tailwind Styling
        html = '<div class="overflow-x-auto my-4 group relative">'
        if self.title:
            html += f"""
            <div class="flex justify-between items-center mb-2">
                <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">{self.title}</h4>
                <button onclick="exportTableToCSV('{self.table_id}', '{self.title or "data"}')" class="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded text-gray-500 transition opacity-0 group-hover:opacity-100">
                    ⬇ CSV
                </button>
            </div>
            """

        # Render Table
        html += f'<table id="{self.table_id}" class="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border dark:border-gray-700 text-sm">'

        # Header
        html += '<thead class="bg-gray-50 dark:bg-gray-800"><tr>'
        for col in df.columns:
            html += f'<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">{col}</th>'
        html += "</tr></thead>"

        # Body
        html += '<tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">'
        for idx, row in df.iterrows():
            html += self._render_row(row, idx)
        html += "</tbody></table></div>"

        return html

    def _render_row(self, row, idx) -> str:
        """Render a single row. Can be overridden."""
        html = "<tr>"
        for val in row:
            html += f'<td class="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">{val}</td>'
        html += "</tr>"
        return html


class MetricsTableElement(TableElement):
    """
    Comparison table that highlights best values.

    Parameters
    ----------
    data : DataFrame
        Comparison data (rows=methods, cols=metrics).
    highlight_cols : List[str], optional
        Columns to highlight best values in.
    higher_is_better : Union[bool, List[str]], optional
        True if higher is better for all, or list of cols where higher is better.
        Default True.
    """

    def __init__(
        self,
        data: Any,
        title: str = "Comparison Metrics",
        highlight_cols: Optional[List[str]] = None,
        higher_is_better: Union[bool, List[str]] = True,
    ):
        super().__init__(data, title)
        self.highlight_cols = highlight_cols
        self.higher_is_better = higher_is_better

        # Pre-compute best values
        self.best_vals = {}
        if isinstance(self.data, pd.DataFrame):
            cols = (
                self.highlight_cols
                if self.highlight_cols
                else self.data.select_dtypes(include=[np.number]).columns
            )
            for col in cols:
                if col not in self.data.columns:
                    continue

                # Determine direction
                is_higher = True
                if isinstance(self.higher_is_better, list):
                    is_higher = col in self.higher_is_better
                else:
                    is_higher = self.higher_is_better

                if is_higher:
                    self.best_vals[col] = self.data[col].max()
                else:
                    self.best_vals[col] = self.data[col].min()

    def _render_row(self, row, idx) -> str:
        html = "<tr>"
        for col, val in row.items():
            # Check if best
            is_best = False
            if col in self.best_vals and np.isclose(val, self.best_vals[col]):
                is_best = True

            style = "text-gray-700 dark:text-gray-300"
            if is_best:
                style = "font-bold text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20"

            # Format numbers
            display_val = val
            if isinstance(val, float):
                display_val = f"{val:.4f}"

            html += (
                f'<td class="px-4 py-3 whitespace-nowrap {style}">{display_val}</td>'
            )
        html += "</tr>"
        return html


class ContainerElement(Element):
    """
    Base class for elements that contain other elements.
    """

    def __init__(self):
        self.children: List[Element] = []

    def add_element(self, element: Union[Element, str]):
        """
        Add a child element.

        Parameters
        ----------
        element : Element or str
            The element to add. specific strings are converted to HtmlElement.

        Returns
        -------
        self
            Fluent interface.
        """
        if isinstance(element, str):
            element = HtmlElement(element)
        self.children.append(element)
        return self  # Fluent interface

    def render_children(self) -> str:
        """Render all child elements concatenated."""
        return "\n".join([c.render() for c in self.children])

    def collect_payload(self, registry: Dict[str, Any]) -> None:
        """Recursively collect payload from children."""
        for child in self.children:
            child.collect_payload(registry)

    def render(self) -> str:
        return self.render_children()


class Section(ContainerElement):
    """
    A logical section of the report.

    Parameters
    ----------
    title : str
        The section title.
    icon : str, optional
        SVG icon or emoji to display next to the title.
    tags : List[str], optional
        Tags for filtering.
    status : str, optional
        Status string ("OK", "WARN", "FAIL"). Default "OK".
    code : str, optional
        Source code snippet to reproduce this section.

    Examples
    --------
    >>> sec = Section("Results", icon="📈", status="OK")
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
    ):
        super().__init__()
        self.title = title
        self.icon = icon
        self.tags = tags if tags else []
        self.status = status
        self.code = code
        self.findings: List[Dict] = []  # List of serialized CheckResults

        # Generated ID (slugify)
        self.id = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")

    def add_finding(self, result: CheckResult) -> None:
        """Add a quality finding and automatically update status."""
        self.findings.append(result.__dict__)  # Store as dict for JSON serialization

        # Upgrade status logic
        if result.status == "FAIL":
            self.status = "FAIL"
        elif result.status == "WARN" and self.status != "FAIL":
            self.status = "WARN"

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
    """

    def __init__(
        self,
        title: str = "CoCo Analysis Report",
        config: Optional[Union[Dict, ReportConfig]] = None,
    ):
        super().__init__()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Validate/Coerce Config
        if config is None:
            config = {}

        if isinstance(config, dict):
            # If title is in config, it overrides arg
            if "title" in config:
                title = config["title"]
            else:
                # Ensure the argument title takes precedence over ReportConfig default
                config["title"] = title

            try:
                self.config = ReportConfig(**config)
            except Exception:
                # If direct validation fails, assume it's a bag of parameters
                self.config = ReportConfig(title=title, run_params=config)
        else:
            self.config = config

        # Ensure title sync
        self.title = self.config.title

        # Auto-capture environment provenance if not provided
        if self.config.provenance is None:
            # metadata from existing functionality
            raw_meta = get_environment_info()
            # raw_meta keys match ProvenanceConfig closely?
            # get_environment_info returns: timestamp_utc, os_platform, python_version, command, git_hash, versions...
            # This matches ProvenanceConfig fields.
            from .config import ProvenanceConfig

            self.config.provenance = ProvenanceConfig(**raw_meta)

        self.metadata = self.config.provenance.model_dump()

    def add_markdown(self, text: str) -> "Report":
        """
        Add a markdown block to the report.

        Note: Requires 'markdown' package. If not present, falls back to raw text in <pre>.
        """
        try:
            import markdown

            html = markdown.markdown(text, extensions=["extra"])
            # Wrap in prose class for consistent styling
            wrapper = (
                f'<div class="prose prose-sm max-w-none text-gray-700">{html}</div>'
            )
            self.add_element(HtmlElement(wrapper))
        except ImportError:
            # Fallback
            safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
            html = f'<div class="whitespace-pre-wrap font-mono text-sm bg-gray-50 p-4 rounded">{safe_text}</div>'
            self.add_element(HtmlElement(html))
        return self

    def add_section(self, section: Section) -> "Report":
        """Syntactic sugar for adding a Section."""
        return self.add_element(section)

    def add_figure(self, fig: Any, caption: Optional[str] = None) -> "Report":
        """
        Add a figure (Matplotlib) or Image.
        """
        self.add_element(ImageElement(fig, caption=caption))
        return self

    def add_container(self, container: Any, name: str = "Data Overview") -> "Report":
        """
        Add a summary section for a DataContainer.
        Automatically runs quality checks (Missingness, Constants).

        Parameters
        ----------
        container : DataContainer
            The data container to summarize.
        name : str
            Title for the section.
        """
        # Create Section
        sec = Section(title=name, icon="💾")

        # 1. Metadata Table
        # Extract general info
        {
            "Dimension": list(container.dims),
            "Size": list(container.shape),
        }

        # Add shape info to meta_info for table
        # We'll pivot this for cleaner display: Dim Name -> Size
        dims_data = [
            {"Dimension": d, "Size": s} for d, s in zip(container.dims, container.shape)
        ]

        sec.add_element(TableElement(dims_data, title="Dimensions"))

        # Coordinates Info
        if container.coords:
            coords_data = [
                {"Name": k, "Type": str(np.array(v).dtype), "Count": len(v)}
                for k, v in container.coords.items()
            ]
            sec.add_element(TableElement(coords_data, title="Coordinates"))

        # 2. Simple Distribution Plot (if applicable)
        try:
            # Quality Checks
            if container.X is not None:
                # Missingness
                res_missing = check_missingness(container.X)
                if res_missing.is_issue:
                    sec.add_finding(res_missing)

                # Constant Columns
                for res in check_constant_columns(container.X):
                    sec.add_finding(res)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))

            # Check for 'y' (Class Distribution)
            if container.y is not None:
                # Plot Class counts
                y_series = pd.Series(container.y)
                y_series.value_counts().plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title("Class Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                caption = "Target label distribution."

            else:
                # Plot simple data histogram (sampled if large)
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
            sec.add_element(
                HtmlElement(
                    f"<div class='text-red-500 text-xs'>Could not generate plot: {e}</div>"
                )
            )

        self.add_section(sec)
        return self

    def add_reduction(self, reducer: Any, name: str = None) -> "Report":
        """
        Add a dimensionality reduction result to the report.

        Parameters
        ----------
        reducer : DimReduction or similar
            Fitted dimensionality reduction object. Must have 'embedding_' attribute.
        name : str, optional
            Name of the reduction (e.g., "UMAP"). If None, inferred from class name.
        """
        if name is None:
            name = type(reducer).__name__

        sec = Section(title=name, icon="📉")

        # 1. Main Interactive Embedding Plot
        if hasattr(reducer, "embedding_"):
            emb = reducer.embedding_
            dims = emb.shape[1]
            if dims > 3:
                dimensions = 3
            else:
                dimensions = dims

            fig = plot_embedding_interactive(
                embedding=emb, title=f"{name} Embedding", dimensions=dimensions
            )
            sec.add_element(PlotlyElement(fig))

        # 2. Metrics Table (TODO: Add if metrics attribute exists)

        # 3. Diagnostics
        # Loss Curve
        if hasattr(reducer, "loss_history_") and reducer.loss_history_ is not None:
            fig_loss = plot_loss_history_interactive(reducer.loss_history_)
            sec.add_element(PlotlyElement(fig_loss, height="350px"))

        # Scree Plot (PCA)
        if hasattr(reducer, "explained_variance_ratio_"):
            fig_scree = plot_scree_interactive(reducer.explained_variance_ratio_)
            sec.add_element(PlotlyElement(fig_scree, height="350px"))

        self.add_section(sec)
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
        sec = Section(title=name, icon="🔍")

        # Extract array
        X = data
        names = None
        if hasattr(data, "X"):  # DataContainer
            X = data.X

        try:
            sample_X = X if X.size < 10000 else X.flat[:10000]
            res_flat = check_flatline(sample_X)
            if res_flat.is_issue:
                sec.add_finding(res_flat)

            res_outlier = check_outliers_zscore(sample_X)
            if res_outlier:
                sec.add_finding(res_outlier)
        except Exception:
            pass

        # Ensure 2D
        if hasattr(X, "ndim") and X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(X, "ndim") and X.ndim > 2:
            # Concatenating for flattened view
            X = X.reshape(X.shape[0] * X.shape[1], -1)

        fig = plot_raw_preview(X, names=names, title=name)
        sec.add_element(PlotlyElement(fig, height="450px"))

        self.add_section(sec)
        return self

    def render(self) -> str:
        """
        Render the full HTML report.
        Collates payloads, compresses data, and passes to template.
        """
        # 1. Collect Payload (Global Data Store)
        data_registry = {}
        self.collect_payload(data_registry)

        # 2. Compress Payload (JSON -> Gzip -> Base64)
        payload_json = json.dumps(data_registry).encode("utf-8")
        compressed = gzip.compress(payload_json)
        payload_b64 = base64.b64encode(compressed).decode("utf-8")

        # 3. Get content from children (Sections)
        # Note: Children now render with data-id references since collect_payload was called.
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
        )

    def save(self, filename: str) -> None:
        """
        Render and save the report to a file.

        Parameters
        ----------
        filename : str
            Path to save the HTML file.
        """
        full_html = self.render()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_html)
