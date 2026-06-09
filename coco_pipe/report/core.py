"""
Core Reporting Classes
======================

Defines the generic reporting primitives and dim-reduction report adapters used
to assemble single-file HTML reports.
"""

import base64
import gzip
import html
import io
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

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

logger = logging.getLogger(__name__)


def _get_reducer_summary(reducer: Any) -> Dict[str, Any]:
    """Collect the strict summary payload from a reduction-like object."""
    if not hasattr(reducer, "get_summary"):
        raise TypeError(
            "Reduction objects passed to Report.add_reduction() must implement "
            "get_summary()."
        )

    summary = reducer.get_summary()
    if not isinstance(summary, dict):
        raise TypeError("Reducer get_summary() must return a dictionary.")

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    metric_records = summary.get("metric_records")
    if not isinstance(metric_records, list):
        metric_records = []

    quality_metadata = summary.get("quality_metadata")
    if not isinstance(quality_metadata, dict):
        quality_metadata = {}

    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    interpretation = summary.get("interpretation")
    if not isinstance(interpretation, dict):
        interpretation = {}

    interpretation_records = summary.get("interpretation_records")
    if not isinstance(interpretation_records, list):
        interpretation_records = []

    return {
        "method": summary.get("method") or type(reducer).__name__,
        "metrics": metrics,
        "metric_records": metric_records,
        "quality_metadata": quality_metadata,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
        "interpretation_records": interpretation_records,
        "capabilities": summary.get("capabilities") or {},
    }


def _metrics_summary_table(metrics: Any) -> pd.DataFrame:
    """Reduce metric observations to a method x metric summary table."""
    from coco_pipe.viz.utils import prepare_metrics_frame

    metrics_df = prepare_metrics_frame(metrics)
    if metrics_df.empty:
        return pd.DataFrame()

    return metrics_df.pivot_table(
        index="method", columns="metric", values="value", aggfunc="mean"
    )


def _trajectory_times(
    diagnostics: Dict[str, Any], times: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Return the explicit trajectory time axis when it aligns with diagnostics."""
    if times is not None:
        time_values = np.asarray(times).reshape(-1)
        if time_values.size > 0:
            return time_values

    diagnostic_times = diagnostics.get("trajectory_times_")
    if diagnostic_times is None:
        return None

    time_values = np.asarray(diagnostic_times).reshape(-1)
    return time_values if time_values.size > 0 else None


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
            <img src="data:image/png;base64,{b64_str}" style="width: {self.width};"
                 class="rounded shadow-sm mx-auto border border-gray-100">
            {
            f'<figcaption class="text-center text-sm text-gray-500 mt-2">'
            f"{self.caption}</figcaption>"
            if self.caption
            else ""
        }
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

        json_str = self.figure.to_json()
        fig_dict = json.loads(json_str)

        fig_dict = self._force_standard_json(fig_dict)

        registry[self.registry_id] = fig_dict

    def _force_standard_json(self, obj: Any) -> Any:
        """Recursively convert Plotly binary-encoded arrays to standard lists."""
        if isinstance(obj, dict):
            # Check for Plotly binary format
            if "dtype" in obj and "bdata" in obj and len(obj) <= 3:
                # Identify keys like 'shape'? Usually just dtype/bdata.
                # Decode!
                try:
                    import base64

                    dtype = obj["dtype"]
                    bdata = obj["bdata"]

                    # Map dtype string to numpy type
                    # common: 'f4' (float32), 'f8' (float64), 'i4' (int32), 'u4'...
                    decoded = base64.b64decode(bdata)
                    arr = np.frombuffer(decoded, dtype=dtype)
                    return arr.tolist()
                except Exception as e:
                    print(f"[Report] Warning: Failed to decode binary data: {e}")
                    return obj

            return {k: self._force_standard_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._force_standard_json(x) for x in obj]
        return obj

    def render(self) -> str:
        # Instead of dumping JSON, we reference the ID
        if self.registry_id is None:
            return self._render_inline()

        html = f"""
        <div class="my-6">
            <div class="lazy-plot w-full rounded shadow-sm border border-gray-100
                        bg-gray-50 flex items-center justify-center text-gray-400
                        animate-pulse"
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
            <div class="lazy-plot w-full rounded shadow-sm border border-gray-100
                        bg-gray-50 flex items-center justify-center text-gray-400
                        animate-pulse"
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

    @staticmethod
    def _to_frame(data: Any) -> pd.DataFrame:
        """Normalize supported table-like inputs to a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            if all(
                isinstance(v, (int, float, str, np.number)) or v is None
                for v in data.values()
            ):
                return pd.DataFrame([data])
            return pd.DataFrame(data)
        return pd.DataFrame(data)

    def render(self) -> str:
        df = self._to_frame(self.data)

        # Basic Tailwind Styling
        html = '<div class="overflow-x-auto my-4 group relative">'
        if self.title:
            html += f"""
            <div class="flex justify-between items-center mb-2">
                <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300
                    uppercase tracking-wide">
                    {self.title}
                </h4>
                <button onclick="exportTableToCSV(
                    '{self.table_id}', '{self.title or "data"}')"
                    class="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200
                           dark:bg-gray-800 dark:hover:bg-gray-700 rounded
                           text-gray-500 transition opacity-0
                           group-hover:opacity-100">
                    ⬇ CSV
                </button>
            </div>
            """

        # Render Table
        # Render Table
        html += (
            f'<table id="{self.table_id}" class="min-w-full divide-y divide-gray-200 '
            'dark:divide-gray-700 border dark:border-gray-700 text-sm">'
        )

        # Header
        html += '<thead class="bg-gray-50 dark:bg-gray-800"><tr>'
        for col in df.columns:
            html += (
                f'<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 '
                f'dark:text-gray-400 uppercase tracking-wider">{col}</th>'
            )
        html += "</tr></thead>"

        # Body
        # Body
        html += (
            '<tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 '
            'dark:divide-gray-700">'
        )
        for idx, row in df.iterrows():
            html += self._render_row(row, idx)
        html += "</tbody></table></div>"

        return html

    def _render_row(self, row, idx) -> str:
        """Render a single row. Can be overridden."""
        html = "<tr>"
        for val in row:
            html += (
                f'<td class="px-4 py-3 whitespace-nowrap text-gray-700 '
                f'dark:text-gray-300">{val}</td>'
            )
        html += "</tr>"
        return html


class InteractiveTableElement(Element):
    """Render a payload-backed interactive data table."""

    def __init__(
        self,
        data: Any,
        title: Optional[str] = None,
        selector_columns: Optional[List[str]] = None,
        default_sort: Optional[Dict[str, str]] = None,
        page_size: int = 50,
    ):
        self.data = data
        self.title = title
        self.selector_columns = list(selector_columns or [])
        self.default_sort = dict(default_sort) if default_sort else None
        self.page_size = int(page_size)
        self.registry_id: Optional[str] = None

    def collect_payload(self, registry: Dict[str, Any]) -> None:
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        df = TableElement._to_frame(self.data)
        payload = {
            "columns": [str(column) for column in df.columns],
            "rows": json.loads(df.to_json(orient="records", date_format="iso")),
        }
        registry[self.registry_id] = payload

    def render(self) -> str:
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        config = {
            "title": self.title,
            "selector_columns": self.selector_columns,
            "default_sort": self.default_sort,
            "page_size": self.page_size,
        }
        config_json = html.escape(json.dumps(config), quote=True)
        title_html = ""
        if self.title:
            title_html = f"""
            <div class="flex justify-between items-center mb-3">
                <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300
                    uppercase tracking-wide">
                    {self.title}
                </h4>
            </div>
            """

        return f"""
        <div class="my-4">
            {title_html}
            <div class="interactive-table" data-id="{self.registry_id}"
                 data-config="{config_json}">
                <div class="rounded border border-gray-200 dark:border-gray-700
                            bg-white dark:bg-gray-900 p-4 text-sm text-gray-500
                            dark:text-gray-400">
                    Loading interactive table...
                </div>
            </div>
        </div>
        """


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
                style = (
                    "font-bold text-green-600 dark:text-green-400 bg-green-50 "
                    "dark:bg-green-900/20"
                )

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

    def add_markdown(self, text: str) -> "ContainerElement":
        """
        Add a markdown block.

        Note: Requires 'markdown' package. If not present, falls back to raw
        text in <pre>.
        """
        try:
            import markdown

            html = markdown.markdown(text, extensions=["extra"])
            # Wrap in prose class for consistent styling
            wrapper = (
                f'<div class="prose prose-sm max-w-none text-gray-700 '
                f'dark:text-gray-200 dark:prose-invert">{html}</div>'
            )
            self.add_element(HtmlElement(wrapper))
        except ImportError:
            # Fallback
            safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
            html = (
                f'<div class="whitespace-pre-wrap font-mono text-sm bg-gray-50 p-4 '
                f'rounded">{safe_text}</div>'
            )
            self.add_element(HtmlElement(html))
        return self

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
            # get_environment_info returns: timestamp_utc, os_platform,
            # python_version, command, git_hash, versions...
            # This matches ProvenanceConfig fields.
            from .config import ProvenanceConfig

            self.config.provenance = ProvenanceConfig(**raw_meta)

        self.metadata = self.config.provenance.model_dump()

    def add_section(self, section: Section) -> "Report":
        """Syntactic sugar for adding a Section."""
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
            sec = Section(title=name, icon="💾")

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

    def add_reduction(
        self,
        reducer: Any,
        name: Optional[str] = None,
        *,
        X_emb: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        times: Optional[np.ndarray] = None,
    ) -> "Report":
        """
        Add one scored and optionally interpreted reduction result to the report.

        Parameters
        ----------
        reducer : Any
            Reduction object implementing ``get_summary()``.
        name : str, optional
            Section title. Defaults to the reduction method name.
        X_emb : np.ndarray, optional
            Explicit embedding to visualize. When omitted, the section renders
            scalar summaries, diagnostics, and interpretation outputs only.
        labels : np.ndarray, optional
            Optional labels aligned with ``X_emb`` for embedding or trajectory
            plots.
        metadata : dict, optional
            Optional column-oriented metadata aligned with the sample axis of a
            2D embedding.
        times : np.ndarray, optional
            Optional explicit time axis aligned with the time dimension of a 3D
            trajectory tensor.

        Returns
        -------
        Report
            The report instance for fluent chaining.

        Raises
        ------
        ValueError
            If the supplied embedding or aligned plotting metadata are invalid.
        TypeError
            If ``reducer`` does not implement the strict summary contract.

        See Also
        --------
        coco_pipe.dim_reduction.core.DimReduction.get_summary
        coco_pipe.viz.plotly_utils.plot_embedding_interactive
        coco_pipe.viz.plotly_utils.plot_interpretation_interactive
        """
        summary = _get_reducer_summary(reducer)
        method_name = summary["method"]
        title = name or method_name
        sec = Section(title=title, icon="📉")

        if X_emb is not None:
            emb = np.asarray(X_emb)
            if emb.ndim == 2:
                from coco_pipe.viz.plotly_utils import plot_embedding_interactive

                fig = plot_embedding_interactive(
                    embedding=emb,
                    labels=labels,
                    metadata=metadata,
                    title=f"{title} Embedding",
                    dimensions=min(emb.shape[1], 3),
                )
                sec.add_element(PlotlyElement(fig))
            elif emb.ndim == 3:
                from coco_pipe.viz.plotly_utils import plot_trajectory_interactive

                time_values = _trajectory_times(summary["diagnostics"], times)
                fig = plot_trajectory_interactive(
                    emb,
                    times=time_values,
                    labels=labels,
                    title=f"{title} Trajectory",
                    dimensions=min(emb.shape[-1], 3),
                )
                sec.add_element(PlotlyElement(fig))
            else:
                msg = "`X_emb` must be a 2D embedding or 3D trajectory tensor."
                raise ValueError(msg)

        metrics = summary["metrics"]
        quality_metadata = summary["quality_metadata"]
        scalar_table = {
            **{
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.number))
                and not isinstance(value, bool)
            },
            **{
                key: value
                for key, value in quality_metadata.items()
                if isinstance(value, (int, float, np.number))
                and not isinstance(value, bool)
            },
        }
        if scalar_table:
            sec.add_element(TableElement(scalar_table, title="Quality Metrics"))

        metric_records = summary["metric_records"]
        if metric_records:
            from coco_pipe.viz.plotly_utils import plot_metric_details

            sec.add_element(
                PlotlyElement(
                    plot_metric_details(metric_records, title="Metric Details"),
                    height="380px",
                )
            )

        diagnostics = summary["diagnostics"]

        loss_history = diagnostics.get("loss_history_")
        if loss_history is not None:
            from coco_pipe.viz.plotly_utils import plot_loss_history_interactive

            sec.add_element(
                PlotlyElement(
                    plot_loss_history_interactive(loss_history),
                    height="350px",
                )
            )

        explained_variance = diagnostics.get("explained_variance_ratio_")
        if explained_variance is not None:
            from coco_pipe.viz.plotly_utils import plot_scree_interactive

            sec.add_element(
                PlotlyElement(
                    plot_scree_interactive(explained_variance),
                    height="350px",
                )
            )

        coranking = diagnostics.get("coranking_matrix_")
        if coranking is not None:
            import plotly.graph_objects as go

            fig_coranking = go.Figure(
                data=[
                    go.Heatmap(
                        z=np.asarray(coranking),
                        colorscale="Viridis",
                        colorbar=dict(title="Count"),
                    )
                ]
            )
            fig_coranking.update_layout(
                title="Co-Ranking Matrix",
                xaxis_title="Embedded Rank",
                yaxis_title="Original Rank",
                margin=dict(l=40, r=40, b=40, t=40),
                template="plotly_white",
            )
            sec.add_element(PlotlyElement(fig_coranking, height="420px"))

        from coco_pipe.viz.plotly_utils import (
            plot_interpretation_interactive,
            plot_trajectory_metric_series_interactive,
        )

        time_values = _trajectory_times(diagnostics, times)
        trajectory_series = (
            "trajectory_speed_",
            "trajectory_acceleration_",
            "trajectory_curvature_",
            "trajectory_turning_angle_",
            "trajectory_dispersion_",
            "trajectory_path_length_",
            "trajectory_displacement_",
        )
        for metric_key in trajectory_series:
            values = diagnostics.get(metric_key)
            if values is None:
                continue
            sec.add_element(
                PlotlyElement(
                    plot_trajectory_metric_series_interactive(
                        values,
                        times=time_values,
                        labels=labels,
                        title=metric_key.rstrip("_").replace("_", " ").title(),
                        ylabel="Value",
                    ),
                    height="360px",
                )
            )

        separation = diagnostics.get("trajectory_separation_")
        if separation is not None:
            sec.add_element(
                PlotlyElement(
                    plot_trajectory_metric_series_interactive(
                        separation,
                        times=time_values,
                        title="Trajectory Separation",
                        ylabel="Separation",
                    ),
                    height="380px",
                )
            )

        interpretation = summary["interpretation"]
        if interpretation:
            interpretation_payload = {
                "analysis": interpretation,
                "records": summary["interpretation_records"],
            }
            record_analyses = {
                record.get("analysis")
                for record in summary["interpretation_records"]
                if isinstance(record, dict) and record.get("analysis")
            }
            analysis_names = sorted(record_analyses | set(interpretation.keys()))
            for analysis_name in analysis_names:
                sec.add_element(
                    PlotlyElement(
                        plot_interpretation_interactive(
                            interpretation_payload,
                            analysis=analysis_name,
                            title=(
                                f"{analysis_name.replace('_', ' ').title()} "
                                "Interpretation"
                            ),
                            method=method_name,
                        ),
                        height="420px",
                    )
                )

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
        except Exception as e:
            logger.debug(f"Data quality checks failed: {e}")

        # Ensure 2D
        if hasattr(X, "ndim") and X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(X, "ndim") and X.ndim > 2:
            # Concatenating for flattened view
            X = X.reshape(X.shape[0] * X.shape[1], -1)

        from coco_pipe.viz.plotly_utils import plot_raw_preview

        fig = plot_raw_preview(X, names=names, title=name)
        sec.add_element(PlotlyElement(fig, height="450px"))

        self.add_section(sec)
        return self

    def add_comparison(
        self, metrics_df: Any, name: str = "Method Comparison"
    ) -> "Report":
        """
        Add a comparison section for multiple reduction methods.

        Parameters
        ----------
        metrics_df : DataFrame or MethodSelector-like
            Wide/tidy metric data or an object exposing ``to_frame()``.
        name : str
            Section title.

        Returns
        -------
        Report
            The report instance for fluent chaining.

        Raises
        ------
        ValueError
            If no comparison metrics are available after normalization.

        See Also
        --------
        coco_pipe.viz.plotly_utils.plot_metric_details
        coco_pipe.dim_reduction.evaluation.core.MethodSelector
        """
        sec = Section(title=name, icon="📊")
        from coco_pipe.viz.plotly_utils import (
            plot_metric_details,
            plot_radar_comparison,
        )
        from coco_pipe.viz.utils import infer_metric_plot_type, prepare_metrics_frame

        long_df = prepare_metrics_frame(metrics_df)
        summary_table = _metrics_summary_table(long_df)

        if summary_table.empty:
            raise ValueError("No comparison metrics available to add to the report.")

        # 1. Metrics Table (Best values highlighted)
        sec.add_element(MetricsTableElement(summary_table, title="Quality Metrics"))

        # 2. Primary visual summaries
        fig_heatmap = plot_metric_details(
            long_df, title="Metric Heatmap", plot_type="heatmap"
        )
        sec.add_element(PlotlyElement(fig_heatmap, height="400px"))

        primary_plot_type = infer_metric_plot_type(long_df)
        fig_primary = plot_metric_details(
            long_df, title="Metric Details", plot_type=primary_plot_type
        )
        sec.add_element(PlotlyElement(fig_primary, height="400px"))

        if (
            long_df["scope_value"].astype(str).nunique() == 1
            and summary_table.shape[1] >= 3
            and summary_table.shape[0] >= 2
        ):
            fig_radar = plot_radar_comparison(summary_table, normalize=True)
            sec.add_element(PlotlyElement(fig_radar, height="400px"))

        self.add_section(sec)
        return self

    def add_decoding_temporal(
        self,
        result: Any,
        metric: Optional[str] = None,
        model: Optional[str] = None,
        name: str = "Temporal Decoding",
    ) -> "Report":
        """
        Add a compact temporal decoding section from an ExperimentResult.
        """
        from coco_pipe.viz.decoding import (
            plot_temporal_generalization_matrix,
            plot_temporal_score_curve,
        )

        if not hasattr(result, "get_temporal_score_summary"):
            raise TypeError(
                "result must provide get_temporal_score_summary() for temporal "
                "decoding report sections."
            )

        summary = result.get_temporal_score_summary()
        if metric is not None:
            summary = summary[summary["Metric"] == metric]
        if model is not None:
            summary = summary[summary["Model"] == model]
        if summary.empty:
            raise ValueError("No temporal decoding scores available for report.")

        sec = Section(title=name, icon="📈")
        sec.add_element(TableElement(summary, title="Temporal Score Summary"))

        if "Time" in summary and summary["Time"].notna().any():
            fig_curve = plot_temporal_score_curve(
                summary, metric=metric, model=model, title="Temporal Score Curve"
            )
            sec.add_element(ImageElement(fig_curve, caption="Temporal score curve"))

        if (
            {"TrainTime", "TestTime"}.issubset(summary.columns)
            and summary["TrainTime"].notna().any()
            and summary["TestTime"].notna().any()
        ):
            fig_matrix = plot_temporal_generalization_matrix(
                summary,
                metric=metric,
                model=model,
                title="Temporal Generalization Matrix",
            )
            sec.add_element(
                ImageElement(fig_matrix, caption="Temporal generalization matrix")
            )

    def add_decoding_summary(
        self,
        result: Any,
        name: str = "Decoding Summary",
    ) -> "Report":
        """Add a summary performance table for all models."""
        if not hasattr(result, "summary"):
            raise TypeError("result must provide a summary() method.")

        summary = result.summary()
        if summary.empty:
            return self

        sec = Section(title=name)
        sec.add_element(
            MetricsTableElement(
                summary,
                title="Model Performance Summary",
                highlight_best=True,
            )
        )
        self.add_section(sec)
        return self

    def add_decoding_diagnostics(
        self,
        result: Any,
        metric: Optional[str] = None,
        model: Optional[str] = None,
        name: str = "Decoding Diagnostics",
    ) -> "Report":
        """Add shallow decoding diagnostics from an ExperimentResult."""
        from coco_pipe.viz.decoding import (
            plot_calibration_curve,
            plot_confusion_matrix,
            plot_fold_score_dispersion,
            plot_pr_curve,
            plot_roc_curve,
        )

        required = [
            "get_detailed_scores",
            "get_fit_diagnostics",
            "get_confusion_matrices",
        ]
        if not all(hasattr(result, attr) for attr in required):
            raise TypeError("result must provide decoding diagnostic accessors.")

        sec = Section(title=name)

        meta = getattr(result, "meta", {}) or {}
        if meta.get("observation_level") or meta.get("inferential_unit"):
            inference_context = pd.DataFrame(
                [
                    {
                        "ObservationLevel": meta.get("observation_level", "sample"),
                        "InferentialUnit": meta.get("inferential_unit", "sample"),
                    }
                ]
            )
            sec.add_element(TableElement(inference_context, title="Inference Context"))

        scores = result.get_detailed_scores()
        if metric is not None and "Metric" in scores:
            scores = scores[scores["Metric"] == metric]
        if model is not None and "Model" in scores:
            scores = scores[scores["Model"] == model]
        if not scores.empty:
            sec.add_element(TableElement(scores, title="Fold Scores"))
            try:
                fig_scores = plot_fold_score_dispersion(
                    scores,
                    metric=metric,
                    model=model,
                    title="Fold Score Dispersion",
                )
                sec.add_element(
                    ImageElement(fig_scores, caption="Fold score dispersion")
                )
            except ValueError as e:
                logger.debug(f"Could not plot fold score dispersion: {e}")

        diagnostics = result.get_fit_diagnostics()
        if model is not None and "Model" in diagnostics:
            diagnostics = diagnostics[diagnostics["Model"] == model]
        if not diagnostics.empty:
            # 1. Clean Timing Table
            timing_cols = ["Model", "Fold", "FitTime", "PredictTime", "TotalTime"]
            timing_cols = [c for c in timing_cols if c in diagnostics.columns]
            timings = diagnostics[timing_cols].drop_duplicates()
            sec.add_element(TableElement(timings, title="Fit Diagnostics"))

            # 2. Warnings Table (only if they exist)
            warns = diagnostics[diagnostics["WarningMessage"].notna()]
            if not warns.empty:
                warn_cols = [
                    "Model",
                    "Fold",
                    "Stage",
                    "WarningCategory",
                    "WarningMessage",
                ]
                warn_cols = [c for c in warn_cols if c in warns.columns]
                sec.add_element(
                    TableElement(warns[warn_cols], title="Training Warnings")
                )

        confusion = result.get_confusion_matrices(model=model)
        if not confusion.empty:
            sec.add_element(TableElement(confusion, title="Confusion Matrix Data"))
            try:
                fig_confusion = plot_confusion_matrix(
                    confusion,
                    model=model,
                    title="Confusion Matrix",
                )
                sec.add_element(ImageElement(fig_confusion, caption="Confusion matrix"))
            except ValueError as e:
                logger.debug(f"Could not plot confusion matrix: {e}")

        for title, plotter in [
            ("ROC Curve", plot_roc_curve),
            ("Precision-Recall Curve", plot_pr_curve),
            ("Calibration Curve", plot_calibration_curve),
        ]:
            try:
                fig = plotter(result, model=model, title=title)
                sec.add_element(ImageElement(fig, caption=title))
            except ValueError as e:
                logger.debug(f"Could not plot curve '{title}': {e}")

        self.add_section(sec)
        return self

    def add_decoding_statistical_assessment(
        self,
        result: Any,
        metric: Optional[str] = None,
        model: Optional[str] = None,
        name: str = "Statistical Assessment",
    ) -> "Report":
        """Add finite-sample decoding statistical assessment rows and plots."""
        from coco_pipe.viz.decoding import plot_temporal_statistical_assessment

        if not hasattr(result, "get_statistical_assessment"):
            raise TypeError("result must provide get_statistical_assessment().")

        assessment = result.get_statistical_assessment()
        if metric is not None and "Metric" in assessment:
            assessment = assessment[assessment["Metric"] == metric]
        if model is not None and "Model" in assessment:
            assessment = assessment[assessment["Model"] == model]
        if assessment.empty:
            raise ValueError("No statistical assessment rows available for report.")

        sec = Section(title=name)
        sec.add_element(
            TableElement(assessment, title="Finite-Sample Statistical Assessment")
        )

        if "Time" in assessment and assessment["Time"].notna().any():
            try:
                fig = plot_temporal_statistical_assessment(
                    assessment,
                    metric=metric,
                    model=model,
                    title="Temporal Statistical Assessment",
                )
                sec.add_element(
                    ImageElement(fig, caption="Temporal statistical assessment")
                )
            except ValueError as e:
                logger.debug(f"Could not plot temporal statistical assessment: {e}")

        self.add_section(sec)
        return self

    def add_decoding_neural_artifacts(
        self,
        result: Any,
        model: Optional[str] = None,
        name: str = "Neural Artifacts",
    ) -> "Report":
        """Add neural/foundation-model artifact metadata to the report."""
        from coco_pipe.viz.decoding import plot_training_history

        if not hasattr(result, "get_model_artifacts"):
            raise TypeError("result must provide get_model_artifacts().")
        artifacts = result.get_model_artifacts()
        if model is not None and "Model" in artifacts:
            artifacts = artifacts[artifacts["Model"] == model]
        if artifacts.empty:
            raise ValueError("No model artifacts available for report.")

        sec = Section(title=name)
        sec.add_element(TableElement(artifacts, title="Model Artifacts"))
        try:
            fig = plot_training_history(artifacts, model=model)
            sec.add_element(ImageElement(fig, caption="Training history"))
        except ValueError:
            pass
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
