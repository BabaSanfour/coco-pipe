"""
Reporting Elements
==================

Generic, reusable report primitives. Each ``Element`` knows how to render
itself to HTML and (optionally) contribute to the global data payload used
for interactive widgets.

Higher-level constructs (:class:`coco_pipe.report.core.Section`,
:class:`coco_pipe.report.core.Report`) live in :mod:`coco_pipe.report.core`.
"""

import base64
import html
import io
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _render_markdown(text: str, *, inline: bool = False) -> str:
    """Render markdown ``text`` to HTML, escaping if the library is missing.

    When ``inline`` is True and the output is a single top-level paragraph,
    the wrapping ``<p>`` tags are stripped so the result can be dropped into
    an existing inline context (e.g. a callout body).
    """
    try:
        import markdown
    except ImportError:
        return html.escape(text)

    rendered = markdown.markdown(text, extensions=["extra"])
    if (
        inline
        and rendered.count("<p>") == 1
        and rendered.startswith("<p>")
        and rendered.endswith("</p>")
    ):
        rendered = rendered[len("<p>") : -len("</p>")]
    return rendered


class Element(ABC):
    """
    Abstract base class for all report elements.
    """

    @abstractmethod
    def render(self) -> str:
        """Render the element to HTML."""

    def collect_payload(self, registry: dict[str, Any]) -> None:  # noqa: B027
        """
        Collect data to be stored in the global payload.
        Default implementation does nothing.

        Parameters
        ----------
        registry : Dict[str, Any]
            Global dictionary accumulating data. Keyed by UUID.
        """


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

    def __init__(self, src: Any, caption: str | None = None, width: str = "100%"):
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

        if isinstance(self.src, (str, Path)):
            p = Path(self.src)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode("utf-8")

        raise ValueError(f"Unsupported image source type: {type(self.src)}")

    def render(self) -> str:
        b64_str = self._encode_image()
        caption_text = html.escape(str(self.caption)) if self.caption else ""
        caption_html = (
            f'<figcaption class="text-center text-sm text-gray-500 mt-2">'
            f"{caption_text}</figcaption>"
            if self.caption
            else ""
        )
        filename_root = re.sub(
            r"[^a-z0-9]+", "_", str(self.caption or "figure").lower()
        ).strip("_")
        filename = f"{filename_root or 'figure'}.png"
        download_html = (
            f'<div class="flex justify-end mt-1 mb-0">'
            f'<a href="data:image/png;base64,{b64_str}" download="{filename}"'
            f' aria-label="Download figure as PNG"'
            f' class="inline-flex items-center gap-1 text-xs text-gray-400'
            f" hover:text-brand-600 dark:hover:text-brand-400 transition px-2 py-0.5"
            f' rounded hover:bg-gray-100 dark:hover:bg-gray-800">'
            f'<svg class="h-3.5 w-3.5" fill="none" stroke="currentColor"'
            f' viewBox="0 0 24 24" aria-hidden="true">'
            f'<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"'
            f' d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14"></path>'
            f"</svg><span>PNG</span></a>"
            f"</div>"
        )
        html_out = (
            f'<figure class="my-6">'
            f"{download_html}"
            f'<img src="data:image/png;base64,{b64_str}" style="width: {self.width};"'
            f' class="rounded shadow-sm mx-auto border border-gray-100">'
            f"{caption_html}"
            f"</figure>"
        )
        return html_out


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

    def collect_payload(self, registry: dict[str, Any]) -> None:
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
                except Exception as exc:
                    logger.debug("Failed to decode Plotly binary array: %s", exc)
                    return obj

            return {k: self._force_standard_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
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
        safe_json = html.escape(json_str, quote=True)

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
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    >>> elem = TableElement(df, title="Metrics")
    """

    def __init__(self, data: Any, title: str | None = None):
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
        title_text = str(self.title or "data")
        title_html = html.escape(title_text)
        title_json = json.dumps(title_text)

        html_out = '<div class="overflow-x-auto my-4 group relative">'
        if self.title:
            html_out += f"""
            <div class="flex justify-between items-center mb-2">
                <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300
                    uppercase tracking-wide">
                    {title_html}
                </h4>
                <button onclick="exportTableToCSV(
                    '{self.table_id}', {title_json})"
                    aria-label="Export {title_html} as CSV"
                    class="inline-flex items-center gap-1 text-xs px-2 py-1 bg-gray-100
                    hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 rounded
                    text-gray-500 transition opacity-0 group-hover:opacity-100">
                    <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor"
                        viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round"
                            stroke-width="2" d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14">
                        </path>
                    </svg>
                    <span>CSV</span>
                </button>
            </div>
            """

        html_out += (
            f'<table id="{self.table_id}" class="min-w-full divide-y divide-gray-200 '
            'dark:divide-gray-700 border dark:border-gray-700 text-sm">'
        )

        # Header
        html_out += '<thead class="bg-gray-50 dark:bg-gray-800"><tr>'
        for col in df.columns:
            html_out += (
                f'<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 '
                f'dark:text-gray-400 uppercase tracking-wider">'
                f"{html.escape(str(col))}</th>"
            )
        html_out += "</tr></thead>"

        # Body
        html_out += (
            '<tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 '
            'dark:divide-gray-700">'
        )
        for idx, row in df.iterrows():
            html_out += self._render_row(row, idx)
        html_out += "</tbody></table></div>"

        return html_out

    def _render_row(self, row, idx) -> str:
        """Render a single row. Can be overridden."""
        html_out = "<tr>"
        for val in row:
            html_out += (
                f'<td class="px-4 py-3 whitespace-nowrap text-gray-700 '
                f'dark:text-gray-300">{html.escape(str(val))}</td>'
            )
        html_out += "</tr>"
        return html_out


class InteractiveTableElement(Element):
    """Render a payload-backed interactive data table."""

    def __init__(
        self,
        data: Any,
        title: str | None = None,
        selector_columns: list[str] | None = None,
        default_sort: dict[str, str] | None = None,
        page_size: int = 50,
    ):
        self.data = data
        self.title = title
        self.selector_columns = list(selector_columns or [])
        self.default_sort = dict(default_sort) if default_sort else None
        self.page_size = int(page_size)
        self.registry_id: str | None = None

    def collect_payload(self, registry: dict[str, Any]) -> None:
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        df = TableElement._to_frame(self.data)
        rows = df.to_dict(orient="records")
        payload = {
            "columns": [str(column) for column in df.columns],
            "rows": rows,
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
        highlight_cols: list[str] | None = None,
        higher_is_better: bool | list[str] = True,
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
                hib = self.higher_is_better
                is_higher = col in hib if isinstance(hib, list) else bool(hib)

                if is_higher:
                    self.best_vals[col] = self.data[col].max()
                else:
                    self.best_vals[col] = self.data[col].min()

    def _render_row(self, row, idx) -> str:
        html_out = "<tr>"
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

            html_out += (
                f'<td class="px-4 py-3 whitespace-nowrap {style}">'
                f"{html.escape(str(display_val))}</td>"
            )
        html_out += "</tr>"
        return html_out


class StatCardElement(Element):
    """
    A KPI-style card displaying a single headline metric.

    Parameters
    ----------
    label : str
        Short metric label (e.g. ``"Mean Accuracy"``).
    value : str or float
        The metric value to display prominently.
    unit : str, optional
        Optional unit suffix appended to the value (e.g. ``"%"``).
    delta : str, optional
        Optional change indicator rendered below the value (e.g. ``"+2.3%"``).
    color : str, optional
        Accent color for the top stripe: ``"blue"`` (default), ``"green"``,
        ``"yellow"``, ``"red"``, or ``"purple"``.

    Examples
    --------
    >>> card = StatCardElement("Mean Accuracy", 0.842, unit="%", color="green")
    >>> report.add_summary_card({"Mean Accuracy": 0.842, "Models": 3})
    """

    _COLORS: ClassVar[dict[str, tuple[str, str]]] = {
        "blue": ("bg-blue-500", "text-blue-600 dark:text-blue-400"),
        "green": ("bg-green-500", "text-green-600 dark:text-green-400"),
        "yellow": ("bg-yellow-500", "text-yellow-600 dark:text-yellow-400"),
        "red": ("bg-red-500", "text-red-600 dark:text-red-400"),
        "purple": ("bg-purple-500", "text-purple-600 dark:text-purple-400"),
    }

    def __init__(
        self,
        label: str,
        value: Any,
        unit: str = "",
        delta: str | None = None,
        color: str = "blue",
    ):
        self.label = label
        self.value = f"{value:.4g}" if isinstance(value, float) else str(value)
        self.unit = unit
        self.delta = delta
        self.color = color

    def render(self) -> str:
        stripe_cls, val_cls = self._COLORS.get(self.color, self._COLORS["blue"])
        delta_html = (
            f'<div class="text-xs text-gray-500 dark:text-gray-400 mt-1">'
            f"{self.delta}</div>"
            if self.delta
            else ""
        )
        return (
            f'<div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border '
            f'border-gray-100 dark:border-gray-700 overflow-hidden">'
            f'<div class="{stripe_cls} h-1"></div>'
            f'<div class="p-4">'
            f'<div class="text-xs text-gray-500 dark:text-gray-400 uppercase '
            f'tracking-wide font-medium mb-1">{self.label}</div>'
            f'<div class="text-2xl font-bold {val_cls}">'
            f'{self.value}<span class="text-sm font-normal ml-0.5">{self.unit}</span>'
            f"</div>"
            f"{delta_html}"
            f"</div></div>"
        )


class CalloutElement(Element):
    """
    A tinted callout box for notes, tips, warnings, or errors.

    Parameters
    ----------
    text : str
        The callout body text (plain text or minimal HTML).
    kind : str, optional
        Visual style: ``"info"`` (default), ``"tip"``, ``"warning"``, or ``"error"``.
    title : str, optional
        Optional bold heading rendered above the body text.

    Examples
    --------
    >>> note = CalloutElement(
    ...     "Probability calibration was unavailable.", kind="warning"
    ... )
    >>> sec.add_element(note)
    """

    _STYLES: ClassVar[dict[str, tuple[str, str]]] = {
        "info": (
            "bg-blue-50 border-blue-400 text-blue-800 "
            "dark:bg-blue-900/20 dark:text-blue-200",
            "INFO",
        ),
        "tip": (
            "bg-green-50 border-green-400 text-green-800 "
            "dark:bg-green-900/20 dark:text-green-200",
            "TIP",
        ),
        "warning": (
            "bg-yellow-50 border-yellow-400 text-yellow-800 "
            "dark:bg-yellow-900/20 dark:text-yellow-200",
            "WARN",
        ),
        "error": (
            "bg-red-50 border-red-400 text-red-800 "
            "dark:bg-red-900/20 dark:text-red-200",
            "ERR",
        ),
    }

    def __init__(self, text: str, kind: str = "info", title: str | None = None):
        self.text = text
        self.kind = kind
        self.title = title

    def render(self) -> str:
        cls, icon = self._STYLES.get(self.kind, self._STYLES["info"])
        title_html = (
            f'<div class="font-bold mb-1">{html.escape(self.title)}</div>'
            if self.title
            else ""
        )
        text_html = _render_markdown(self.text, inline=True)
        return (
            f'<div class="flex items-start p-4 my-3 rounded-md border-l-4 '
            f'text-sm {cls}">'
            f'<span class="mr-3 shrink-0 rounded border px-1.5 py-0.5 text-[10px] '
            f'font-semibold tracking-wide">{icon}</span>'
            f"<div>{title_html}{text_html}</div>"
            f"</div>"
        )


class CodeBlockElement(Element):
    """
    A syntax-highlighted, optionally copyable code block.

    Parameters
    ----------
    code : str
        The code or text to display verbatim.
    language : str, optional
        Language hint shown as a badge (e.g. ``"python"``, ``"json"``). Cosmetic only.
    title : str, optional
        Optional caption rendered in the block header.
    copyable : bool, optional
        If True (default), renders a one-click Copy button.

    Examples
    --------
    >>> block = CodeBlockElement(
    ...     json.dumps(config, indent=2), language="json", title="Run Configuration"
    ... )
    >>> sec.add_element(block)
    """

    def __init__(
        self,
        code: str,
        language: str = "",
        title: str | None = None,
        copyable: bool = True,
    ):
        self._code = code
        self.language = language
        self.title = title
        self.copyable = copyable
        self._id = f"cb-{uuid.uuid4().hex[:8]}"

    def render(self) -> str:
        escaped = html.escape(self._code)
        header = ""
        if self.title or self.language or self.copyable:
            lang_badge = (
                f'<span class="text-xs font-mono text-gray-400">'
                f"{html.escape(self.language)}</span>"
                if self.language
                else ""
            )
            title_span = (
                f'<span class="text-xs font-semibold text-gray-300">'
                f"{html.escape(self.title)}</span>"
                if self.title
                else ""
            )
            copy_btn = ""
            if self.copyable:
                copy_btn = (
                    f'<button onclick="navigator.clipboard.writeText('
                    f"document.getElementById('{self._id}').textContent);"
                    f"this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)\""
                    f' class="text-xs px-2 py-0.5 bg-gray-700 hover:bg-gray-600 '
                    f'text-gray-300 rounded transition">Copy</button>'
                )
            header = (
                f'<div class="flex justify-between items-center px-4 py-2 '
                f'bg-gray-800 border-b border-gray-700 rounded-t-lg">'
                f'<div class="flex items-center gap-2">{title_span}{lang_badge}</div>'
                f"{copy_btn}</div>"
            )
        return (
            f'<div class="my-4 rounded-lg overflow-hidden border '
            f'border-gray-700 shadow-sm">'
            f"{header}"
            f'<pre id="{self._id}" class="bg-gray-900 text-gray-100 p-4 text-xs '
            f'font-mono overflow-x-auto leading-relaxed">{escaped}</pre>'
            f"</div>"
        )


class ContainerElement(Element):
    """
    Base class for elements that contain other elements.
    """

    def __init__(self):
        self.children: list[Element] = []

    def add_element(self, element: Element | str):
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
        """
        self.add_element(MarkdownElement(text))
        return self

    def render_children(self) -> str:
        """Render all child elements concatenated."""
        return "\n".join([c.render() for c in self.children])

    def collect_payload(self, registry: dict[str, Any]) -> None:
        """Recursively collect payload from children."""
        for child in self.children:
            child.collect_payload(registry)

    def render(self) -> str:
        return self.render_children()


class ColumnsElement(ContainerElement):
    """
    Lay out child elements side by side in a responsive CSS grid row.

    Parameters
    ----------
    elements : list of Element
        Child elements to place in columns. Each element occupies one cell.
    cols : int, optional
        Number of columns. Inferred from ``len(elements)`` when omitted, capped at 4.
    gap : str, optional
        Tailwind gap class applied to the grid. Default ``"gap-4"``.

    Examples
    --------
    >>> row = ColumnsElement(
    ...     [ImageElement(fig1, caption="ROC"), ImageElement(fig2, caption="PR")]
    ... )
    >>> sec.add_element(row)
    """

    def __init__(
        self,
        elements: list[Element],
        cols: int | None = None,
        gap: str = "gap-4",
    ):
        super().__init__()
        for elem in elements:
            self.add_element(elem)
        self._cols = cols
        self._gap = gap

    def render(self) -> str:
        n = self._cols or min(len(self.children), 4) or 1
        parts = [f'<div class="grid grid-cols-1 sm:grid-cols-{n} {self._gap} my-4">']
        for child in self.children:
            parts.append(f'<div class="min-w-0">{child.render()}</div>')
        parts.append("</div>")
        return "\n".join(parts)


class AccordionElement(ContainerElement):
    """
    A collapsible ``<details>`` disclosure block for secondary content.

    Parameters
    ----------
    summary : str
        The always-visible summary label (acts as the toggle trigger).
    open : bool, optional
        If True the block starts expanded. Default False.

    Examples
    --------
    >>> acc = AccordionElement("Show raw configuration")
    >>> acc.add_element(CodeBlockElement(json_str, language="json"))
    >>> sec.add_element(acc)
    """

    def __init__(self, summary: str, open: bool = False):
        super().__init__()
        self._summary = summary
        self._open = open

    def render(self) -> str:
        open_attr = " open" if self._open else ""
        content = self.render_children()
        return (
            f'<details class="my-3 rounded-lg border border-gray-200 '
            f"dark:border-gray-700 bg-white dark:bg-gray-800 "
            f'overflow-hidden"{open_attr}>'
            f'<summary class="cursor-pointer px-4 py-3 text-sm font-medium '
            f"text-gray-700 dark:text-gray-300 hover:bg-gray-50 "
            f"dark:hover:bg-gray-700/50 "
            f'transition list-none flex items-center justify-between">'
            f"<span>{self._summary}</span>"
            f'<svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" '
            f'viewBox="0 0 24 24">'
            f'<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" '
            f'd="M19 9l-7 7-7-7"/></svg></summary>'
            f'<div class="px-4 pb-4 pt-2">{content}</div>'
            f"</details>"
        )


class MarkdownElement(Element):
    """Render markdown text as HTML."""

    def __init__(self, text: str):
        self.text = text

    def render(self) -> str:
        try:
            import markdown

            html_content = markdown.markdown(self.text, extensions=["extra"])
            return (
                f'<div class="prose prose-sm max-w-none text-gray-700 '
                f'dark:text-gray-200 dark:prose-invert">{html_content}</div>'
            )
        except ImportError:
            safe_text = html.escape(self.text)
            return (
                f'<div class="whitespace-pre-wrap font-mono text-sm bg-gray-50 '
                f'dark:bg-gray-800 p-4 rounded">{safe_text}</div>'
            )


class BadgeElement(Element):
    """A small inline status badge."""

    def __init__(self, text: str, color: str = "blue"):
        self.text = text
        self.color = color

    def render(self) -> str:
        return (
            f'<span class="inline-flex items-center rounded-md bg-{self.color}-50 '
            f"px-2 py-1 text-xs font-medium text-{self.color}-700 ring-1 ring-inset "
            f"ring-{self.color}-700/10 dark:bg-{self.color}-900/30 "
            f'dark:text-{self.color}-300 dark:ring-{self.color}-400/20">'
            f"{html.escape(self.text)}</span>"
        )


class ProgressBarElement(Element):
    """A visual progress bar or gauge."""

    def __init__(
        self,
        value: float,
        max_value: float = 100.0,
        label: str | None = None,
        color: str = "blue",
    ):
        self.value = max(0.0, min(value, max_value))
        self.max_value = max_value
        self.label = label
        self.color = color

    def render(self) -> str:
        pct = (self.value / self.max_value) * 100
        label_html = (
            f'<div class="mb-1 text-sm font-medium text-gray-700 '
            f'dark:text-gray-300">{html.escape(self.label)}</div>'
            if self.label
            else ""
        )
        return (
            f'<div class="w-full my-2">'
            f"{label_html}"
            f'<div class="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 '
            f'overflow-hidden">'
            f'<div class="bg-{self.color}-600 h-2.5 rounded-full transition-all '
            f'duration-500" style="width: {pct}%"></div>'
            f"</div>"
            f"</div>"
        )


class TimelineElement(Element):
    """A vertical timeline of events."""

    def __init__(self, events: list[dict[str, str]]):
        self.events = events

    def render(self) -> str:
        parts = [
            '<ol class="relative border-l border-gray-200 '
            'dark:border-gray-700 ml-3 my-4">'
        ]
        for event in self.events:
            title = html.escape(event.get("title", ""))
            time_str = html.escape(event.get("time", ""))
            desc = html.escape(event.get("description", ""))
            status = event.get("status", "blue")

            parts.append(
                f'<li class="mb-6 ml-4">'
                f'<div class="absolute w-3 h-3 bg-{status}-500 rounded-full mt-1.5 '
                f'-left-1.5 border border-white dark:border-gray-900"></div>'
                f'<time class="mb-1 text-xs font-normal leading-none text-gray-400 '
                f'dark:text-gray-500">{time_str}</time>'
                f'<h3 class="text-sm font-semibold text-gray-900 '
                f'dark:text-white">{title}</h3>'
                f'<p class="mb-4 text-sm font-normal text-gray-500 '
                f'dark:text-gray-400">{desc}</p>'
                f"</li>"
            )
        parts.append("</ol>")
        return "".join(parts)


class TabsElement(ContainerElement):
    """A tabbed container for multiple elements."""

    def __init__(self, tabs: dict[str, Element]):
        super().__init__()
        self.tabs = tabs
        self._group_id = str(uuid.uuid4())[:8]
        for elem in tabs.values():
            self.add_element(elem)

    def render(self) -> str:
        parts = [f'<div id="tabs-{self._group_id}" class="coco-tabs">']
        parts.append(
            '<div class="coco-tab-bar flex border-b border-gray-200 '
            'dark:border-gray-700 mb-4">'
        )
        for idx, (title, _) in enumerate(self.tabs.items()):
            active_class = (
                "text-blue-600 border-blue-600 dark:text-blue-500 "
                "dark:border-blue-500 active-tab"
                if idx == 0
                else "border-transparent hover:text-gray-600 "
                "hover:border-gray-300 dark:hover:text-gray-300"
            )
            parts.append(
                f'<button class="inline-flex items-center justify-center p-4 '
                f'border-b-2 rounded-t-lg group transition-colors {active_class}" '
                f"onclick=\"cocoSwitchTab(this, 'panel-{self._group_id}-{idx}', "
                f"'tabs-{self._group_id}')\">"
                f"{html.escape(title)}"
                f"</button>"
            )
        parts.append("</div>")

        parts.append('<div class="coco-tab-panels">')
        for idx, (_, elem) in enumerate(self.tabs.items()):
            hidden = "" if idx == 0 else "hidden"
            parts.append(
                f'<div id="panel-{self._group_id}-{idx}" '
                f'class="tab-panel {hidden}">{elem.render()}</div>'
            )
        parts.append("</div>")
        parts.append("</div>")

        script = """
<script>
if (typeof window.cocoSwitchTab === 'undefined') {
    window.cocoSwitchTab = function(btn, panelId, groupId) {
        const root = document.getElementById(groupId);
        if (!root) return;
        // Scope to direct children so nested tab groups are unaffected.
        const panelHost = root.querySelector(':scope > .coco-tab-panels');
        if (panelHost) {
            Array.from(panelHost.children).forEach(p => p.classList.add('hidden'));
        }
        const barHost = root.querySelector(':scope > .coco-tab-bar');
        if (barHost) {
            Array.from(barHost.children).forEach(b => {
                b.classList.remove('text-blue-600', 'border-blue-600',
                                   'dark:text-blue-500', 'dark:border-blue-500',
                                   'active-tab');
                b.classList.add('border-transparent', 'hover:text-gray-600',
                                'hover:border-gray-300', 'dark:hover:text-gray-300');
            });
        }
        const target = document.getElementById(panelId);
        if (target) target.classList.remove('hidden');
        btn.classList.remove('border-transparent', 'hover:text-gray-600',
                             'hover:border-gray-300', 'dark:hover:text-gray-300');
        btn.classList.add('text-blue-600', 'border-blue-600',
                          'dark:text-blue-500', 'dark:border-blue-500',
                          'active-tab');
    };
}
</script>
        """
        parts.append(script)
        return "".join(parts)


class DownloadAssetElement(Element):
    """A button that downloads binary/text data stored in the report payload."""

    def __init__(
        self,
        data: str | bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
        label: str = "Download Asset",
        style: str = "blue",
    ):
        self.data = data
        self.filename = filename
        self.mime_type = mime_type
        self.label = label
        self.style = style
        self.registry_id = None

        # Warn if data is too large (e.g. > 10MB)
        size = len(data) if isinstance(data, (str, bytes)) else 0
        if size > 10 * 1024 * 1024:
            logger.warning(
                "DownloadAssetElement data is very large (%.1f MB). "
                "This will inflate the HTML size considerably.",
                size / 1024 / 1024,
            )

    def collect_payload(self, registry: dict[str, Any]) -> None:
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        # Convert to base64 for safe JSON transport in payload
        if isinstance(self.data, str):
            b64_data = base64.b64encode(self.data.encode("utf-8")).decode("utf-8")
        else:
            b64_data = base64.b64encode(self.data).decode("utf-8")

        registry[self.registry_id] = {
            "type": "asset",
            "filename": self.filename,
            "mime_type": self.mime_type,
            "data_b64": b64_data,
        }

    def render(self) -> str:
        if self.registry_id is None:
            self.registry_id = str(uuid.uuid4())

        script = """
        <script>
        if (typeof window.cocoDownloadAsset === 'undefined') {
            window.cocoDownloadAsset = function(btn, uuid) {
                const el = document.getElementById('report-payload');
                const payloadStr = el.textContent;
                const pako = window.pako;
                const compressed = Uint8Array.from(
                    atob(payloadStr), c => c.charCodeAt(0)
                );
                const jsonBytes = pako.inflate(compressed);
                const jsonStr = new TextDecoder().decode(jsonBytes);
                const registry = JSON.parse(jsonStr);
                const asset = registry[uuid];

                const byteCharacters = atob(asset.data_b64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {type: asset.mime_type});

                const link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = asset.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
        }
        </script>
        """

        label_html = html.escape(self.label)
        return (
            f"<button onclick=\"cocoDownloadAsset(this, '{self.registry_id}')\" "
            f'class="inline-flex items-center px-4 py-2 border border-transparent '
            f"text-sm font-medium rounded-md shadow-sm text-white "
            f"bg-{self.style}-600 hover:bg-{self.style}-700 focus:outline-none "
            f'focus:ring-2 focus:ring-offset-2 focus:ring-{self.style}-500">'
            f'<svg class="-ml-1 mr-2 h-5 w-5" fill="none" stroke="currentColor" '
            f'viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" '
            f'stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 '
            f'003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>'
            f"{label_html}"
            f"</button>"
            f"{script}"
        )


__all__ = [
    "AccordionElement",
    "BadgeElement",
    "CalloutElement",
    "CodeBlockElement",
    "ColumnsElement",
    "ContainerElement",
    "DownloadAssetElement",
    "Element",
    "HtmlElement",
    "ImageElement",
    "InteractiveTableElement",
    "MarkdownElement",
    "MetricsTableElement",
    "PlotlyElement",
    "ProgressBarElement",
    "StatCardElement",
    "TableElement",
    "TabsElement",
    "TimelineElement",
]
