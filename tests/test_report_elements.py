"""Tests for individual Element classes in coco_pipe.report.elements."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from coco_pipe.report.core import Report
from coco_pipe.report.elements import (
    AccordionElement,
    BadgeElement,
    CalloutElement,
    CodeBlockElement,
    ColumnsElement,
    DownloadAssetElement,
    HtmlElement,
    ImageElement,
    InteractiveTableElement,
    MarkdownElement,
    MetricsTableElement,
    PlotlyElement,
    ProgressBarElement,
    StatCardElement,
    TableElement,
    TabsElement,
    TimelineElement,
)


@pytest.fixture
def tmp_report_file(tmp_path):
    return tmp_path / "test_report.html"


def test_html_element_rendering():
    el = HtmlElement("<p>Test</p>")
    assert el.render() == "<p>Test</p>"


def test_image_element_sources(tmp_path):
    # 1. Bytes
    img_bytes = b"fake-image-data"
    el_bytes = ImageElement(img_bytes)
    assert "ZmFrZS1pbWFnZS1kYXRh" in el_bytes.render()

    # 2. Path
    img_file = tmp_path / "test.png"
    img_file.write_bytes(img_bytes)
    el_path = ImageElement(img_file)
    assert "ZmFrZS1pbWFnZS1kYXRh" in el_path.render()

    # 3. Matplotlib (check savefig call)
    fig_mock = MagicMock()
    el_fig = ImageElement(fig_mock)
    el_fig.render()
    assert fig_mock.savefig.called

    # 4. Unsupported source
    with pytest.raises(ValueError, match="Unsupported image source type"):
        ImageElement(123).render()


def test_image_element_renders_data_uri(tmp_path):
    img_data = b"fake-image-data"
    elem = ImageElement(img_data)
    assert "data:image/png;base64" in elem.render()

    p = tmp_path / "test.png"
    p.write_bytes(img_data)
    elem_p = ImageElement(p)
    assert "data:image/png;base64" in elem_p.render()

    with pytest.raises(ValueError, match="Unsupported image source type"):
        ImageElement(123)._encode_image()


def test_plotly_element_binary_decoding():
    """Simulates Plotly's optimisation that base64-encodes large numeric arrays."""
    import base64 as b64

    fig_mock = MagicMock()
    data_bytes = np.array([1.0, 2.0], dtype="float32").tobytes()
    b64_data = b64.b64encode(data_bytes).decode()

    fig_mock.to_json.return_value = f"""
    {{
        "data": [{{
            "x": {{"dtype": "float32", "bdata": "{b64_data}"}},
            "y": [3, 4]
        }}]
    }}
    """
    el = PlotlyElement(fig_mock)
    registry = {}
    el.collect_payload(registry)

    payload = list(registry.values())[0]
    assert np.allclose(payload["data"][0]["x"], [1.0, 2.0])


def test_plotly_element_decodes_bdata_inside_collect():
    class MockFig:
        def to_json(self):
            return '{"data": [{"y": {"dtype": "f8", "bdata": "AAAAAAAAAAA="}}]}'

        def to_dict(self):
            return {"data": []}

    elem_plotly = PlotlyElement(MockFig())
    registry = {}
    elem_plotly.collect_payload(registry)
    assert elem_plotly.registry_id in registry


def test_table_element_dict_inputs():
    # Scalar dict -> 1 row table
    el_scalar = TableElement({"A": 1, "B": 2})
    assert "A" in el_scalar.render()
    assert "1" in el_scalar.render()

    # List of dicts
    el_list = TableElement([{"A": 1}, {"A": 2}])
    assert "A" in el_list.render()
    assert "2" in el_list.render()

    # Non-scalar dict (dict of lists)
    el_non_scalar = TableElement({"A": [1, 2], "B": [3, 4]})
    assert "A" in el_non_scalar.render()
    assert "2" in el_non_scalar.render()


def test_table_element_normalizes_mapping():
    assert isinstance(TableElement._to_frame({"a": 1, "b": 2}), pd.DataFrame)


def test_interactive_table_element_payload_and_render(tmp_report_file):
    df = pd.DataFrame(
        {
            "eval_name": ["epilepsy", "adhd"],
            "reducer": ["PCA", "UMAP"],
            "score": [0.71, 0.82],
        }
    )
    element = InteractiveTableElement(
        df,
        title="Interactive Metrics",
        selector_columns=["eval_name", "reducer"],
        default_sort={"column": "score", "direction": "desc"},
        page_size=25,
    )

    registry = {}
    element.collect_payload(registry)
    assert len(registry) == 1
    payload = next(iter(registry.values()))
    assert payload["columns"] == ["eval_name", "reducer", "score"]
    assert len(payload["rows"]) == 2

    html = element.render()
    assert 'class="interactive-table"' in html
    assert 'data-id="' in html
    assert 'data-config="' in html

    report = Report(title="Interactive Table Report")
    report.add_element(element)
    report.save(str(tmp_report_file))
    content = tmp_report_file.read_text(encoding="utf-8")
    assert "interactive-table" in content
    assert "initInteractiveTables" in content
    assert "data-table-search" in content
    assert "data-sort-column" in content
    assert "data-selector-column" in content
    assert "data-export-table" in content
    assert "data-page-size" in content


def test_metrics_table_highlighting():
    df = pd.DataFrame(
        {"method": ["A", "B"], "acc": [0.8, 0.9], "loss": [0.2, 0.1]}
    )  # B is better in both

    el = MetricsTableElement(
        df, highlight_cols=["acc", "loss"], higher_is_better=["acc"]
    )
    html = el.render()
    assert "font-bold text-green-600" in html

    # Missing highlight col is tolerated
    el_miss = MetricsTableElement(df, highlight_cols=["nonexistent"])
    assert el_miss.render()


def test_metrics_table_best_values_directions():
    df = pd.DataFrame({"m": ["a", "b"], "score": [0.8, 0.9], "error": [0.1, 0.05]})
    elem_metrics = MetricsTableElement(
        df, higher_is_better=["score"], highlight_cols=["score", "error"]
    )
    assert elem_metrics.best_vals["score"] == 0.9

    elem_metrics_low = MetricsTableElement(df, higher_is_better=False)
    assert elem_metrics_low.best_vals["score"] == 0.8


def test_stat_card_element():
    card = StatCardElement("Accuracy", 0.95, unit="%", delta="+2%", color="green")
    html = card.render()
    assert "Accuracy" in html
    assert "0.95" in html
    assert "%" in html
    assert "+2%" in html
    assert "bg-green-500" in html


def test_callout_element():
    callout = CalloutElement("This is a warning", kind="warning", title="Attention")
    html = callout.render()
    assert "This is a warning" in html
    assert "Attention" in html
    assert "bg-yellow-50" in html


def test_code_block_element():
    cb = CodeBlockElement(
        'print("Hello")', language="python", title="Snippet", copyable=True
    )
    html = cb.render()
    assert "print(&quot;Hello&quot;)" in html
    assert "python" in html
    assert "Snippet" in html
    assert "Copied!" in html  # copy-button hover text exists


def test_columns_element():
    col = ColumnsElement([HtmlElement("A"), HtmlElement("B")], cols=2, gap="gap-2")
    html = col.render()
    assert "grid-cols-2" in html
    assert "gap-2" in html
    assert "A" in html
    assert "B" in html


def test_accordion_element():
    acc = AccordionElement("Details", open=True)
    acc.add_element(HtmlElement("Inside"))
    html = acc.render()
    assert "<details" in html
    assert " open" in html
    assert "Details" in html
    assert "Inside" in html


def test_markdown_element():
    elem = MarkdownElement("**Bold** and *italic*")
    html_out = elem.render()
    assert "Bold" in html_out
    assert "italic" in html_out
    assert "prose" in html_out or "whitespace-pre-wrap" in html_out


def test_markdown_fallback_when_package_missing(monkeypatch):
    """When the markdown package is missing, MarkdownElement falls back to <pre>."""
    monkeypatch.setitem(sys.modules, "markdown", None)
    rep = Report()
    rep.add_markdown("# Fallback")
    assert "whitespace-pre-wrap" in rep.render()


def test_badge_element():
    elem = BadgeElement("Experimental", color="red")
    html_out = elem.render()
    assert "Experimental" in html_out
    assert "bg-red-50" in html_out


def test_progress_bar_element():
    elem = ProgressBarElement(75.0, 100.0, "Accuracy", "green")
    html_out = elem.render()
    assert "Accuracy" in html_out
    assert "width: 75.0%" in html_out
    assert "bg-green-600" in html_out


def test_timeline_element():
    events = [
        {"title": "Start", "time": "10:00", "description": "Initiated"},
        {"title": "End", "time": "11:00", "description": "Finished", "status": "green"},
    ]
    elem = TimelineElement(events)
    html_out = elem.render()
    assert "Start" in html_out
    assert "10:00" in html_out
    assert "Initiated" in html_out
    assert "End" in html_out
    assert "bg-green-500" in html_out


def test_tabs_element():
    tabs = {
        "Tab1": HtmlElement("<b>Tab 1 Content</b>"),
        "Tab2": HtmlElement("<i>Tab 2 Content</i>"),
    }
    elem = TabsElement(tabs)
    html_out = elem.render()
    assert "Tab1" in html_out
    assert "Tab 1 Content" in html_out
    assert "Tab 2 Content" in html_out
    assert "cocoSwitchTab" in html_out
    assert "hidden" in html_out


def test_download_asset_element_bytes():
    elem = DownloadAssetElement(b"Hello World", "test.txt", "text/plain")
    registry = {}
    elem.collect_payload(registry)

    assert elem.registry_id in registry
    assert registry[elem.registry_id]["filename"] == "test.txt"
    # Base64 for 'Hello World'
    assert "SGVsbG8gV29ybGQ=" in registry[elem.registry_id]["data_b64"]

    html_out = elem.render()
    assert "Download Asset" in html_out
    assert "cocoDownloadAsset" in html_out
    assert elem.registry_id in html_out


def test_download_asset_element_string():
    elem = DownloadAssetElement("String data", "data.csv")
    registry = {}
    elem.collect_payload(registry)
    assert registry[elem.registry_id]["filename"] == "data.csv"


def _decode_report_payload(html: str) -> dict:
    """Decompress the gzip+base64 payload script and return the parsed dict.

    Mirrors what the browser does in initPayload(). Raising ``json.JSONDecodeError``
    here would mean the browser-side ``JSON.parse`` would also reject the payload,
    blanking the entire report.
    """
    import base64
    import gzip
    import json
    import re

    match = re.search(
        r'<script type="application/json" id="report-payload">([^<]+)</script>',
        html,
    )
    assert match, "report-payload script tag not found in rendered HTML"
    raw = base64.b64decode(match.group(1).strip())
    return json.loads(gzip.decompress(raw))


def test_payload_replaces_nan_with_null_in_table():
    """NaN/Inf in a TableElement must serialize as JSON null, not literal NaN."""
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "value": [1.0, float("nan"), float("inf")],
            "neg_inf": [float("-inf"), 2.0, 3.0],
        }
    )
    report = Report(title="NaN payload regression")
    section = report  # report subclasses ContainerElement

    section.add_element(InteractiveTableElement(df, title="With NaN"))
    html = report.render()
    payload = _decode_report_payload(html)  # raises if invalid JSON

    table = next(p for p in payload.values() if isinstance(p, dict) and "rows" in p)
    rows = table["rows"]
    assert rows[0]["value"] == 1.0
    assert rows[1]["value"] is None  # was NaN
    assert rows[2]["value"] is None  # was +Inf
    assert rows[0]["neg_inf"] is None  # was -Inf


def test_payload_replaces_nan_inside_plotly_trace():
    """NaN inside a Plotly trace y-array must also become JSON null."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1.0, float("nan"), 3.0]))
    report = Report(title="Plotly NaN regression")
    report.add_element(PlotlyElement(fig))

    payload = _decode_report_payload(report.render())
    fig_payload = next(
        p
        for p in payload.values()
        if isinstance(p, dict) and "data" in p and "layout" in p
    )
    y = fig_payload["data"][0]["y"]
    assert y[0] == 1.0
    assert y[1] is None
    assert y[2] == 3.0
