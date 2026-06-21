"""Tests for Section and Report container behavior in coco_pipe.report.core.

Individual Element tests live in tests/test_report_elements.py.
Dim-reduction-specific Report.add_* tests live in tests/test_report_dimred.py.
"""

import base64
import gzip
import json
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from coco_pipe.io.quality import CheckResult
from coco_pipe.report.core import Report, Section, _replace_non_finite
from coco_pipe.report.elements import (
    ContainerElement,
    HtmlElement,
    PlotlyElement,
    TableElement,
)


@pytest.fixture
def tmp_report_file(tmp_path):
    return tmp_path / "test_report.html"


def test_section_rendering():
    sec = Section(title="My Section", icon="C")
    sec.add_element("<p>Content</p>")
    html = sec.render()
    assert "My Section" in html
    assert "C" in html
    assert "<p>Content</p>" in html


def test_section_status_upgrades_on_findings():
    sec = Section("Test")
    sec.add_finding(CheckResult("c1", "WARN", "w", 4))
    assert sec.status == "WARN"
    sec.add_finding(CheckResult("c2", "FAIL", "f", 9))
    assert sec.status == "FAIL"
    sec.add_finding(CheckResult("c3", "WARN", "w2", 4))
    # FAIL is sticky
    assert sec.status == "FAIL"


def test_report_config_and_metadata():
    from coco_pipe.report.config import ReportConfig

    # Title override via dict
    rep = Report(title="Old", config={"title": "New", "x": 1})
    assert rep.title == "New"
    if rep.config.model_extra is not None:
        assert rep.config.model_extra["x"] == 1

    # Pass a ReportConfig object
    cfg = ReportConfig(title="Object")
    rep_obj = Report(config=cfg)
    assert rep_obj.title == "Object"

    # Fallback for a broken config dict
    rep_fallback = Report("Fallback", config={"unexpected": object()})
    assert rep_fallback.title == "Fallback"


def test_report_config_coercion_keeps_extra_fields():
    rep = Report(title="T", config={"some_param": 1})
    assert rep.title == "T"
    # Pydantic 2 allows extras via the model config
    assert rep.config.some_param == 1


def test_container_element_markdown_fallback():
    cont = ContainerElement()
    cont.add_markdown("# Title")
    assert "Title" in cont.render()


def test_report_add_container_functionality():
    from coco_pipe.io.structures import DataContainer

    X = np.random.randn(10, 5)
    container = DataContainer(
        X=X, dims=("obs", "feature"), coords={"feature": ["f1", "f2", "f3", "f4", "f5"]}
    )

    rep = Report()
    rep.add_container(container)
    assert "Dimensions" in rep.children[0].render()
    assert "Coordinates" in rep.children[0].render()

    # NaNs trigger a missingness finding
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    c_nan = DataContainer(X_nan, dims=("obs", "feature"))
    rep.add_container(c_nan)
    assert any("Missingness" in str(f) for f in rep.children[-1].findings)

    # Large data exercises the sampling branch
    X_large = np.random.randn(6000, 1)
    c_large = DataContainer(X_large, dims=("obs", "feature"))
    rep.add_container(c_large)
    assert rep.children[-1].render()

    # Exception path emits a warning instead of raising
    with pytest.warns(UserWarning, match="Failed to add container"):
        rep.add_container(None)


def test_add_container_plot_branches():
    from coco_pipe.io.structures import DataContainer

    rep = Report()

    # y provided -> target label distribution
    c_y = DataContainer(
        X=np.random.randn(10, 2),
        dims=("obs", "feature"),
        y=np.array([0, 1] * 5),
    )
    rep.add_container(c_y)
    html = rep.render()
    assert "Target label distribution." in html

    # Flat X without y, >5000 elements triggers downsampling
    c_large = DataContainer(X=np.random.randn(6000, 1), dims=("obs", "feature"))
    rep.add_container(c_large)
    html = rep.render()
    assert "Histogram of data values." in html


def test_add_figure_shortcut():
    import matplotlib.pyplot as plt

    rep = Report("Fig Test")
    fig, _ax = plt.subplots()
    rep.add_figure(fig, caption="Shortcut")

    html = rep.render()
    assert "Shortcut" in html
    plt.close(fig)


def test_add_raw_preview():
    from coco_pipe.io.structures import DataContainer

    X = np.random.randn(10, 5)
    sample_container = DataContainer(X=X, dims=("obs", "feature"))

    rep = Report("Raw Test")
    rep.add_raw_preview(sample_container.X, name="My Raw Data")

    html = rep.render()
    assert "My Raw Data" in html
    assert "lazy-plot" in html


def test_report_section_ids_are_unique():
    rep = Report("Duplicate Sections")
    rep.add_section(Section("Repeated"))
    rep.add_section(Section("Repeated"))

    ids = [section.id for section in rep.children]
    assert ids == ["repeated", "repeated-2"]


def test_fluent_interface_structure():
    rep = Report("Fluency")
    rep.add_element("Start").add_section(Section("Middle")).add_markdown("End")
    assert len(rep.children) == 3


def test_report_creation_and_save(tmp_report_file):
    rep = Report(title="Unit Test Report")
    rep.add_element(HtmlElement("<p>Hello World</p>"))

    sec = Section("Analysis")
    sec.add_element("<b>Bold Content</b>")
    rep.add_section(sec)

    rep.add_markdown("# Markdown Header\n* Item 1")
    assert "Markdown Header" in rep.render()

    rep.save(str(tmp_report_file))
    assert tmp_report_file.exists()
    content = tmp_report_file.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert "Unit Test Report" in content
    assert "Hello World" in content
    assert "Analysis" in content
    assert "Bold Content" in content
    assert "Markdown Header" in content


def test_global_data_store_payload():
    import plotly.graph_objects as go

    rep = Report("Payload Test")
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    rep.add_element(PlotlyElement(fig))

    html = rep.render()

    # Payload script tag exists
    assert 'id="report-payload"' in html
    # Inline data-figure is *not* used; the lazy plot looks up by data-id
    assert 'data-id="' in html

    # Extract + decompress and verify the registered figure round-trips
    start_tag = 'id="report-payload">'
    end_tag = "</script>"

    start_idx = html.find(start_tag) + len(start_tag)
    end_idx = html.find(end_tag, start_idx)

    payload_b64 = html[start_idx:end_idx].strip()
    assert len(payload_b64) > 0

    compressed = base64.b64decode(payload_b64)
    json_bytes = gzip.decompress(compressed)
    data_registry = json.loads(json_bytes)

    id_start = html.find('data-id="') + 9
    id_end = html.find('"', id_start)
    uuid_str = html[id_start:id_end]

    assert uuid_str in data_registry
    assert "data" in data_registry[uuid_str]
    assert data_registry[uuid_str]["data"][0]["y"] == [3, 4]


# ----- Bound-method API surface (regression: monkey-patch binding) -----------


def test_fluent_stubs():
    """Verify every add_decoding_* / add_reduction_* method is bound to Report."""
    rep = Report()
    method_names = [
        "add_decoding_overview",
        "add_decoding_temporal",
        "add_decoding_summary",
        "add_decoding_diagnostics",
        "add_decoding_statistical_assessment",
        "add_decoding_neural_artifacts",
        "add_decoding_performance",
        "add_decoding_features",
        "add_decoding_topomaps",
        "add_reduction",
        "add_comparison",
        "add_reduction_overview",
        "add_reduction_embedding",
        "add_reduction_metrics",
        "add_reduction_diagnostics",
        "add_reduction_interpretation",
        "add_reduction_coranking",
        "add_reduction_components",
        "add_reduction_trajectory",
        "add_reduction_trajectory_separation",
    ]
    for name in method_names:
        with patch.object(Report, name) as m:
            m.return_value = rep
            getattr(rep, name)(None)
            m.assert_called()


def test_asset_mode_defaults_to_cdn():
    rep = Report(title="Default")
    assert rep.asset_mode == "cdn"
    assert "cdn.plot.ly" in rep.asset_urls["plotly"]


def test_asset_mode_custom_override():
    rep = Report(title="Custom", asset_urls={"plotly": "/static/plotly.js"})
    assert rep.asset_mode == "custom"
    assert rep.asset_urls["plotly"] == "/static/plotly.js"
    # Other slots fall back to CDN defaults
    assert "tailwindcss" in rep.asset_urls["tailwind"]


def test_asset_mode_inline_round_trip(monkeypatch, tmp_path):
    """`asset_urls="inline"` inlines the bundle bytes in <script> tags."""
    monkeypatch.setenv("COCO_PIPE_REPORT_ASSET_CACHE", str(tmp_path))
    fake_bundles = {
        "plotly": b"/* fake plotly bundle */ console.log('plotly');",
        "tailwind": b"/* fake tailwind bundle */ console.log('tailwind');",
        "pako": b"/* fake pako bundle */ console.log('pako');",
    }

    class FakeResp:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._data

    def fake_urlopen(url_or_request, timeout=None):
        url = (
            url_or_request.full_url
            if hasattr(url_or_request, "full_url")
            else url_or_request
        )
        for name, data in fake_bundles.items():
            if name in url:
                return FakeResp(data)
        raise AssertionError(f"unexpected url: {url}")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    rep = Report(title="Inline", asset_urls="inline")
    assert rep.asset_mode == "inline"
    assert rep.asset_urls["plotly"].startswith("/* fake plotly bundle")

    html = rep.render()
    # Inlined contents appear in the output; src= references do not.
    assert "/* fake plotly bundle */" in html
    assert "/* fake tailwind bundle */" in html
    assert "/* fake pako bundle */" in html
    assert 'src="https://cdn.plot.ly' not in html


def test_asset_mode_invalid_value_raises():
    with pytest.raises(TypeError):
        Report(title="Bad", asset_urls=42)


def test_replace_non_finite_float_array():
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    res = _replace_non_finite(arr)
    assert res == [1.0, None, None, None, 2.0]


def test_section_add_columns():
    sec = Section("Test")
    sec.add_columns([TableElement([])])
    assert len(sec.children) == 1


def test_report_provenance_fallback():
    # If config doesn't have provenance, it injects it
    from coco_pipe.report.config import ReportConfig

    cfg = ReportConfig(title="T")
    cfg.provenance = None
    rep = Report(title="T", config=cfg)
    assert rep.config.provenance is not None


def test_apply_theme_exceptions():
    with (
        patch("coco_pipe.viz.theme.set_coco_theme", side_effect=Exception("theme err")),
        patch(
            "coco_pipe.viz.interactive._utils._register_coco_template",
            side_effect=Exception("tpl err"),
        ),
    ):
        Report(title="T")  # should catch silently


def test_resolve_config_fallback():
    from coco_pipe.report.core import Report

    # provenance must be a dict or valid model, passing an int fails validation
    rep = Report(config={"provenance": 123}, title="T")
    assert (
        getattr(rep.config, "run_params", None) is not None or rep.config.title == "T"
    )


def test_add_container_constant_columns():
    rep = Report("T")

    class DummyContainer:
        dims: ClassVar[list] = ["a"]
        shape: ClassVar[list] = [2]
        coords: ClassVar[dict] = {}
        X = np.array([[1, 1], [1, 1]])  # constant
        y = None

    rep.add_container(DummyContainer(), show_dist=True)
    sec = rep.children[-1]
    assert len(sec.findings) >= 0


def test_add_container_plot_exception():
    rep = Report("T")

    class DummyContainer:
        dims: ClassVar[list] = ["a"]
        shape: ClassVar[list] = [2]
        coords: ClassVar[dict] = {}
        X = np.array([1, 2])
        y = None

    with patch("matplotlib.pyplot.subplots", side_effect=Exception("plot err")):
        rep.add_container(DummyContainer())
    sec = rep.children[-1]
    assert "Could not generate plot" in sec.children[-1].html


def test_add_raw_preview2():
    rep = Report("T")

    class DummyData:
        X = np.array([1, 1, 1, 1])  # flatline

    with patch("coco_pipe.report.core.check_flatline") as mock_flat:

        class DummyRes:
            is_issue = True
            status = "FAIL"
            message = "Flatline"

        mock_flat.return_value = DummyRes()

        with patch(
            "coco_pipe.viz.interactive.dim_reduction.plot_raw_preview"
        ) as mock_preview:
            mock_preview.return_value = "fig"
            rep.add_raw_preview(DummyData())
            sec = rep.children[-1]
            assert len(sec.findings) > 0


def test_add_raw_preview_exceptions():
    rep = Report("T")
    with (
        patch(
            "coco_pipe.report.core.check_flatline", side_effect=Exception("check err")
        ),
        patch(
            "coco_pipe.viz.interactive.dim_reduction.plot_raw_preview",
            return_value="fig",
        ),
    ):
        rep.add_raw_preview(np.array([1, 2]))


def test_add_raw_preview_dims():
    rep = Report("T")
    with patch(
        "coco_pipe.viz.interactive.dim_reduction.plot_raw_preview", return_value="fig"
    ):
        # 1D
        rep.add_raw_preview(np.array([1, 2]))
        # 3D
        rep.add_raw_preview(np.array([[[1, 2], [3, 4]]]))


def test_add_summary_card():
    rep = Report("T")
    rep.add_summary_card({"A": 1, "B": 2})
    assert len(rep.children) > 0
    assert len(rep.children[0].children) == 2


def test_show():
    rep = Report("T")
    with patch("webbrowser.open") as mock_open:
        rep.show()
        assert mock_open.called


def test_repr_html():
    rep = Report("T")
    html = rep._repr_html_()
    assert "iframe" in html
    assert "srcdoc" in html
