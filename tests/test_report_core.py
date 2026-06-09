import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from coco_pipe.report.core import (
    ContainerElement,
    HtmlElement,
    ImageElement,
    InteractiveTableElement,
    MetricsTableElement,
    PlotlyElement,
    Report,
    Section,
    TableElement,
    _get_reducer_summary,
    _metrics_summary_table,
    _trajectory_times,
)
from coco_pipe.report.quality import CheckResult


@pytest.fixture
def tmp_report_file(tmp_path):
    return tmp_path / "test_report.html"


def test_element_rendering():
    el = HtmlElement("<p>Test</p>")
    assert el.render() == "<p>Test</p>"


def test_section_rendering():
    sec = Section(title="My Section", icon="📊")
    sec.add_element("<p>Content</p>")
    html = sec.render()
    assert "My Section" in html
    assert "📊" in html
    assert "<p>Content</p>" in html
    # Check for Tailwind class (was bg-white in previous versions,
    # now might be different but let's check title)
    assert "My Section" in html


def test_get_reducer_summary_edge_cases():
    # 1. Missing get_summary
    with pytest.raises(TypeError, match="must implement get_summary"):
        _get_reducer_summary(object())

    # 2. get_summary returns non-dict
    mock = MagicMock()
    mock.get_summary.return_value = "not a dict"
    with pytest.raises(TypeError, match="must return a dictionary"):
        _get_reducer_summary(mock)

    # 3. Partial summary (fills defaults)
    mock.get_summary.return_value = {"method": "PCA"}
    summary = _get_reducer_summary(mock)
    assert summary["method"] == "PCA"
    assert summary["metrics"] == {}
    assert summary["metric_records"] == []


def test_image_element_sources(tmp_path):
    # 1. Bytes
    img_bytes = b"fake-image-data"
    el_bytes = ImageElement(img_bytes)
    # base64.b64encode(b"fake-image-data") -> ZmFrZS1pbWFnZS1kYXRh
    assert "ZmFrZS1pbWFnZS1kYXRh" in el_bytes.render()

    # 2. Path
    img_file = tmp_path / "test.png"
    img_file.write_bytes(img_bytes)
    el_path = ImageElement(img_file)
    assert "ZmFrZS1pbWFnZS1kYXRh" in el_path.render()

    # 3. Matplotlib (tested in integration, but check savefig call)
    fig_mock = MagicMock()
    el_fig = ImageElement(fig_mock)
    el_fig.render()
    assert fig_mock.savefig.called

    # 4. Unsupported source
    with pytest.raises(ValueError, match="Unsupported image source type"):
        ImageElement(123).render()


def test_plotly_element_binary_decoding():
    # Mock a plotly figure that would have binary encoded arrays in its JSON
    # This simulates Plotly's performance optimization for large arrays
    import base64 as b64

    fig_mock = MagicMock()
    # Create valid float32 binary data for [1.0, 2.0]
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


def test_table_element_dict_inputs():
    # 1. Scalar dict -> 1 row table
    el_scalar = TableElement({"A": 1, "B": 2})
    assert "A" in el_scalar.render()
    assert "1" in el_scalar.render()

    # 2. List of dicts
    el_list = TableElement([{"A": 1}, {"A": 2}])
    assert "A" in el_list.render()
    assert "1" in el_list.render()
    assert "2" in el_list.render()

    # 3. Non-scalar dict (e.g. dict of lists)
    el_non_scalar = TableElement({"A": [1, 2], "B": [3, 4]})
    assert "A" in el_non_scalar.render()
    assert "2" in el_non_scalar.render()


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

    # Highlight higher acc, lower loss
    el = MetricsTableElement(
        df, highlight_cols=["acc", "loss"], higher_is_better=["acc"]
    )
    html = el.render()
    assert "font-bold text-green-600" in html

    # Missing highlight col
    el_miss = MetricsTableElement(df, highlight_cols=["nonexistent"])
    assert el_miss.render()


def test_report_config_and_metadata():
    # 1. Title override via dict
    from coco_pipe.report.config import ReportConfig

    rep = Report(title="Old", config={"title": "New", "x": 1})
    assert rep.title == "New"
    # Note: Pydantic V2 extra fields are in model_extra
    if rep.config.model_extra is not None:
        assert rep.config.model_extra["x"] == 1

    # 2. Pass ReportConfig object
    cfg = ReportConfig(title="Object")
    rep_obj = Report(config=cfg)
    assert rep_obj.title == "Object"

    # 3. Fallback for broken config dict
    rep_fallback = Report("Fallback", config={"unexpected": object()})
    assert rep_fallback.title == "Fallback"


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

    # 2. Case with NaNs (triggers missingness finding)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    c_nan = DataContainer(X_nan, dims=("obs", "feature"))
    rep.add_container(c_nan)
    assert any("Missingness" in str(f) for f in rep.children[-1].findings)

    # 3. Large data (sampling logic)
    X_large = np.random.randn(6000, 1)
    c_large = DataContainer(X_large, dims=("obs", "feature"))
    rep.add_container(c_large)
    assert rep.children[-1].render()

    # 4. Exception path (trigger warning)
    with pytest.warns(UserWarning, match="Failed to add container"):
        rep.add_container(None)


def test_markdown_fallback(monkeypatch):
    # Simulate missing markdown package
    monkeypatch.setitem(sys.modules, "markdown", None)
    rep = Report()
    rep.add_markdown("# Fallback")
    assert "whitespace-pre-wrap" in rep.render()


def test_internal_helpers_coverage():
    # 1. _metrics_summary_table
    assert _metrics_summary_table({}).empty

    # 2. _trajectory_times
    assert _trajectory_times({}, np.array([1, 2, 3])) is not None
    assert _trajectory_times({}, np.array([])) is None
    assert _trajectory_times({"trajectory_times_": [1, 2]}, None) is not None
    assert _trajectory_times({"trajectory_times_": []}, None) is None


def test_report_creation_and_save(tmp_report_file):
    rep = Report(title="Unit Test Report")

    # Add simple HTML
    rep.add_element(HtmlElement("<p>Hello World</p>"))

    # Add a section
    sec = Section("Analysis")
    sec.add_element("<b>Bold Content</b>")
    rep.add_section(sec)

    # Add markdown
    rep.add_markdown("# Markdown Header\n* Item 1")
    assert "Markdown Header" in rep.render()

    # Save
    rep.save(str(tmp_report_file))

    assert tmp_report_file.exists()
    content = tmp_report_file.read_text(encoding="utf-8")

    # Verify Content
    assert "<!DOCTYPE html>" in content
    assert "Unit Test Report" in content
    assert "Hello World" in content
    assert "Analysis" in content
    assert "Bold Content" in content
    assert "Markdown Header" in content


def test_report_add_reduction_coverage():
    from coco_pipe.io.structures import DataContainer

    rep = Report()

    # Mock reducer with diagnostics
    mock_reducer = MagicMock()
    mock_reducer.get_summary.return_value = {
        "method": "MockDR",
        "metrics": {"trust": 0.9},
        "metric_records": [{"method": "MockDR", "metric": "trust", "value": 0.9}],
        "diagnostics": {
            "embedding_": np.random.randn(10, 2),
            "reconstruction_": np.random.randn(10, 5),
        },
        "quality_metadata": {},
    }

    X = np.random.randn(10, 5)
    _ = DataContainer(X, dims=("obs", "feature"))

    # 1. Basic add
    rep.add_reduction(mock_reducer, name="Mock Reduction")
    assert "Mock Reduction" in rep.children[-1].render()

    # 2. Add with explicit embedding and labels
    X_emb = np.random.randn(10, 2)
    labels = np.array([0, 1] * 5)
    metadata = {"feat": np.random.randn(10)}
    rep.add_reduction(
        mock_reducer,
        name="With Embedding",
        X_emb=X_emb,
        labels=labels,
        metadata=metadata,
    )
    assert "With Embedding" in rep.children[-1].render()


def test_fluent_interface_structure():
    rep = Report("Fluency")
    rep.add_element("Start").add_section(Section("Middle")).add_markdown("End")
    assert len(rep.children) == 3


def test_report_elements_hardening(tmp_path):
    # ImageElement
    img_data = b"fake-image-data"
    elem = ImageElement(img_data)
    assert "data:image/png;base64" in elem.render()

    p = tmp_path / "test.png"
    p.write_bytes(img_data)
    elem_p = ImageElement(p)
    assert "data:image/png;base64" in elem_p.render()

    with pytest.raises(ValueError, match="Unsupported image source type"):
        ImageElement(123)._encode_image()

    # PlotlyElement binary decoding
    class MockFig:
        def to_json(self):
            return '{"data": [{"y": {"dtype": "f8", "bdata": "AAAAAAAAAAA="}}]}'

        def to_dict(self):
            return {"data": []}

    elem_plotly = PlotlyElement(MockFig())
    registry = {}
    elem_plotly.collect_payload(registry)
    assert elem_plotly.registry_id in registry

    # TableElement normalization
    assert isinstance(TableElement._to_frame({"a": 1, "b": 2}), pd.DataFrame)

    # MetricsTableElement directions
    df = pd.DataFrame({"m": ["a", "b"], "score": [0.8, 0.9], "error": [0.1, 0.05]})
    elem_metrics = MetricsTableElement(
        df, higher_is_better=["score"], highlight_cols=["score", "error"]
    )
    assert elem_metrics.best_vals["score"] == 0.9

    elem_metrics_low = MetricsTableElement(df, higher_is_better=False)
    assert elem_metrics_low.best_vals["score"] == 0.8

    # ContainerElement markdown fallback
    cont = ContainerElement()
    cont.add_markdown("# Title")
    assert "Title" in cont.render()

    # Section status upgrades
    sec = Section("Test")
    sec.add_finding(CheckResult("c1", "WARN", "w", 4))
    assert sec.status == "WARN"
    sec.add_finding(CheckResult("c2", "FAIL", "f", 9))
    assert sec.status == "FAIL"
    sec.add_finding(CheckResult("c3", "WARN", "w2", 4))
    assert sec.status == "FAIL"

    # Report config coercion
    rep = Report(title="T", config={"some_param": 1})
    assert rep.title == "T"
    # In Pydantic 2, extra fields are allowed but might be on the object
    # if extra='allow'
    assert getattr(rep.config, "some_param") == 1
