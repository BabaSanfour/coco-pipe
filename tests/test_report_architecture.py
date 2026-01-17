import base64
import gzip
import json

import plotly.graph_objects as go
import pytest

from coco_pipe.report.core import PlotlyElement, Report


def test_global_data_store_payload():
    """Verify Phase D architecture: Global Compressed Data Store."""

    # 1. Create Report with a Plot
    rep = Report("Payload Test")
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    rep.add_element(PlotlyElement(fig))

    html = rep.render()

    # 2. Check for Payload Script
    assert 'id="report-payload"' in html

    # 3. Check for data-id (and NOT inline data-figure)
    assert 'data-id="' in html

    # 4. Extract Payload and Verify Content
    # Naive extraction logic for test
    start_tag = 'id="report-payload">'
    end_tag = "</script>"

    start_idx = html.find(start_tag) + len(start_tag)
    end_idx = html.find(end_tag, start_idx)

    payload_b64 = html[start_idx:end_idx].strip()
    assert len(payload_b64) > 0

    # Decompress
    try:
        compressed = base64.b64decode(payload_b64)
        json_bytes = gzip.decompress(compressed)
        data_registry = json.loads(json_bytes)

        id_start = html.find('data-id="') + 9
        id_end = html.find('"', id_start)
        uuid_str = html[id_start:id_end]

        assert uuid_str in data_registry
        assert "data" in data_registry[uuid_str]
        assert data_registry[uuid_str]["data"][0]["y"] == [3, 4]

    except Exception as e:
        pytest.fail(f"Payload decompression failed: {e}")
