import pandas as pd
import pytest

from coco_pipe.viz.interactive._utils import _discrete_colorscale, _marker_payload


def test_marker_payload_maps_categorical_dtype_values():
    df = pd.DataFrame({"Group": pd.Categorical([1, 2, 1], categories=[1, 2])})

    payload = _marker_payload(
        df,
        "Group",
        cmap="Viridis",
        palette=["#111111", "#222222"],
        color_kind="categorical",
        restyle=False,
    )

    assert payload["color"] == [0, 1, 0]
    assert payload["colorbar"]["ticktext"] == ["1", "2"]


def test_marker_payload_restyle_wraps_trace_arrays_and_none_bounds():
    df = pd.DataFrame({"Score": [0.1, 0.2, 0.3]})

    payload = _marker_payload(
        df,
        "Score",
        cmap="Viridis",
        palette=None,
        color_kind="continuous",
        restyle=True,
    )

    assert len(payload["marker.color"]) == 1
    assert payload["marker.cmin"] == [None]
    assert payload["marker.cmax"] == [None]


def test_discrete_colorscale_rejects_empty_palette():
    with pytest.raises(ValueError, match="palette"):
        _discrete_colorscale(["A"], palette=[])


def test_apply_layout_kwargs():
    import plotly.graph_objects as go

    from coco_pipe.viz.interactive._utils import _apply_layout

    fig = go.Figure()
    _apply_layout(
        fig,
        title="T",
        xaxis_title="X",
        yaxis_title="Y",
        height=400,
        barmode="group",
        legend_horizontal=True,
    )
    layout = fig.layout
    assert layout.title.text == "T"
    assert layout.xaxis.title.text == "X"
    assert layout.yaxis.title.text == "Y"
    assert layout.height == 400
    assert layout.barmode == "group"


def test_discrete_colorscale_palettes():
    # None palette
    _, scale = _discrete_colorscale(["A"])
    assert scale is not None

    # string palette
    _, scale = _discrete_colorscale(["A", "B"], palette="Pastel")
    assert scale is not None


def test_marker_payload_exceptions_and_non_cat():
    # invalid kind
    with pytest.raises(ValueError):
        _marker_payload(
            pd.DataFrame({"A": [1]}), "A", "Viridis", None, "invalid", False
        )

    # categorical without .cat accessor
    df = pd.DataFrame({"Group": ["a", "b", "a"]})
    payload = _marker_payload(df, "Group", "Viridis", None, "categorical", True)
    assert "marker.color" in payload

    # continuous restyle
    payload = _marker_payload(df, "Group", "Viridis", None, "categorical", False)
    assert "color" in payload

    # continuous restyle false
    payload2 = _marker_payload(
        pd.DataFrame({"A": [1.0]}), "A", "Viridis", None, "continuous", False
    )
    assert "color" in payload2
