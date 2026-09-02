import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from coco_pipe.viz.interactive.base import (
    plot_bar,
    plot_distribution_groups,
    plot_grouped_bar,
    plot_heatmap,
    plot_ranked_bar,
    plot_scatter,
    plot_timecourses,
)


def test_plot_bar():
    scores = {"A": 10, "B": 20, "C": 15}
    fig = plot_bar(scores, title="Bar Plot", orientation="vertical")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

    fig_h = plot_bar(
        scores,
        errors={"A": 1, "B": 2, "C": 1.5},
        title="Bar Plot H",
        orientation="horizontal",
        cmap="Viridis",
        abs_values=True,
        top_n=2,
    )
    assert isinstance(fig_h, go.Figure)


def test_plot_scatter():
    df = pd.DataFrame(
        {
            "n": [1, 2, 3, 1, 2, 3],
            "score": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
            "grp": ["a", "a", "a", "b", "b", "b"],
            "label": ["p1", "p2", "p3", "q1", "q2", "q3"],
        }
    )
    # Single trace, no grouping
    fig = plot_scatter(df, x="n", y="score")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].showlegend is False

    # Grouped line-scatter with text + hovertemplate + legend title
    fig_g = plot_scatter(
        df,
        x="n",
        y="score",
        color="grp",
        text="label",
        hovertemplate="%{text}<extra></extra>",
        mode="lines+markers",
        color_map={"a": "red"},
        legend_title="Group",
        title="t",
    )
    assert isinstance(fig_g, go.Figure)
    assert len(fig_g.data) == 2
    # color_map override honored for group 'a'
    assert fig_g.data[0].marker.color == "red"


def test_plot_grouped_bar():
    df = pd.DataFrame(
        {
            "condition": ["c1", "c2", "c1", "c2"],
            "score": [0.5, 0.7, 0.6, 0.8],
            "reducer": ["pca", "pca", "umap", "umap"],
            "ntext": ["n=2", "n=3", "n=4", "n=5"],
        }
    )
    fig = plot_grouped_bar(
        df,
        x="condition",
        y="score",
        group="reducer",
        text="ntext",
        x_order=["c2", "c1"],
        legend_title="Reducer",
        title="t",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "group"


def test_plot_grouped_bar_horizontal():
    df = pd.DataFrame(
        {
            "unit": ["u1", "u2", "u3"],
            "score": [0.5, 0.7, 0.6],
            "grp": ["a", "a", "b"],
            "ntext": ["n=2", "n=3", "n=4"],
        }
    )
    fig = plot_grouped_bar(
        df,
        x="unit",
        y="score",
        group="grp",
        text="ntext",
        orientation="horizontal",
        x_order=["u3", "u2", "u1"],
        xaxis_title="score",
        yaxis_title="unit",
    )
    assert isinstance(fig, go.Figure)
    # Values land on the x-axis, categories on the y-axis.
    assert all(trace.orientation == "h" for trace in fig.data)
    assert list(fig.data[0].x) == [0.5, 0.7]
    assert list(fig.data[0].y) == ["u1", "u2"]
    # x_order drives the (category) y-axis ordering.
    assert fig.layout.yaxis.categoryorder == "array"
    assert list(fig.layout.yaxis.categoryarray) == ["u3", "u2", "u1"]
    assert fig.layout.xaxis.title.text == "score"
    assert fig.layout.yaxis.title.text == "unit"


def test_plot_ranked_bar():
    df = pd.DataFrame(
        {
            "lab": ["a", "b", "c", "d"],
            "v": [3.0, 1.0, 4.0, 2.0],
            "grp": ["x", "x", "y", "y"],
            "t": ["t1", "t2", "t3", "t4"],
        }
    )
    fig = plot_ranked_bar(
        df,
        value="v",
        category="lab",
        color="grp",
        text="t",
        top_n=3,
        value_title="V",
        category_title="L",
        legend_title="Group",
    )
    assert isinstance(fig, go.Figure)
    # Only the top-3 rows survive (c=4, a=3, d=2); b=1 is dropped.
    assert sum(len(trace.x) for trace in fig.data) == 3
    assert all(trace.orientation == "h" for trace in fig.data)
    # Highest-ranked category ends up last so it renders at the top.
    assert list(fig.layout.yaxis.categoryarray) == ["d", "a", "c"]
    assert fig.layout.xaxis.title.text == "V"
    assert fig.layout.yaxis.title.text == "L"
    assert fig.layout.legend.title.text == "Group"


def test_plot_ranked_bar_vertical_keeps_rank_order():
    df = pd.DataFrame({"lab": ["a", "b", "c"], "v": [1.0, 3.0, 2.0], "grp": ["x"] * 3})
    fig = plot_ranked_bar(
        df, value="v", category="lab", color="grp", orientation="vertical"
    )
    # Vertical keeps left-to-right rank order (b=3, c=2, a=1).
    assert list(fig.layout.xaxis.categoryarray) == ["b", "c", "a"]


def test_plot_grouped_bar_baseline_and_value_range():
    df = pd.DataFrame(
        {
            "unit": ["u1", "u2"],
            "score": [0.55, 0.62],
            "grp": ["a", "b"],
        }
    )
    fig = plot_grouped_bar(
        df,
        x="unit",
        y="score",
        group="grp",
        orientation="horizontal",
        baseline=0.5,
        baseline_label="chance",
        value_range=(0.5, 0.7),
    )
    # Horizontal -> value axis is x: range applied there, reference line vertical.
    assert list(fig.layout.xaxis.range) == [0.5, 0.7]
    assert any(
        shape.type == "line" and shape.x0 == 0.5 and shape.x1 == 0.5
        for shape in fig.layout.shapes
    )


def test_plot_distribution_groups_baseline():
    fig = plot_distribution_groups(
        [[0.55, 0.6], [0.5, 0.58]],
        labels=["u1", "u2"],
        baseline=0.5,
        baseline_label="chance",
        value_range=(0.4, 0.7),
    )
    # Vertical -> value axis is y: range there, reference line horizontal.
    assert list(fig.layout.yaxis.range) == [0.4, 0.7]
    assert any(
        shape.type == "line" and shape.y0 == 0.5 and shape.y1 == 0.5
        for shape in fig.layout.shapes
    )


def test_plot_ranked_bar_missing_column_raises():
    df = pd.DataFrame({"v": [1.0], "lab": ["a"], "grp": ["x"]})
    with pytest.raises(KeyError):
        plot_ranked_bar(df, value="v", category="lab", color="missing")


def test_plot_scatter_grouped_bar_errors():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(TypeError):
        plot_scatter([[1, 2]], x="a", y="b")  # not a DataFrame
    with pytest.raises(KeyError):
        plot_scatter(df, x="a", y="missing")
    with pytest.raises(TypeError):
        plot_grouped_bar([[1, 2]], x="a", y="b", group="g")
    with pytest.raises(KeyError):
        plot_grouped_bar(df, x="a", y="b", group="missing")


def test_plot_distribution_groups():
    groups = [np.random.randn(50), np.random.randn(50)]
    labels = ["A", "B"]
    fig = plot_distribution_groups(
        groups,
        labels,
        kind="violin",
        color=["red", "blue"],
        show_points=True,
        sig_pairs=[(0, 1, "*")],
    )
    assert isinstance(fig, go.Figure)

    fig2 = plot_distribution_groups(groups, labels, kind="box", color="green")
    assert isinstance(fig2, go.Figure)


def test_plot_heatmap():
    data = np.random.randn(5, 5)
    fig = plot_heatmap(
        data, x_labels=["A", "B", "C", "D", "E"], y_labels=["1", "2", "3", "4", "5"]
    )
    assert isinstance(fig, go.Figure)

    fig2 = plot_heatmap(
        pd.DataFrame(data),
        cmap="Blues",
        center=0,
        annotate=True,
        annotation_format=".2f",
    )
    assert isinstance(fig2, go.Figure)


def test_plot_timecourses():
    data = np.random.randn(20, 2, 10)  # n_trials, n_channels, n_times
    group_labels = [0] * 10 + [1] * 10

    fig = plot_timecourses(
        data,
        times=np.arange(10),
        channel_names=["Ch1", "Ch2"],
        group_labels=group_labels,
        error_style="band",
    )
    assert isinstance(fig, go.Figure)

    data_2d = np.random.randn(2, 10)  # n_channels, n_times
    fig2 = plot_timecourses(data_2d, add_zero=True, error_style=None)
    assert isinstance(fig2, go.Figure)


def test_interactive_base_edge_cases():
    # plot_bar list input
    plot_bar([1, 2], labels=["A", "B"])
    plot_bar(pd.Series([1, 2], index=["A", "B"]))
    with pytest.raises(ValueError):
        plot_bar([1, 2], labels=["A"])  # mismatched

    # plot_distribution_groups errors
    with pytest.raises(ValueError):
        plot_distribution_groups([[1]], ["A", "B"])
    with pytest.raises(ValueError):
        plot_distribution_groups([[1]], ["A"], kind="invalid")

    plot_distribution_groups(
        [[1], [2]], ["A", "B"], color=["red"]
    )  # too short color list
    plot_distribution_groups([[1]], ["A"], color="blue")  # string color

    # plot_heatmap errors
    with pytest.raises(ValueError):
        plot_heatmap(np.random.randn(5))
    plot_heatmap(np.random.randn(2, 2), vmin=0, vmax=1)

    # plot_timecourses errors
    with pytest.raises(ValueError):
        plot_timecourses(np.random.randn(10))  # not 2D or 3D
    with pytest.raises(ValueError):
        plot_timecourses(np.random.randn(2, 10), times=np.arange(5))
    with pytest.raises(ValueError):
        plot_timecourses(np.random.randn(2, 10), channel_names=["Ch1"])

    # pd.DataFrame to numpy in plot_timecourses
    df = pd.DataFrame(np.random.randn(2, 10))
    plot_timecourses(df)

    # Empty data group test
    plot_timecourses(np.random.randn(2, 10), rois={"R1": ["missing"]})
    plot_timecourses(np.random.randn(2, 10), rois=["0"])  # pass strings

    # palette variations
    plot_timecourses(np.random.randn(1, 2, 10), group_labels=[0], palette={"0": "red"})
    plot_timecourses(np.random.randn(1, 2, 10), group_labels=[0], palette=["blue"])
    with pytest.raises(ValueError):
        plot_timecourses(
            np.random.randn(1, 2, 10),
            group_labels=[0],
            palette=["invalid_color_test_xyz"],
        )
