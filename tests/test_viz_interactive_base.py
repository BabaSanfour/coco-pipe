import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from coco_pipe.viz.interactive.base import (
    plot_bar,
    plot_distribution_groups,
    plot_heatmap,
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
