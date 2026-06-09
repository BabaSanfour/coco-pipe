import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from coco_pipe import viz as top_viz
from coco_pipe.viz._utils import (
    SensorLayout,
    coerce_decoding_frame,
    coerce_sensor_layout,
    finalize_axes,
    require_columns,
    select_rows,
)
from coco_pipe.viz.base import (
    _colored_line_collection,
    _plot_alpha_encoded_line,
    plot_bar,
    plot_distribution_groups,
    plot_error_points,
    plot_heatmap,
    plot_hexbin,
    plot_line,
    plot_scatter2d,
    plot_scatter3d,
    plot_streamfield,
    plot_topomap,
)
from coco_pipe.viz.theme import figure_size
from tests.fixtures.synthetic_result import make_synthetic_result


def test_coerce_decoding_frame_accepts_result_and_dataframe():
    result = make_synthetic_result()
    scores = coerce_decoding_frame(result, accessor="get_detailed_scores")
    assert {"Model", "Metric", "Value"}.issubset(scores.columns)
    df = pd.DataFrame({"A": [1]})
    assert coerce_decoding_frame(df).equals(df)
    with pytest.raises(TypeError):
        coerce_decoding_frame([{"A": 1}], accessor="get_detailed_scores")


def test_require_columns_and_select_rows():
    df = pd.DataFrame(
        {
            "Model": ["a", "b"],
            "Metric": ["acc", "auc"],
            "Fold": [0, 1],
            "Group": ["g1", "g2"],
        }
    )
    assert len(select_rows(df, model="a", metric="acc", fold=0, group="g1")) == 1
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(df, ["Missing"], context="accessor")


def test_coerce_sensor_layout_from_coords():
    coords = {"E1": (0.0, 0.1), "E2": (0.2, 0.3)}
    layout = coerce_sensor_layout(coords=coords, names=["E2"])
    assert isinstance(layout, SensorLayout)
    assert layout.names == ["E2"]
    assert layout.positions.shape == (1, 2)
    with pytest.raises(ValueError):
        coerce_sensor_layout()


def test_finalize_axes_sets_common_labels_and_legend():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    finalize_axes(
        ax,
        title="title",
        xlabel="x",
        ylabel="y",
        legend=True,
        legend_title="Legend",
        xtick_rotation=30,
        xtick_ha="right",
    )
    assert ax.get_title() == "title"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_legend().get_title().get_text() == "Legend"
    plt.close(fig)


def test_plot_bar_uses_theme_size_and_colors_horizontal_bars():
    fig, ax = plot_bar(
        pd.Series({"a": 0.1, "b": 0.8, "c": 0.4}),
        orientation="horizontal",
        cmap="viridis",
    )
    width, height = fig.get_size_inches()
    expected_width, expected_height = figure_size(columns=2, aspect_ratio=4 / 7)
    assert round(width, 1) == round(expected_width, 1)
    assert round(height, 1) == round(expected_height, 1)

    colors = [patch.get_facecolor() for patch in ax.patches]
    assert len(set(colors)) > 1
    plt.close(fig)


def test_plot_bar_signed_cmap_and_empty_input():
    fig, ax = plot_bar(
        pd.Series({"a": -2.0, "b": 1.0}),
        cmap="RdBu_r",
        orientation="horizontal",
    )
    assert len(ax.patches) == 2
    plt.close(fig)

    with pytest.raises(ValueError, match="at least one"):
        plot_bar(pd.Series(dtype=float))


def test_plot_bar_can_preserve_input_order():
    fig, ax = plot_bar(pd.Series({"first": 0.1, "second": 0.9}), sort=False)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["first", "second"]
    plt.close(fig)


def test_plot_scatter2d_supports_colorbar_references_and_errors():
    fig, ax = plot_scatter2d(
        [0.0, 1.0],
        [1.0, 2.0],
        c=[0.2, 0.8],
        cmap="viridis",
        colorbar=True,
        xerr=[[0.1, 0.1], [0.2, 0.2]],
        reference_x=0.0,
        reference_y=1.0,
        xlabel="x",
        ylabel="y",
        title="scatter",
    )
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "scatter"
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_scatter2d_supports_categorical_labels():
    fig, ax = plot_scatter2d(
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        labels=["a", "b", "a"],
        legend_title="Group",
    )
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Group"
    plt.close(fig)


def test_plot_heatmap_supports_dataframe_labels_annotations_and_colorbar():
    matrix = pd.DataFrame([[1, 2], [3, 4]], index=["r1", "r2"], columns=["c1", "c2"])
    fig, ax = plot_heatmap(
        matrix,
        annotate=True,
        colorbar_label="Value",
        xlabel="x",
        ylabel="y",
        title="heatmap",
    )
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "heatmap"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["c1", "c2"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["r1", "r2"]
    assert len(ax.texts) == 4
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_line_supports_uncertainty_band_and_labels():
    fig, ax = plot_line(
        [0, 1, 2],
        [0.1, 0.3, 0.2],
        yerr=[0.01, 0.02, 0.03],
        marker="o",
        label="score",
        xlabel="time",
        ylabel="value",
        title="line",
        legend=True,
    )
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "value"
    assert ax.get_title() == "line"
    assert ax.get_legend() is not None
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_plot_hexbin_supports_colorbar_and_identity_reference():
    fig, ax = plot_hexbin(
        [0.0, 1.0, 2.0],
        [0.1, 1.1, 1.9],
        colorbar_label="Density",
        reference_identity=True,
        xlabel="original",
        ylabel="embedded",
        title="hex",
        legend=True,
    )
    assert ax.get_xlabel() == "original"
    assert ax.get_ylabel() == "embedded"
    assert ax.get_title() == "hex"
    assert len(fig.axes) == 2
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_streamfield_supports_points_and_colorbar():
    x = y = [0.0, 1.0]
    x_grid, y_grid = np.meshgrid(x, y)
    u_grid = np.ones_like(x_grid)
    v_grid = np.zeros_like(y_grid)

    fig, ax = plot_streamfield(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        points=[[0.0, 0.0], [1.0, 1.0]],
        colorbar_label="Speed",
        xlabel="x",
        ylabel="y",
        title="stream",
    )
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "stream"
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_error_points_supports_labels_errors_and_reference():
    fig, ax = plot_error_points(
        [0, 1],
        [0.4, 0.6],
        yerr=[0.05, 0.04],
        labels=["a", "b"],
        reference_y=0.5,
        ylabel="score",
        title="points",
    )
    assert ax.get_ylabel() == "score"
    assert ax.get_title() == "points"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b"]
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_plot_distribution_groups_supports_box_and_violin():
    groups = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    labels = ["a", "b"]

    fig, ax = plot_distribution_groups(groups, labels, kind="box", ylabel="score")
    assert ax.get_ylabel() == "score"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
    assert len(ax.collections) >= 1
    plt.close(fig)

    fig, ax = plot_distribution_groups(groups, labels, kind="violin", show_points=False)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
    plt.close(fig)


def test_colored_line_collection_exceptions():
    with pytest.raises(ValueError):
        _colored_line_collection([1], [2], [3], "viridis", 1.0)


def test_colored_line_collection_success():
    lc = _colored_line_collection([1, 2, 3], [1, 2, 3], [0, 1, 2], "viridis", 1.0)
    assert lc is not None


def test_plot_alpha_encoded_line():
    fig, ax = plt.subplots()
    # Less than 2 points
    _plot_alpha_encoded_line(ax, [1], [2], [3], "red")

    # All nan values
    _plot_alpha_encoded_line(ax, [1, 2], [2, 3], [np.nan, np.nan], "red")

    # Is close vmin vmax
    _plot_alpha_encoded_line(ax, [1, 2], [2, 3], [1.0, 1.0], "red")

    # Bad color
    _plot_alpha_encoded_line(ax, [1, 2], [2, 3], [1.0, 2.0], "badcolor")

    # Label set
    _plot_alpha_encoded_line(ax, [1, 2], [2, 3], [1.0, 2.0], "blue", label="line")
    plt.close(fig)


def test_plot_bar_exceptions():
    with pytest.raises(ValueError, match="must be 'vertical' or 'horizontal'"):
        plot_bar([1, 2], orientation="invalid")

    # with top_n
    fig, ax = plot_bar([3, 2, 1], top_n=2)
    plt.close(fig)


def test_plot_heatmap_exceptions():
    with pytest.raises(ValueError, match="must be 2D"):
        plot_heatmap([1, 2, 3])

    # center logic
    fig, ax = plot_heatmap([[1, 2], [3, 4]], center=2.5)
    plt.close(fig)

    # center logic outside bounds
    fig, ax = plot_heatmap([[1, 2], [3, 4]], center=5.0)
    plt.close(fig)


def test_plot_line_exceptions():
    with pytest.raises(ValueError, match="same length"):
        plot_line([1, 2], [1, 2, 3])

    with pytest.raises(ValueError, match="error_style"):
        plot_line([1, 2], [1, 2], yerr=[0.1, 0.1], error_style="invalid")


def test_plot_error_points_exceptions():
    with pytest.raises(ValueError, match="same length"):
        plot_error_points([1, 2], [1, 2, 3])


def test_plot_distribution_groups_exceptions():
    with pytest.raises(ValueError, match="groups and labels must have the same length"):
        plot_distribution_groups([[1]], ["a", "b"])

    with pytest.raises(ValueError, match="at least one finite value"):
        plot_distribution_groups([[np.nan]], ["a"])

    with pytest.raises(ValueError, match="kind must be 'box' or 'violin'"):
        plot_distribution_groups([[1]], ["a"], kind="invalid")

    # empty group skipping
    fig, ax = plot_distribution_groups([[1], []], ["a", "b"], kind="box")
    plt.close(fig)


def test_plot_scatter2d_exceptions():
    with pytest.raises(ValueError):
        plot_scatter2d([1, 2], [1, 2, 3])

    with pytest.raises(ValueError):
        plot_scatter2d([1, 2], [1, 2], c=[1])

    with pytest.raises(Exception):
        plot_scatter2d([1, 2], [1, 2], labels=["a"])

    # cmap object
    fig, ax = plot_scatter2d([1, 2], [1, 2], c=[1, 2], cmap=plt.get_cmap("viridis"))
    plt.close(fig)


def test_plot_scatter3d_exceptions():
    with pytest.raises(ValueError):
        plot_scatter3d([1, 2], [1, 2, 3], [1, 2])

    with pytest.raises(ValueError):
        plot_scatter3d([1, 2], [1, 2], [1, 2, 3])

    with pytest.raises(Exception):
        plot_scatter3d([1, 2], [1, 2], [1, 2], labels=["a"])

    # 3D
    fig, ax = plot_scatter3d([1, 2], [1, 2], [1, 2], labels=["a", "b"])
    plt.close(fig)

    # 3D colorbar
    fig, ax = plot_scatter3d([1, 2], [1, 2], [1, 2], c=[1, 2], colorbar=True)
    plt.close(fig)


def test_plot_hexbin_exceptions():
    with pytest.raises(ValueError):
        plot_hexbin([1, 2], [1, 2, 3])


def test_plot_streamfield_exceptions():
    with pytest.raises(ValueError):
        plot_streamfield(
            np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)), np.ones((2, 2))
        )

    # mask
    x, y = np.meshgrid([0, 1], [0, 1])
    u = np.ones((2, 2))
    u[0, 0] = np.nan
    fig, ax = plot_streamfield(x, y, u, np.ones((2, 2)))
    plt.close(fig)


def test_plot_with_existing_ax():
    fig, ax = plt.subplots()
    plot_bar([1, 2], ax=ax)
    plot_heatmap([[1, 2], [3, 4]], ax=ax)
    plot_line([1, 2], [1, 2], ax=ax)
    plot_error_points([1, 2], [1, 2], ax=ax)
    plot_distribution_groups([[1, 2]], ["a"], ax=ax)
    plot_scatter2d([1, 2], [1, 2], ax=ax)
    plot_hexbin([1, 2], [1, 2], ax=ax)
    x, y = np.meshgrid([0, 1], [0, 1])
    plot_streamfield(x, y, np.ones((2, 2)), np.ones((2, 2)), ax=ax)
    plt.close(fig)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    plot_scatter3d([1, 2], [1, 2], [1, 2], ax=ax3d)
    plt.close(fig)


def test_plot_topomap():
    values = pd.Series([1.0, -1.0], index=["EEG 001", "EEG 002"])
    coords = {"EEG 001": (0.0, 0.1), "EEG 002": (0.2, 0.3)}

    # default
    fig, ax = plot_topomap(values, coords=coords)
    plt.close(fig)

    # existing ax, cbar, not symmetric, labels
    fig, ax = plt.subplots()
    plot_topomap(
        values, coords=coords, ax=ax, symmetric=False, cbar=True, sensors="labels"
    )
    plt.close(fig)

    # exceptions
    with pytest.raises(ValueError, match="No matching"):
        plot_topomap(values, coords={"BAD": (0, 0)})


def test_top_level_viz_exports_new_helpers():
    assert top_viz.plot_bar
    assert top_viz.plot_topomap
    assert top_viz.plot_decoding_scores
    assert top_viz.plot_null_interval_summary
    assert top_viz.plot_coranking_matrix
    assert not hasattr(top_viz, "finalize_axes")
    assert not hasattr(top_viz, "SensorLayout")
    assert not hasattr(top_viz, "coerce_sensor_layout")
    assert not hasattr(top_viz, "coerce_decoding_frame")
    assert not hasattr(top_viz, "require_columns")
    assert not hasattr(top_viz, "select_rows")
