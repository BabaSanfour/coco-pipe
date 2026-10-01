import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from coco_pipe.viz.theme import (
    DIVERGING,
    QUALITATIVE,
    SEQUENTIAL,
    _palette_for,
    coco_theme,
    figure_size,
    save_figure,
    set_coco_theme,
)


def test_theme_context_restores_rcparams():
    original = mpl.rcParams["axes.labelsize"]
    with coco_theme("poster"):
        assert mpl.rcParams["axes.labelsize"] == 16
    assert mpl.rcParams["axes.labelsize"] == original


def test_theme_globals_and_helpers(tmp_path):
    assert DIVERGING == "RdBu_r"
    assert SEQUENTIAL == "viridis"
    assert QUALITATIVE == "tab10"
    width, height = figure_size(columns=2, aspect_ratio=0.5)
    assert width == 7.0
    assert height == 3.5
    set_coco_theme("paper")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "figure.png"
    save_figure(fig, str(path), dpi=72)
    plt.close(fig)
    assert path.exists()


def test_coco_theme_invalid_mode():
    with (
        pytest.raises(ValueError, match="Unknown coco theme mode"),
        coco_theme(mode="invalid_mode"),
    ):
        pass


def test_coco_theme_colorblind():
    with coco_theme(mode="paper", colorblind=True):
        cycle = mpl.rcParams["axes.prop_cycle"]
        # The colorblind cycle is set
        assert len(cycle) == 8


def test_figure_size_invalid():
    with pytest.raises(ValueError, match="columns must be 1 or 2"):
        figure_size(columns=3)
    with pytest.raises(ValueError, match="aspect_ratio must be positive"):
        figure_size(aspect_ratio=-1)


def test_figure_size_width_pt():
    width, height = figure_size(width_pt=72.27, aspect_ratio=2.0)
    assert width == 1.0
    assert height == 2.0


def test_palette_for():
    assert _palette_for(True) == DIVERGING
    assert _palette_for(False) == SEQUENTIAL
