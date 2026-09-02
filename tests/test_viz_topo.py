"""Tests for the montage-based topographic helpers in coco_pipe.viz.topo."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from coco_pipe.viz.topo import (
    feature_names_are_channels,
    plot_topomap_from_channel_values,
    plot_topomap_selector,
    standard_montage_channels,
)

_CHANNELS = ["Fz", "Cz", "Pz", "Oz"]


# --- standard_montage_channels ----------------------------------------------


def test_standard_montage_channels_lowercased_and_cached():
    names = standard_montage_channels("standard_1020")
    assert isinstance(names, frozenset)
    assert {"fz", "cz", "pz", "oz"} <= names
    assert all(name == name.lower() for name in names)
    # lru_cache returns the same object for the same montage.
    assert standard_montage_channels("standard_1020") is names


# --- feature_names_are_channels ---------------------------------------------


def test_feature_names_are_channels_default_montage():
    assert feature_names_are_channels(_CHANNELS) is True


def test_feature_names_are_channels_accepts_legacy_names_by_default():
    # MNE's standard_1020 montage carries both legacy (T3/T4) and modern
    # (T7/T8) nomenclature, so both pass the default gate.
    assert feature_names_are_channels(["T3", "T4", "Cz"]) is True
    assert feature_names_are_channels(["T7", "T8", "Cz"]) is True


def test_feature_names_are_channels_custom_vocab_enables_and_restricts():
    # A custom vocabulary lets non-montage names through ...
    made_up = ["nz1", "nz2", "nz3"]
    assert feature_names_are_channels(made_up) is False
    assert feature_names_are_channels(made_up, ["nz1", "nz2", "nz3"]) is True
    # ... and restricts otherwise-valid montage names to the given set.
    assert feature_names_are_channels(["Fz", "Cz", "Pz"], ["fz", "cz"]) is False


def test_feature_names_are_channels_rejects_non_channels():
    assert feature_names_are_channels(["alpha_Cz", "beta_Fz", "theta_Pz"]) is False


def test_feature_names_are_channels_requires_at_least_three():
    assert feature_names_are_channels(["Fz", "Cz"]) is False
    assert feature_names_are_channels(None) is False
    assert feature_names_are_channels([]) is False


# --- plot_topomap_from_channel_values ---------------------------------------


def test_plot_topomap_from_channel_values_returns_figure():
    fig = plot_topomap_from_channel_values(_CHANNELS, [1.0, 2.0, 3.0, 4.0], "title")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_topomap_from_channel_values_bad_channels_overlay():
    fig = plot_topomap_from_channel_values(
        _CHANNELS, [1.0, 2.0, 3.0, 4.0], "title", bad_channels=["Cz"]
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_topomap_from_channel_values_none_on_empty():
    assert plot_topomap_from_channel_values([], [], "t") is None


def test_plot_topomap_from_channel_values_none_on_length_mismatch():
    assert plot_topomap_from_channel_values(_CHANNELS, [1.0, 2.0], "t") is None


def test_plot_topomap_from_channel_values_rejects_duplicate_names():
    with pytest.raises(ValueError, match=r"must be unique.*Fz"):
        plot_topomap_from_channel_values(
            ["Fz", "Cz", "Pz", "Fz"],
            [1.0, 2.0, 3.0, 4.0],
            "t",
        )


def test_plot_topomap_from_channel_values_aligns_values_after_montage_filter():
    fig = plot_topomap_from_channel_values(
        ["Fz", "NotAChannel", "Cz", "Pz"],
        [1.0, 99.0, 2.0, 3.0],
        "t",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_topomap_from_channel_values_none_when_fewer_than_three_resolve():
    # Only one name resolves against the montage -> below the 3-channel floor.
    assert (
        plot_topomap_from_channel_values(["Cz", "NOPE1", "NOPE2"], [1.0, 2.0, 3.0], "t")
        is None
    )


# --- plot_topomap_selector --------------------------------------------------


def test_plot_topomap_selector_returns_figure():
    fig = plot_topomap_selector(
        {
            "PC1": (_CHANNELS, [1.0, 2.0, 3.0, 4.0]),
            "PC2": (_CHANNELS, [4.0, 3.0, 2.0, 1.0]),
        },
        title="loadings",
        unit="loading",
    )
    assert isinstance(fig, go.Figure)
    # One dropdown button per rendered map.
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["PC1", "PC2"]


def test_plot_topomap_selector_none_on_empty():
    assert plot_topomap_selector({}, "t") is None


def test_plot_topomap_selector_none_when_no_maps_render():
    # Non-resolving channels -> every map is dropped -> None.
    assert plot_topomap_selector({"PC1": (["NOPE1", "NOPE2"], [1.0, 2.0])}, "t") is None
