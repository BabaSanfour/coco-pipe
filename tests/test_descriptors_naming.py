"""Tests for the descriptor column-name contract (descriptors.naming)."""

import pytest

from coco_pipe.descriptors._constants import KNOWN_FAMILY_TOKENS
from coco_pipe.descriptors.naming import (
    parse_descriptor_feature_column,
    split_family_token,
)


def test_split_family_token_prefix():
    assert split_family_token("band_abs_alpha", KNOWN_FAMILY_TOKENS) == (
        "band",
        "abs_alpha",
    )


def test_split_family_token_embedded():
    # An aggregation-stat prefix before the family token is preserved.
    family, remainder = split_family_token(
        "mean_complexity_sample_entropy", KNOWN_FAMILY_TOKENS
    )
    assert (family, remainder) == ("complexity", "mean_sample_entropy")


def test_split_family_token_no_known_family():
    assert split_family_token("global_rms", KNOWN_FAMILY_TOKENS) == (None, "global_rms")


def test_split_family_token_honours_caller_tokens():
    assert split_family_token("custom_metric", ("custom",)) == ("custom", "metric")


def test_parse_sensor_column():
    assert parse_descriptor_feature_column(
        "band_abs_alpha_ch-Fz", KNOWN_FAMILY_TOKENS
    ) == {
        "column": "band_abs_alpha_ch-Fz",
        "family": "band",
        "feature": "abs_alpha",
        "scope": "sensor",
        "sensor": "Fz",
    }


def test_parse_sensor_group_column():
    parsed = parse_descriptor_feature_column(
        "complexity_sample_entropy_chgrp-front_left", KNOWN_FAMILY_TOKENS
    )
    assert parsed["scope"] == "sensor_group"
    assert parsed["sensor"] == "front_left"
    assert parsed["feature"] == "sample_entropy"


def test_parse_keeps_earlier_channel_markers_in_measure():
    # Only the last scope marker is the scope; earlier _ch- stays in the measure.
    parsed = parse_descriptor_feature_column(
        "band_cross_ch-Fz_ch-Pz", KNOWN_FAMILY_TOKENS
    )
    assert parsed["sensor"] == "Pz"
    assert parsed["feature"] == "cross_ch-Fz"


def test_parse_without_scope_marker_raises():
    with pytest.raises(ValueError, match="Could not parse descriptor column"):
        parse_descriptor_feature_column("band_global_rms", KNOWN_FAMILY_TOKENS)


def test_parse_without_known_family_raises():
    with pytest.raises(ValueError, match="does not contain a known family token"):
        parse_descriptor_feature_column("mystery_metric_ch-Fz", KNOWN_FAMILY_TOKENS)


def test_parse_prefixed_family_column():
    parsed = parse_descriptor_feature_column(
        "mean_complexity_sample_entropy_chgrp-front_left", KNOWN_FAMILY_TOKENS
    )
    assert parsed["family"] == "complexity"
    assert parsed["feature"] == "mean_sample_entropy"


def test_parse_honours_caller_supplied_family_tokens():
    parsed = parse_descriptor_feature_column("custom_metric_ch-Fz", ("custom",))
    assert parsed["family"] == "custom"
    assert parsed["feature"] == "metric"
