from coco_pipe.descriptors._constants import (
    DESCRIPTOR_SCOPE_RE,
    KNOWN_FAMILY_TOKENS,
)


def test_known_family_tokens_values():
    assert KNOWN_FAMILY_TOKENS == ("band", "param", "complexity")


def test_scope_re_sensor():
    match = DESCRIPTOR_SCOPE_RE.match("band_abs_alpha_ch-Fz")

    assert match is not None
    assert match.group(2) == "ch"
    assert match.group(3) == "Fz"


def test_scope_re_sensor_group():
    match = DESCRIPTOR_SCOPE_RE.match(
        "band_log_beta_chgrp-frontal",
    )

    assert match is not None
    assert match.group(2) == "chgrp"
    assert match.group(3) == "frontal"


def test_scope_re_rightmost_match():
    match = DESCRIPTOR_SCOPE_RE.match("band_cross_ch-Fz_ch-Pz")

    assert match is not None
    assert match.group(1) == "band_cross_ch-Fz"
    assert match.group(3) == "Pz"


def test_scope_re_no_scope_returns_none():
    assert DESCRIPTOR_SCOPE_RE.match("band_global_rms") is None
