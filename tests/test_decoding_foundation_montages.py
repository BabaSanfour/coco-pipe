"""Faithful fixed-montage channel construction for biot/bendr/eegpt.

The pure-logic tests (adapters, capability gates, braindecode-constant drift)
run offline. An opt-in end-to-end smoke that downloads the real checkpoints is
gated behind ``COCO_PIPE_FM_NETWORK_TESTS=1``.
"""

import os

import numpy as np
import pytest

from coco_pipe.decoding.foundation_models import _montages

NAMES_19 = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _biot_pairs():
    from braindecode.models.biot import BIOT_CHANNEL_ORDER

    return [tuple(label.split("-")) for label in BIOT_CHANNEL_ORDER[:16]]


def _bendr_order():
    from braindecode.models.bendr import BENDR_CHANNEL_ORDER

    from coco_pipe.decoding.foundation_models._channels import normalize_channel_names

    return normalize_channel_names(BENDR_CHANNEL_ORDER[:-1])


# --------------------------------------------------------------------------- #
# braindecode-constant sanity
# --------------------------------------------------------------------------- #
def test_bendr_order_is_19_without_scale():
    order = _bendr_order()
    assert len(order) == 19
    assert "SCALE" not in order


def test_eegpt_vocab_covers_standard_19():
    from braindecode.models.eegpt import CHANNEL_DICT

    vocab = {name.upper() for name in CHANNEL_DICT}
    assert all(name in vocab for name in NAMES_19)


# --------------------------------------------------------------------------- #
# BIOT — exact bipolar derivation
# --------------------------------------------------------------------------- #
def test_biot_adapter_is_exact_subtraction():
    plan = _montages.plan_channels("biot", NAMES_19)
    assert plan.status == "available"
    assert plan.model_n_chans == 16
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 19, 50)).astype(np.float32)
    out = plan.adapter(X)
    assert out.shape == (3, 16, 50)
    index = {name: i for i, name in enumerate(NAMES_19)}
    for k, (anode, cathode) in enumerate(_biot_pairs()):
        expected = X[:, index[anode], :] - X[:, index[cathode], :]
        np.testing.assert_allclose(out[:, k, :], expected, rtol=1e-6)


def test_biot_missing_electrode_is_unsupported():
    missing = [n for n in NAMES_19 if n != "F7"]
    plan = _montages.plan_channels("biot", missing)
    assert plan.status == "unsupported"
    assert "F7" in plan.reason


# --------------------------------------------------------------------------- #
# BENDR — reorder + dn3 normalization + computed SCALE
# --------------------------------------------------------------------------- #
def test_bendr_adapter_reorders_and_appends_scale():
    plan = _montages.plan_channels("bendr", NAMES_19)
    assert plan.status == "available"
    assert plan.model_n_chans == 20
    assert plan.provenance["scale_channel"] == "computed"
    rng = np.random.default_rng(1)
    # microvolt-scale window
    X = (rng.standard_normal((2, 19, 64)) * 30).astype(np.float32)
    out = plan.adapter(X)
    assert out.shape == (2, 20, 64)
    order = _bendr_order()
    index = {name: i for i, name in enumerate(NAMES_19)}
    order_idx = [index[name] for name in order]
    for w in range(2):
        window = X[w, order_idx, :]
        ptp = float(window.max() - window.min())
        scale = 2.0 * (min(ptp / _montages.BENDR_SCALE_MAX, 1.0) - 0.5)
        np.testing.assert_allclose(
            out[w, :19], _montages.min_max_normalize(window), rtol=1e-6
        )
        np.testing.assert_allclose(out[w, 19], scale, rtol=1e-6)
        # EEG rows are min-max normalized into [-1, 1].
        assert out[w, :19].min() >= -1.0 - 1e-6
        assert out[w, :19].max() <= 1.0 + 1e-6


def test_bendr_scale_constant_matches_tueg_config():
    # data_max - data_min from SPOClab-ca/BENDR pretraining.yml (TUEG).
    assert pytest.approx(4860.625830472267) == _montages.BENDR_SCALE_MAX


def test_bendr_missing_electrode_is_unsupported():
    plan = _montages.plan_channels("bendr", [n for n in NAMES_19 if n != "CZ"])
    assert plan.status == "unsupported"
    assert "CZ" in plan.reason


def test_min_max_normalize_constant_window_is_zero():
    x = np.full((4, 10), 7.0)
    np.testing.assert_array_equal(_montages.min_max_normalize(x), np.zeros((4, 10)))


# --------------------------------------------------------------------------- #
# EEGPT — name-indexed subset
# --------------------------------------------------------------------------- #
def test_eegpt_all_vocab_is_passthrough():
    plan = _montages.plan_channels("eegpt", NAMES_19)
    assert plan.status == "available"
    assert plan.model_n_chans == 19
    assert plan.provenance["dropped_channels"] == []
    # No channel is dropped → no adapter needed.
    assert plan.adapter is None


def test_eegpt_drops_non_vocab_channels():
    names = [*NAMES_19, "EMG1", "ECG"]
    plan = _montages.plan_channels("eegpt", names)
    assert plan.status == "available"
    assert plan.model_n_chans == 19
    assert set(plan.provenance["dropped_channels"]) == {"EMG1", "ECG"}
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2, 21, 40)).astype(np.float32)
    out = plan.adapter(X)
    assert out.shape == (2, 19, 40)
    np.testing.assert_allclose(out, X[:, :19, :], rtol=1e-6)


def test_eegpt_no_vocab_overlap_is_unsupported():
    plan = _montages.plan_channels("eegpt", ["EMG1", "ECG", "EOG"])
    assert plan.status == "unsupported"


def test_plan_channels_rejects_non_special_model():
    with pytest.raises(ValueError, match="no faithful construction"):
        _montages.plan_channels("cbramod", NAMES_19)


# --------------------------------------------------------------------------- #
# fill_missing_channels (opt-in unipolar interpolation)
# --------------------------------------------------------------------------- #
def test_fill_missing_makes_biot_available_and_runs():
    missing = [n for n in NAMES_19 if n != "F7"]  # 18 present, F7 absent
    plan = _montages.plan_channels("biot", missing, fill_missing=True, sfreq=200.0)
    assert plan.status == "available"
    assert plan.provenance["filled_channels"] == ["F7"]
    rng = np.random.default_rng(3)
    X = (rng.standard_normal((2, 18, 64)) * 20).astype(np.float32)
    out = plan.adapter(X)  # interpolates F7, then derives 16 bipolar channels
    assert out.shape == (2, 16, 64)


# --------------------------------------------------------------------------- #
# Opt-in end-to-end smoke against the real checkpoints
# --------------------------------------------------------------------------- #
network = pytest.mark.skipif(
    os.environ.get("COCO_PIPE_FM_NETWORK_TESTS") != "1",
    reason="set COCO_PIPE_FM_NETWORK_TESTS=1 to download checkpoints",
)


@network
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "model_key,n_times,feat_dim",
    [("biot", 2000, 256), ("bendr", 768, 512), ("eegpt", 1000, 2048)],
)
def test_real_checkpoint_loads_and_forwards(model_key, n_times, feat_dim):
    from coco_pipe.decoding._specs import SignalMetadata
    from coco_pipe.decoding.foundation_models.extraction import (
        FoundationEmbeddingExtractor,
    )

    rng = np.random.default_rng(0)
    X = (rng.standard_normal((2, 19, n_times)) * 20).astype(np.float32)
    meta = SignalMetadata(sfreq=200.0, ch_names=NAMES_19)
    extractor = FoundationEmbeddingExtractor(model_key)
    result = extractor.extract(X, signal_metadata=meta)
    assert result.window_embeddings.shape == (2, feat_dim)
