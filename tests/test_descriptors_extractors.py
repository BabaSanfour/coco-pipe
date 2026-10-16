"""Band, parametric and complexity descriptors computed on real signals."""

import sys

import numpy as np
import pytest
from scipy.stats import kurtosis

from coco_pipe.descriptors import DescriptorPipeline

SFREQ = 250.0
CHANNELS = ["Fz", "Cz", "Pz"]
BANDS = {"alpha": (8.0, 13.0), "beta": (15.0, 30.0)}


@pytest.fixture(scope="module")
def signals():
    """1/f noise with a 10 Hz rhythm on Fz and a 20 Hz rhythm on Cz."""
    rng = np.random.default_rng(42)
    t = np.arange(0, 4, 1 / SFREQ)
    weights = 1 / (np.fft.rfftfreq(t.size, 1 / SFREQ) + 1.0)
    noise = np.fft.rfft(rng.standard_normal((5, len(CHANNELS), t.size)))
    X = np.fft.irfft(noise * weights, n=t.size)
    X[:, 0] += 2.0 * np.sin(2 * np.pi * 10 * t)
    X[:, 1] += 1.5 * np.sin(2 * np.pi * 20 * t)
    return X


def _extract(X, families, channels=CHANNELS, sfreq=SFREQ, **config):
    result = DescriptorPipeline({"families": families, **config}).extract(
        X=X, sfreq=sfreq, channel_names=channels
    )
    return dict(zip(result.coords["feature"], result.X.T, strict=True)), result.meta[
        "failures"
    ]


def test_band_powers_follow_the_injected_rhythms(signals):
    outputs = ["absolute_power", "log_absolute_power", "relative_power", "ratios"]
    bands = {"enabled": True, "outputs": outputs, "bands": BANDS}
    values, failures = _extract(
        signals, {"bands": {**bands, "ratio_pairs": [("alpha", "beta")]}}
    )

    assert not failures
    assert len(values) == 4 * len(CHANNELS) * len(BANDS) - len(CHANNELS)
    assert (values["band_abs_alpha_ch-Fz"] > 100 * values["band_abs_alpha_ch-Pz"]).all()
    assert (values["band_rel_beta_ch-Cz"] > 0.9).all()
    np.testing.assert_allclose(
        values["band_log_abs_alpha_ch-Fz"],
        np.log10(values["band_abs_alpha_ch-Fz"]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        values["band_ratio_alpha_beta_ch-Fz"],
        values["band_abs_alpha_ch-Fz"] / values["band_abs_beta_ch-Fz"],
        rtol=1e-4,
    )


def test_band_powers_of_white_noise_tile_the_frequency_window():
    sfreq, sigma = 200.0, 10e-6
    X = np.random.default_rng(0).normal(scale=sigma, size=(20, 3, 2000))
    tiling = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    bands = {
        "enabled": True,
        "fmin": 1.0,
        "fmax": 45.0,
        "bands": tiling,
        "outputs": ["absolute_power", "relative_power"],
    }
    values, failures = _extract(X, {"bands": bands}, sfreq=sfreq, precision="float64")

    assert not failures
    for channel in CHANNELS:
        relative = [values[f"band_rel_{band}_ch-{channel}"] for band in tiling]
        np.testing.assert_allclose(np.sum(relative, axis=0), 1.0, rtol=1e-12)
    # one-sided density of white noise: 2 sigma^2 / sfreq
    for band, (low, high) in tiling.items():
        power = np.mean([values[f"band_abs_{band}_ch-{c}"] for c in CHANNELS])
        assert power == pytest.approx(2 * sigma**2 / sfreq * (high - low), rel=0.05)


def test_relative_power_of_a_flat_channel_is_nan_and_logged(signals):
    X = signals.copy()
    X[:, 2] = 0.0
    bands = {"enabled": True, "outputs": ["relative_power"], "bands": BANDS}
    values, failures = _extract(X, {"bands": bands})

    assert np.isnan(values["band_rel_alpha_ch-Pz"]).all()
    assert np.isfinite(values["band_rel_alpha_ch-Fz"]).all()
    assert {failure["channel_name"] for failure in failures} == {"Pz"}


def test_ratios_below_the_denominator_floor_are_nan(signals):
    bands = {
        "enabled": True,
        "outputs": ["ratios"],
        "bands": BANDS,
        "ratio_pairs": [("alpha", "beta")],
        "min_denominator_power": 1e6,
    }
    values, failures = _extract(signals, {"bands": bands})

    assert all(np.isnan(column).all() for column in values.values())
    assert {failure["exception_type"] for failure in failures} == {"NumericalIssue"}


def test_parametric_fit_finds_the_peaks_and_the_aperiodic_slope(signals):
    values, failures = _extract(
        signals,
        {"parametric": {"enabled": True, "outputs": ["aperiodic", "peak_summary"]}},
    )

    assert not failures
    for channel in CHANNELS:
        assert (values[f"param_exponent_ch-{channel}"] > 0).all()
    np.testing.assert_allclose(values["param_peak_freq_dom_ch-Fz"], 10.0, atol=0.5)
    np.testing.assert_allclose(values["param_alpha_peak_freq_ch-Fz"], 10.0, atol=0.5)
    np.testing.assert_allclose(values["param_peak_freq_dom_ch-Cz"], 20.0, atol=0.5)
    # Cz has no peak in the alpha band
    assert np.isnan(values["param_alpha_peak_freq_ch-Cz"]).all()


def test_corrected_band_power_is_measured_only_where_a_peak_was_fitted(signals):
    values, failures = _extract(
        signals,
        {
            "bands": {
                "enabled": True,
                "outputs": [
                    "corrected_absolute_power",
                    "corrected_log_absolute_power",
                    "corrected_relative_power",
                ],
                "bands": BANDS,
            },
            "parametric": {"enabled": True, "outputs": ["peak_summary"]},
        },
    )

    # "no peak" is a result, not a failure
    assert not failures
    assert (values["band_corr_abs_alpha_ch-Fz"] > 0).all()
    assert (values["band_corr_abs_beta_ch-Cz"] > 0).all()
    assert (values["band_corr_rel_alpha_ch-Fz"] > 0.9).all()
    # BANDS["alpha"] matches the band behind alpha_peak_freq
    for channel in CHANNELS:
        no_peak = np.isnan(values[f"param_alpha_peak_freq_ch-{channel}"])
        for output in ("abs", "log_abs", "rel"):
            column = values[f"band_corr_{output}_alpha_ch-{channel}"]
            np.testing.assert_array_equal(np.isnan(column), no_peak)
    assert np.isnan(values["band_corr_abs_alpha_ch-Cz"]).all()
    assert np.isnan(values["band_corr_log_abs_beta_ch-Fz"]).all()


def test_simple_complexity_measures_match_their_definition():
    signal = np.array([1.0, -2.0, 2.5, -4.0, 8.0])
    values, _ = _extract(
        signal.reshape(1, 1, -1),
        {
            "complexity": {
                "enabled": True,
                "measures": ["zero_crossings", "rms", "kurtosis"],
            }
        },
        channels=["Fz"],
    )

    # measured on the epoch centred on its own mean
    centred = signal - signal.mean()
    assert values["complexity_zero_crossings_ch-Fz"][0] == 3
    assert values["complexity_rms_ch-Fz"][0] == pytest.approx(
        np.sqrt(np.mean(centred**2))
    )
    assert values["complexity_kurtosis_ch-Fz"][0] == pytest.approx(
        kurtosis(signal, fisher=True, bias=False)
    )


def test_complexity_bandpass_removes_drift_below_the_band():
    times = np.arange(2000) / 200.0
    sine = np.sin(2 * np.pi * 10.0 * times)
    drifting = (sine + 3.0 * np.sin(2 * np.pi * 0.2 * times)).reshape(1, 1, -1)
    config = {"complexity": {"enabled": True, "measures": ["zero_crossings"]}}
    raw, _ = _extract(drifting, config, channels=["Fz"])
    config["complexity"]["bandpass"] = (1.0, 45.0)
    filtered, _ = _extract(drifting, config, channels=["Fz"])

    # a 10 Hz sine crosses zero 200 times in 10 s; a slow drift three times larger
    # hides most crossings until it is filtered out
    assert raw["complexity_zero_crossings_ch-Fz"][0] < 50
    assert filtered["complexity_zero_crossings_ch-Fz"][0] == pytest.approx(200, abs=2)


def test_auto_backend_computes_every_measure_and_separates_order_from_noise():
    measures = [
        "sample_entropy",
        "spectral_entropy",
        "approx_entropy",
        "svd_entropy",
        "petrosian_fd",
        "katz_fd",
        "higuchi_fd",
        "hjorth_mobility",
        "hjorth_complexity",
        "fuzzy_entropy",
        "shannon_entropy",
        "dispersion_entropy",
        "hurst_exponent",
    ]
    t = np.arange(512) / 128.0
    noise = np.random.default_rng(0).standard_normal(t.size)
    X = np.stack([np.sin(2 * np.pi * 5 * t), noise])[None]

    values, failures = _extract(
        X,
        {"complexity": {"enabled": True, "backend": "auto", "measures": measures}},
        channels=["sine", "noise"],
        sfreq=128.0,
    )

    assert not failures
    assert all(np.isfinite(column).all() for column in values.values())
    for measure in ("sample_entropy", "spectral_entropy", "svd_entropy"):
        assert (
            values[f"complexity_{measure}_ch-sine"]
            < values[f"complexity_{measure}_ch-noise"]
        )


@pytest.mark.parametrize(
    ("backend", "measure"),
    [("antropy", "fuzzy_entropy"), ("neurokit2", "katz_fd")],
)
def test_backends_reject_measures_they_do_not_provide(signals, backend, measure):
    with pytest.raises(ValueError, match=f"not supported by backend '{backend}'"):
        _extract(
            signals,
            {
                "complexity": {
                    "enabled": True,
                    "backend": backend,
                    "measures": [measure],
                }
            },
        )


@pytest.mark.parametrize(
    ("module", "backend", "measure"),
    [
        ("antropy", "antropy", "sample_entropy"),
        ("neurokit2", "neurokit2", "fuzzy_entropy"),
    ],
)
def test_missing_backend_names_the_install_extra(
    monkeypatch, signals, module, backend, measure
):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=r"coco-pipe\[descriptors\]"):
        _extract(
            signals,
            {
                "complexity": {
                    "enabled": True,
                    "backend": backend,
                    "measures": [measure],
                }
            },
        )
