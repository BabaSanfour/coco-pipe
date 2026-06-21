"""
Comprehensive Test Suite for Descriptor Extractors
==================================================

Unified tests for Spectral, Parametric, and Complexity descriptor extractors.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import kurtosis as scipy_kurtosis

import coco_pipe.descriptors.extractors._parametric_fit as param_fit_module
from coco_pipe.descriptors.configs import (
    BandDescriptorConfig,
    ComplexityDescriptorConfig,
    ParametricDescriptorConfig,
)
from coco_pipe.descriptors.extractors._parametric_fit import _ParametricFitBatch
from coco_pipe.descriptors.extractors._psd import compute_psd
from coco_pipe.descriptors.extractors.base import (
    _DescriptorBlock,
    make_failure_record,
)
from coco_pipe.descriptors.extractors.complexity import ComplexityDescriptorExtractor
from coco_pipe.descriptors.extractors.parametric import ParametricDescriptorExtractor
from coco_pipe.descriptors.extractors.spectral import BandDescriptorExtractor


@pytest.fixture
def signal_data():
    """Standard signal data: (n_obs, n_channels, n_times)."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    t = np.arange(0, 2, 1 / sfreq)
    n_obs, n_chans = 5, 3

    freqs = np.fft.rfftfreq(len(t), 1 / sfreq)
    weights = 1 / (freqs + 1.0)

    X = np.zeros((n_obs, n_chans, len(t)))
    for o in range(n_obs):
        for c in range(n_chans):
            white = rng.standard_normal(len(t))
            X[o, c, :] = np.fft.irfft(np.fft.rfft(white) * weights, n=len(t))

    X[:, 0, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += 1.5 * np.sin(2 * np.pi * 20 * t)

    return X, sfreq, ["Fz", "Cz", "Pz"]


@pytest.fixture
def psd_data(signal_data):
    """Standard PSD data: (n_obs, n_channels, n_freqs)."""
    X, sfreq, _ = signal_data
    psds, freqs = compute_psd(
        X, sfreq=sfreq, method="welch", fmin=1.0, fmax=45.0, n_jobs=1
    )
    return psds, freqs


@pytest.fixture
def mock_fit_batch(psd_data):
    """A mock ParametricFitBatch for testing corrected bands."""
    psds, freqs = psd_data
    n_obs, n_chans, _n_freqs = psds.shape

    periodic_psds = np.zeros_like(psds)
    # Add a "peak" at 10Hz (approx index)
    f_idx = np.argmin(np.abs(freqs - 10.0))
    periodic_psds[:, :, f_idx] = 1.0

    metrics = {
        "offset": np.zeros((n_obs, n_chans)),
        "exponent": np.ones((n_obs, n_chans)) * 1.5,
    }

    return _ParametricFitBatch(
        freqs=freqs,
        metrics=metrics,
        errors=[],
        periodic_psds=periodic_psds,
    )


def test_descriptor_block_structure():
    """Verify _DescriptorBlock simple data container."""
    X = np.zeros((5, 10))
    names = [f"desc_{i}" for i in range(10)]

    block = _DescriptorBlock(family="test", X=X, descriptor_names=names)
    assert block.X.shape == (5, 10)
    assert block.descriptor_names == names


def test_make_failure_record_schema():
    """Check fixed schema for failure records."""
    rec = make_failure_record(
        family="spectral",
        obs_index=5,
        exception_type="ValueError",
        message="test error",
        channel_index=2,
        channel_name="Cz",
    )
    assert rec["obs_index"] == 5
    assert rec["family"] == "spectral"
    assert rec["exception_type"] == "ValueError"
    assert rec["message"] == "test error"
    assert rec["channel_index"] == 2
    assert rec["channel_name"] == "Cz"


def test_basic_extraction(psd_data, signal_data):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    config = BandDescriptorConfig(
        enabled=True,
        outputs=["absolute_power", "relative_power"],
        bands={"alpha": (8, 12), "beta": (15, 30)},
    )
    extractor = BandDescriptorExtractor(config)

    block = extractor.extract_psd(
        psds,
        freqs,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
    )
    assert block.family == "bands"
    # 2 bands * 2 outputs * 3 channels = 12 columns
    assert block.X.shape == (psds.shape[0], 12)
    assert "band_abs_alpha_ch-Fz" in block.descriptor_names
    assert "band_rel_beta_ch-Pz" in block.descriptor_names


def test_corrected_outputs(psd_data, signal_data, mock_fit_batch):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    config = BandDescriptorConfig(
        enabled=True,
        outputs=["corrected_absolute_power", "corrected_ratios"],
        bands={"alpha": (8, 12), "beta": (13, 30)},
        ratio_pairs=[("alpha", "beta")],
    )
    extractor = BandDescriptorExtractor(config)

    block = extractor.extract_psd(
        psds,
        freqs,
        channel_names=ch_names,
        ids=None,
        fit_batch=mock_fit_batch,
        runtime=MagicMock(),
    )
    assert block.X.shape == (psds.shape[0], 9)
    assert "band_corr_abs_alpha_ch-Fz" in block.descriptor_names
    assert "band_corr_ratio_alpha_beta_ch-Pz" in block.descriptor_names


def test_missing_fit_batch_raises(psd_data, signal_data):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    config = BandDescriptorConfig(
        enabled=True,
        outputs=["corrected_absolute_power"],
    )
    extractor = BandDescriptorExtractor(config)

    with pytest.raises(ValueError, match="require a supplied parametric fit_batch"):
        extractor.extract_psd(
            psds,
            freqs,
            channel_names=ch_names,
            ids=None,
            runtime=MagicMock(),
        )


def test_band_resolution_error(psd_data, signal_data):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    config = BandDescriptorConfig(
        enabled=True,
        fmax=250.0,
        bands={"ultra": (100, 200)},
        outputs=["absolute_power"],
    )
    extractor = BandDescriptorExtractor(config)

    runtime = MagicMock()
    runtime.on_error = "collect"

    block = extractor.extract_psd(
        psds,
        freqs,
        channel_names=ch_names,
        ids=None,
        runtime=runtime,
    )
    assert len(block.failures) > 0
    assert block.failures[0]["exception_type"] == "BandResolutionError"


def test_spectral_capabilities_and_requests():
    config = BandDescriptorConfig(enabled=True)
    extractor = BandDescriptorExtractor(config)
    assert extractor.capabilities["requires_sfreq"] is True

    # corrected without fit_config raises
    config_corr = BandDescriptorConfig(
        enabled=True, outputs=["corrected_absolute_power"]
    )
    extractor_corr = BandDescriptorExtractor(config_corr, fit_config=None)
    with pytest.raises(ValueError, match="Corrected band outputs require"):
        extractor_corr.psd_request()


def test_spectral_extract_psd_edge_cases(signal_data):
    from unittest.mock import MagicMock

    _X, _sfreq, ch_names = signal_data

    # explicit log output coverage
    config = BandDescriptorConfig(
        enabled=True,
        outputs=["log_absolute_power", "corrected_log_absolute_power"],
    )
    extractor = BandDescriptorExtractor(config)
    psds = np.ones((1, 3, 10))
    freqs = np.linspace(1, 45, 10)
    fit_batch = _ParametricFitBatch(
        freqs=freqs,
        metrics={},
        periodic_psds=np.ones((1, 3, 10)),
        errors=[],
        meta={},
    )
    block = extractor.extract_psd(
        psds,
        freqs,
        ch_names,
        None,
        MagicMock(),
        fit_batch=fit_batch,
    )
    assert "band_log_abs_delta_ch-Fz" in block.descriptor_names
    assert "band_corr_log_abs_delta_ch-Fz" in block.descriptor_names

    config_rel = BandDescriptorConfig(
        enabled=True,
        outputs=["relative_power"],
        fmin=100.0,
        fmax=200.0,
        bands={"high": (120, 150)},
    )
    extractor_rel = BandDescriptorExtractor(config_rel)
    block_rel = extractor_rel.extract_psd(psds, freqs, ch_names, None, MagicMock())
    assert np.isnan(block_rel.X).all()

    fit_batch = _ParametricFitBatch(
        freqs=np.array([1, 10, 20]),  # Must be non-empty and non-None
        metrics={},
        periodic_psds=np.zeros((1, 3, 3)),
        errors=[(0, 0, "FakeError", "Fit failed")],
        meta={},
    )
    config_corr = BandDescriptorConfig(
        enabled=True, outputs=["corrected_relative_power"]
    )
    extractor_corr = BandDescriptorExtractor(config_corr)
    block_corr = extractor_corr.extract_psd(
        psds, freqs, ch_names, None, MagicMock(), fit_batch=fit_batch
    )
    assert len(block_corr.failures) > 0
    assert "FakeError" in block_corr.failures[0]["exception_type"]


def test_spectral_standalone_extract(signal_data):
    from unittest.mock import MagicMock

    X, sfreq, ch_names = signal_data
    config = BandDescriptorConfig(enabled=True, outputs=["absolute_power"])
    extractor = BandDescriptorExtractor(config)
    block = extractor.extract(X, sfreq, ch_names, None, MagicMock())
    assert block.X.shape == (5, 15)


def test_spectral_empty_output_block():
    # Empty output when no families enabled or no outputs requested
    config = BandDescriptorConfig(enabled=True, outputs=[])
    extractor = BandDescriptorExtractor(config)
    psds = np.ones((2, 2, 10))
    freqs = np.linspace(1, 45, 10)
    block = extractor.extract_psd(psds, freqs, ["ch1", "ch2"], None, MagicMock())
    assert block.X.shape == (2, 0)


def test_spectral_standalone_extract_raises_for_corrected(signal_data):
    from unittest.mock import MagicMock

    X, sfreq, ch_names = signal_data
    config = BandDescriptorConfig(enabled=True, outputs=["corrected_absolute_power"])
    extractor = BandDescriptorExtractor(config)
    with pytest.raises(ValueError, match="Corrected band outputs are only available"):
        extractor.extract(X, sfreq, ch_names, None, MagicMock())


def test_ratio_denominator_floor_turns_near_zero_ratios_into_nan():
    from unittest.mock import MagicMock

    freqs = np.array([9.0, 10.0, 20.0, 21.0], dtype=float)
    psds = np.zeros((1, 1, freqs.size), dtype=float)
    psds[0, 0, :2] = 1.0
    psds[0, 0, 2:] = 1e-14

    config = BandDescriptorConfig(
        enabled=True,
        outputs=["ratios"],
        bands={"alpha": (8.0, 12.0), "beta": (19.0, 22.0)},
        ratio_pairs=[("alpha", "beta")],
        min_denominator_power=1e-12,
    )
    extractor = BandDescriptorExtractor(config)

    block = extractor.extract_psd(
        psds,
        freqs,
        channel_names=["Fz"],
        ids=np.array(["obs-0"], dtype=object),
        runtime=MagicMock(),
    )

    assert block.descriptor_names == ["band_ratio_alpha_beta_ch-Fz"]
    assert np.isnan(block.X[0, 0])
    assert len(block.failures) == 1
    assert "NumericalIssue" in block.failures[0]["exception_type"]


def test_standalone_extract(signal_data):
    """Verify real specparam-based extraction."""
    X, sfreq, ch_names = signal_data
    config = ParametricDescriptorConfig(
        enabled=True,
        outputs=["aperiodic"],
        psd_method="welch",
        freq_range=(1.0, 45.0),
    )
    extractor = ParametricDescriptorExtractor(config)

    block = extractor.extract(
        X,
        sfreq=sfreq,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )
    assert "param_exponent_ch-Fz" in block.descriptor_names
    assert block.X.shape == (X.shape[0], 6)
    exponent_idx = block.descriptor_names.index("param_exponent_ch-Fz")
    offset_idx = block.descriptor_names.index("param_offset_ch-Fz")
    assert np.all(block.X[:, exponent_idx] > 0)
    assert np.all(np.isfinite(block.X[:, offset_idx]))


def test_extract_psd_requires_fit_batch(psd_data, signal_data):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    extractor = ParametricDescriptorExtractor(ParametricDescriptorConfig())

    with pytest.raises(ValueError, match="requires a supplied fit_batch"):
        extractor.extract_psd(
            psds,
            freqs,
            channel_names=ch_names,
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )


def test_peak_summary_emits_extended_peak_metrics(psd_data, signal_data):
    psds, freqs = psd_data
    _, _, ch_names = signal_data
    n_obs, n_chans, _ = psds.shape
    extractor = ParametricDescriptorExtractor(
        ParametricDescriptorConfig(enabled=True, outputs=["peak_summary"])
    )
    fit_batch = _ParametricFitBatch(
        freqs=freqs,
        metrics={
            "peak_count": np.full((n_obs, n_chans), 2.0),
            "peak_freq_dom": np.full((n_obs, n_chans), 10.0),
            "peak_power_dom": np.full((n_obs, n_chans), 0.5),
            "peak_bandwidth_dom": np.full((n_obs, n_chans), 2.0),
            "alpha_peak_freq": np.full((n_obs, n_chans), 10.0),
            "alpha_peak_power": np.full((n_obs, n_chans), 0.5),
        },
        errors=[],
        periodic_psds=None,
    )

    block = extractor.extract_psd(
        psds,
        freqs,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        fit_batch=fit_batch,
        obs_offset=0,
    )

    assert block.X.shape == (n_obs, 18)
    assert "param_peak_bandwidth_dom_ch-Fz" in block.descriptor_names
    assert "param_alpha_peak_freq_ch-Fz" in block.descriptor_names
    assert "param_alpha_peak_power_ch-Fz" in block.descriptor_names


def test_fit_single_spectrum_reads_dominant_and_alpha_peak_metrics(monkeypatch):
    class _FakeResultsModel:
        def get_component(self, name):
            raise AssertionError("Periodic PSD reconstruction is not used here.")

    class _FakeResults:
        has_model = True
        model = _FakeResultsModel()

        def get_params(self, name):
            if name == "aperiodic":
                return np.array([0.1, 1.5], dtype=float)
            if name == "periodic":
                return np.array(
                    [
                        [6.0, 0.2, 1.0],
                        [10.0, 0.7, 2.5],
                        [20.0, 0.5, 3.0],
                    ],
                    dtype=float,
                )
            raise KeyError(name)

        def get_metrics(self, *names):
            if names == ("error",):
                return np.array([0.05], dtype=float)
            if names == ("gof", "rsquared"):
                return np.array([0.98], dtype=float)
            raise KeyError(names)

    class _FakeSpectralModel:
        def __init__(self, **kwargs):
            self.results = _FakeResults()

        def fit(self, freqs, spectrum, freq_range):
            return None

    monkeypatch.setattr(
        param_fit_module,
        "import_optional_dependency",
        lambda *args, **kwargs: _FakeSpectralModel,
    )

    metrics, periodic_psd = param_fit_module.fit_single_spectrum(
        freqs=np.array([1.0, 10.0, 20.0], dtype=float),
        spectrum=np.array([1.0, 2.0, 1.5], dtype=float),
        config=ParametricDescriptorConfig(enabled=True, outputs=["peak_summary"]),
        need_periodic_psd=False,
    )

    assert periodic_psd is None
    assert metrics["peak_count"] == 3.0
    assert metrics["peak_freq_dom"] == pytest.approx(10.0)
    assert metrics["peak_power_dom"] == pytest.approx(0.7)
    assert metrics["peak_bandwidth_dom"] == pytest.approx(2.5)
    assert metrics["alpha_peak_freq"] == pytest.approx(10.0)
    assert metrics["alpha_peak_power"] == pytest.approx(0.7)


def test_fit_single_spectrum_returns_nan_for_missing_alpha_peak(monkeypatch):
    class _FakeResultsModel:
        def get_component(self, name):
            raise AssertionError("Periodic PSD reconstruction is not used here.")

    class _FakeResults:
        has_model = True
        model = _FakeResultsModel()

        def get_params(self, name):
            if name == "aperiodic":
                return np.array([0.2, 1.1], dtype=float)
            if name == "periodic":
                return np.array([[20.0, 0.5, 3.0]], dtype=float)
            raise KeyError(name)

        def get_metrics(self, *names):
            if names == ("error",):
                return np.array([0.02], dtype=float)
            if names == ("gof", "rsquared"):
                return np.array([0.99], dtype=float)
            raise KeyError(names)

    class _FakeSpectralModel:
        def __init__(self, **kwargs):
            self.results = _FakeResults()

        def fit(self, freqs, spectrum, freq_range):
            return None

    monkeypatch.setattr(
        param_fit_module,
        "import_optional_dependency",
        lambda *args, **kwargs: _FakeSpectralModel,
    )

    metrics, periodic_psd = param_fit_module.fit_single_spectrum(
        freqs=np.array([1.0, 10.0, 20.0], dtype=float),
        spectrum=np.array([1.0, 2.0, 1.5], dtype=float),
        config=ParametricDescriptorConfig(enabled=True, outputs=["peak_summary"]),
        need_periodic_psd=False,
    )

    assert periodic_psd is None
    assert metrics["peak_count"] == 1.0
    assert np.isnan(metrics["alpha_peak_freq"])
    assert np.isnan(metrics["alpha_peak_power"])


def test_backend_dispatch_antropy(signal_data):
    X, sfreq, ch_names = signal_data
    config = ComplexityDescriptorConfig(
        enabled=True, backend="antropy", measures=["spectral_entropy"]
    )
    extractor = ComplexityDescriptorExtractor(config)

    block = extractor.extract(
        X,
        sfreq=sfreq,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )
    assert "complexity_spectral_entropy_ch-Fz" in block.descriptor_names
    assert not np.isnan(block.X).any()


def test_backend_dispatch_neurokit2(signal_data):
    X, sfreq, ch_names = signal_data
    config = ComplexityDescriptorConfig(
        enabled=True, backend="neurokit2", measures=["shannon_entropy"]
    )
    extractor = ComplexityDescriptorExtractor(config)

    block = extractor.extract(
        X,
        sfreq=sfreq,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )
    assert "complexity_shannon_entropy_ch-Fz" in block.descriptor_names
    # Check that it's finite
    assert not np.isnan(block.X).any()


def test_mixed_execution_strategy(signal_data):
    """Verify execution paths for combined complexity measures."""
    X, sfreq, ch_names = signal_data
    config = ComplexityDescriptorConfig(
        enabled=True,
        backend="antropy",
        measures=["spectral_entropy", "sample_entropy"],
    )
    extractor = ComplexityDescriptorExtractor(config)

    block = extractor.extract(
        X,
        sfreq=sfreq,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )

    # 2 measures * 3 channels = 6 columns
    assert block.X.shape == (X.shape[0], 6)
    assert "complexity_spectral_entropy_ch-Fz" in block.descriptor_names
    assert "complexity_sample_entropy_ch-Pz" in block.descriptor_names


def test_complexity_collect_nonfinite_numerical_issue(signal_data):
    from unittest.mock import patch

    from coco_pipe.descriptors.configs import DescriptorRuntimeConfig

    X, sfreq, ch_names = signal_data
    config = ComplexityDescriptorConfig(
        enabled=True,
        measures=["sample_entropy"],
    )
    extractor = ComplexityDescriptorExtractor(config)
    # Mock backend result to include inf
    with patch.object(
        extractor,
        "_load_antropy",
        return_value=MagicMock(sample_entropy=lambda x, **kwargs: np.inf),
    ):
        block = extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            ids=None,
            runtime=DescriptorRuntimeConfig(on_error="collect"),
        )
        assert np.isnan(block.X).all()
        assert any(f["exception_type"] == "NumericalIssue" for f in block.failures)


def test_complexity_raise_on_error(signal_data):
    from unittest.mock import patch

    from coco_pipe.descriptors.configs import DescriptorRuntimeConfig

    X, sfreq, ch_names = signal_data
    config = ComplexityDescriptorConfig(
        enabled=True,
        measures=["sample_entropy"],
    )
    extractor = ComplexityDescriptorExtractor(config)
    with (
        patch.object(
            extractor,
            "_load_antropy",
            return_value=MagicMock(sample_entropy=lambda x, **kwargs: np.inf),
        ),
        pytest.raises(ValueError, match="produced a non-finite result"),
    ):
        extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            ids=None,
            runtime=DescriptorRuntimeConfig(on_error="raise"),
        )


def test_easy_batch_complexity_measures_emit_expected_columns(signal_data):
    X, sfreq, ch_names = signal_data
    measures = [
        "approx_entropy",
        "svd_entropy",
        "petrosian_fd",
        "katz_fd",
        "higuchi_fd",
        "zero_crossings",
        "kurtosis",
        "rms",
    ]
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="antropy",
            measures=measures,
        )
    )

    block = extractor.extract(
        X,
        sfreq=sfreq,
        channel_names=ch_names,
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )

    assert block.X.shape == (X.shape[0], len(measures) * len(ch_names))
    for measure in measures:
        assert f"complexity_{measure}_ch-Fz" in block.descriptor_names


def test_zero_crossings_matches_manual_count():
    X = np.array([[[-1.0, 1.0, -2.0, 3.0, -4.0]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=["zero_crossings"],
        )
    )

    block = extractor.extract(
        X,
        sfreq=250.0,
        channel_names=["Fz"],
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )

    assert block.descriptor_names == ["complexity_zero_crossings_ch-Fz"]
    assert block.X[0, 0] == pytest.approx(4.0)


def test_rms_matches_manual_calculation():
    X = np.array([[[1.0, -1.0, 1.0, -1.0]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=["rms"],
        )
    )

    block = extractor.extract(
        X,
        sfreq=250.0,
        channel_names=["Fz"],
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )

    assert block.descriptor_names == ["complexity_rms_ch-Fz"]
    assert block.X[0, 0] == pytest.approx(1.0)


def test_kurtosis_matches_scipy_definition():
    signal = np.array([1.0, 2.0, 2.5, 4.0, 8.0], dtype=float)
    X = signal.reshape(1, 1, -1)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=["kurtosis"],
        )
    )

    block = extractor.extract(
        X,
        sfreq=250.0,
        channel_names=["Fz"],
        ids=None,
        runtime=MagicMock(),
        obs_offset=0,
    )

    assert block.descriptor_names == ["complexity_kurtosis_ch-Fz"]
    assert block.X[0, 0] == pytest.approx(
        scipy_kurtosis(signal, fisher=True, bias=False)
    )


@pytest.mark.parametrize(
    ("measure", "backend_attr", "backend_result", "expected"),
    [
        ("shannon_entropy", "entropy_shannon", (1.25, {"base": 2}), 1.25),
        ("fuzzy_entropy", "entropy_fuzzy", (0.75, {"Tolerance": 0.2}), 0.75),
        (
            "dispersion_entropy",
            "entropy_dispersion",
            (0.5, {"dimension": 3}),
            0.5,
        ),
        ("hurst_exponent", "fractal_hurst", np.array([0.62]), 0.62),
    ],
)
def test_medium_batch_neurokit_measures_are_wired(
    measure,
    backend_attr,
    backend_result,
    expected,
):
    from unittest.mock import patch

    X = np.array([[[0.1, 0.2, 0.4, 0.8, 0.3, 0.1]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=[measure],
        )
    )
    fake_nk = MagicMock()
    setattr(
        fake_nk,
        backend_attr,
        lambda signal, _result=backend_result, **kwargs: _result,
    )

    with patch.object(extractor, "_load_neurokit", return_value=fake_nk):
        block = extractor.extract(
            X,
            sfreq=250.0,
            channel_names=["Fz"],
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )

    assert block.descriptor_names == [f"complexity_{measure}_ch-Fz"]
    assert block.X[0, 0] == pytest.approx(expected)


def test_antropy_backend_rejects_medium_batch_measures():
    X = np.array([[[0.1, 0.2, 0.4, 0.8, 0.3, 0.1]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="antropy",
            measures=["fuzzy_entropy"],
        )
    )

    with pytest.raises(ValueError, match="not supported by backend 'antropy'"):
        extractor.extract(
            X,
            sfreq=250.0,
            channel_names=["Fz"],
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )


def test_auto_backend_mixes_antropy_and_neurokit_measures():
    from unittest.mock import patch

    X = np.array([[[0.1, 0.2, 0.4, 0.8, 0.3, 0.1]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="auto",
            measures=["sample_entropy", "fuzzy_entropy"],
        )
    )
    fake_ant = MagicMock(sample_entropy=lambda signal, **kwargs: 0.25)
    fake_nk = MagicMock(entropy_fuzzy=lambda signal, **kwargs: (0.75, {}))

    with (
        patch.object(extractor, "_load_antropy", return_value=fake_ant),
        patch.object(extractor, "_load_neurokit", return_value=fake_nk),
    ):
        block = extractor.extract(
            X,
            sfreq=250.0,
            channel_names=["Fz"],
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )

    assert "complexity_sample_entropy_ch-Fz" in block.descriptor_names
    assert "complexity_fuzzy_entropy_ch-Fz" in block.descriptor_names
    assert block.meta["measure_backends"] == {
        "sample_entropy": "antropy",
        "fuzzy_entropy": "neurokit2",
    }


def test_medium_batch_neurokit_collects_nonfinite_values():
    from unittest.mock import patch

    from coco_pipe.descriptors.configs import DescriptorRuntimeConfig

    X = np.array([[[0.1, 0.2, 0.4, 0.8, 0.3, 0.1]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=["fuzzy_entropy"],
        )
    )
    fake_nk = MagicMock(entropy_fuzzy=lambda signal, **kwargs: (np.inf, {}))

    with patch.object(extractor, "_load_neurokit", return_value=fake_nk):
        block = extractor.extract(
            X,
            sfreq=250.0,
            channel_names=["Fz"],
            ids=None,
            runtime=DescriptorRuntimeConfig(on_error="collect"),
            obs_offset=0,
        )

    assert np.isnan(block.X[0, 0])
    assert any(f["exception_type"] == "NumericalIssue" for f in block.failures)


def test_medium_batch_neurokit_raises_on_nonfinite_values():
    from unittest.mock import patch

    from coco_pipe.descriptors.configs import DescriptorRuntimeConfig

    X = np.array([[[0.1, 0.2, 0.4, 0.8, 0.3, 0.1]]], dtype=float)
    extractor = ComplexityDescriptorExtractor(
        ComplexityDescriptorConfig(
            enabled=True,
            backend="neurokit2",
            measures=["fuzzy_entropy"],
        )
    )
    fake_nk = MagicMock(entropy_fuzzy=lambda signal, **kwargs: (np.inf, {}))

    with (
        patch.object(extractor, "_load_neurokit", return_value=fake_nk),
        pytest.raises(ValueError, match="produced a non-finite result"),
    ):
        extractor.extract(
            X,
            sfreq=250.0,
            channel_names=["Fz"],
            ids=None,
            runtime=DescriptorRuntimeConfig(on_error="raise"),
            obs_offset=0,
        )


def test_lazy_loading_failure_antropy(monkeypatch):
    """Verify informative error when antropy is missing."""
    monkeypatch.setitem(sys.modules, "antropy", None)
    config = ComplexityDescriptorConfig(enabled=True, backend="antropy")
    extractor = ComplexityDescriptorExtractor(config)

    with pytest.raises(ImportError, match="antropy"):
        extractor._load_antropy()


def test_lazy_loading_failure_neurokit2(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", None)
    config = ComplexityDescriptorConfig(enabled=True, backend="neurokit2")
    extractor = ComplexityDescriptorExtractor(config)

    with pytest.raises(ImportError, match="neurokit"):
        extractor._load_neurokit()
