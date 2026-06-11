import builtins
import inspect

import numpy as np
import pytest

import coco_pipe.descriptors.core as descriptors_core
from coco_pipe.descriptors import DescriptorPipeline
from coco_pipe.descriptors.extractors.complexity import ComplexityDescriptorExtractor


def test_empty_pipeline_returns_explicit_result_structure():
    X = np.random.default_rng(0).normal(size=(5, 2, 64))
    pipe = DescriptorPipeline({})
    result = pipe.extract(X=X, sfreq=128.0)

    assert set(result) == {"X", "descriptor_names", "failures"}
    assert result["X"].shape == (5, 0)
    assert result["descriptor_names"] == []


def test_band_pipeline_smoke_sensor_level():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 3, 128))
    pipe = DescriptorPipeline(
        {
            "families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}},
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert result["X"].shape[0] == 6
    assert result["X"].shape[1] == len(result["descriptor_names"])
    assert "band_abs_alpha_ch-Fz" in result["descriptor_names"]
    assert "band_abs_alpha_ch-Cz" in result["descriptor_names"]
    assert not any(
        name.endswith("chgrp-Frontal") for name in result["descriptor_names"]
    )


def test_pool_channels_replaces_sensor_columns_with_grouped_columns():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(4, 3, 128))
    pipe = DescriptorPipeline(
        {
            "families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}},
        }
    )
    result = pipe.extract(
        X=X,
        sfreq=128.0,
        channel_names=["Fz", "Cz", "Pz"],
    )
    pooled = pipe.pool_channels(result, {"Frontal": ["Fz", "Cz"]})

    assert "band_abs_alpha_chgrp-Frontal" in pooled["descriptor_names"]
    assert "band_abs_alpha_ch-Fz" not in pooled["descriptor_names"]
    fz_idx = result["descriptor_names"].index("band_abs_alpha_ch-Fz")
    cz_idx = result["descriptor_names"].index("band_abs_alpha_ch-Cz")
    grp_idx = pooled["descriptor_names"].index("band_abs_alpha_chgrp-Frontal")
    expected = np.nanmean(result["X"][:, [fz_idx, cz_idx]], axis=1)

    assert np.allclose(pooled["X"][:, grp_idx], expected, equal_nan=True)


def test_pool_channels_preserves_non_channel_features():
    pipe = DescriptorPipeline({})
    result = {
        "X": np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=float),
        "descriptor_names": ["global_metric", "toy_mean_ch-Fz", "toy_mean_ch-Cz"],
        "failures": [],
    }

    pooled = pipe.pool_channels(result, {"Frontal": ["Fz", "Cz"]})

    assert pooled["descriptor_names"] == ["global_metric", "toy_mean_chgrp-Frontal"]
    assert np.allclose(pooled["X"][:, 0], result["X"][:, 0], equal_nan=True)
    assert np.allclose(pooled["X"][:, 1], [2.5, 6.0], equal_nan=True)


def test_complexity_can_omit_sfreq_when_config_disables_it():
    X = np.random.default_rng(18).normal(size=(4, 2, 64))
    pipe = DescriptorPipeline(
        {
            "input": {"require_sfreq": False},
            "families": {
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy"],
                }
            },
        }
    )
    result = pipe.extract(X=X, channel_names=["Fz", "Cz"])

    assert result["X"].shape == (4, 2)


def test_output_precision_is_respected():
    X = np.zeros((2, 2, 128), dtype=float)
    pipe = DescriptorPipeline(
        {
            "precision": "float64",
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic"],
                }
            },
            "runtime": {"on_error": "collect"},
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    assert result["X"].dtype == np.float64


def test_missing_sfreq_is_explicit_error():
    X = np.random.default_rng(2).normal(size=(4, 2, 64))
    pipe = DescriptorPipeline(
        {"families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}}}
    )
    with pytest.raises(ValueError, match="`sfreq`"):
        pipe.extract(X=X, channel_names=["Fz", "Cz"])


def test_wrong_ndim_is_rejected():
    X = np.random.default_rng(21).normal(size=(4, 64))
    pipe = DescriptorPipeline({})

    with pytest.raises(ValueError, match="Descriptors expect 3D input"):
        pipe.extract(X=X, sfreq=128.0)


def test_wrong_channel_names_length_is_rejected():
    X = np.random.default_rng(22).normal(size=(4, 2, 64))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="channel_names"):
        pipe.extract(X=X, sfreq=128.0, channel_names=["C3"])


def test_pool_channels_reject_unknown_channel_names():
    X = np.random.default_rng(22).normal(size=(4, 2, 64))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    with pytest.raises(ValueError, match="unknown channel"):
        pipe.pool_channels(result, {"Frontal": ["C3", "C4"]})


def test_pool_channels_reject_overlapping_assignments():
    X = np.random.default_rng(22).normal(size=(4, 3, 64))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    with pytest.raises(ValueError, match="multiple channel_groups"):
        pipe.pool_channels(
            result,
            {
                "Frontal": ["Fz", "Cz"],
                "Central": ["Cz", "Pz"],
            },
        )


@pytest.mark.parametrize(
    "result, groups, match",
    [
        pytest.param(
            {"X": np.zeros((2, 2, 2)), "descriptor_names": ["a", "b"], "failures": []},
            {"G": ["ch1"]},
            "2D",
            id="non_2d_x",
        ),
        pytest.param(
            {"X": np.zeros((2, 1)), "descriptor_names": ["a", "b"], "failures": []},
            {"G": ["ch1"]},
            r"align with result\['X'\]",
            id="mismatched_names_and_columns",
        ),
        pytest.param(
            {
                "X": np.zeros((2, 2)),
                "descriptor_names": ["a_ch-Fz", "b_ch-Cz"],
                "failures": [],
            },
            {},
            "at least one group",
            id="empty_groups",
        ),
        pytest.param(
            {
                "X": np.zeros((2, 2)),
                "descriptor_names": ["a_ch-Fz", "b_ch-Cz"],
                "failures": [],
            },
            {"": ["Fz"]},
            "non-empty strings",
            id="empty_group_name",
        ),
        pytest.param(
            {
                "X": np.zeros((2, 2)),
                "descriptor_names": ["a_ch-Fz", "b_ch-Cz"],
                "failures": [],
            },
            {"G": []},
            "at least one channel",
            id="empty_channel_list",
        ),
        pytest.param(
            {
                "X": np.zeros((2, 2)),
                "descriptor_names": ["a_ch-Fz", "b_ch-Cz"],
                "failures": [],
            },
            {"G": ["Fz", "Fz"]},
            "not contain duplicates",
            id="duplicate_channels",
        ),
        pytest.param(
            {
                "X": np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=float),
                "descriptor_names": [
                    "toy_mean_ch-Fz",
                    "other_mean_ch-Fz",
                    "other_mean_ch-Cz",
                ],
                "failures": [],
            },
            {"Frontal": ["Fz", "Cz"]},
            "could not form group",
            id="incomplete_grouped_feature_base",
        ),
        pytest.param(
            {"X": np.ones((2, 2))},
            {"G1": ["Fz"]},
            "'X', 'descriptor_names', and 'failures'",
            id="missing_result_keys",
        ),
        pytest.param(
            {
                "X": np.ones((2, 1)),
                "descriptor_names": ["global_metric"],
                "failures": [],
            },
            {"G1": ["Fz"]},
            "sensor-level descriptor names",
            id="global_only_names",
        ),
        pytest.param(
            {"X": np.zeros((2, 2)), "descriptor_names": ["a_ch-Fz"], "failures": []},
            {"G": ["Fz"]},
            r"align with result\['X'\] columns",
            id="mismatched_columns",
        ),
    ],
)
def test_pool_channels_rejects(result, groups, match):
    pipe = DescriptorPipeline({})
    with pytest.raises(ValueError, match=match):
        pipe.pool_channels(result, groups)


def test_require_channel_names_flag_is_enforced():
    X = np.random.default_rng(25).normal(size=(4, 2, 64))
    pipe = DescriptorPipeline(
        {
            "input": {"require_channel_names": True},
            "families": {
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy"],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="channel_names"):
        pipe.extract(X=X, sfreq=128.0)


def test_complexity_collects_short_segment_failures():
    X = np.ones((4, 2, 3), dtype=float)
    pipe = DescriptorPipeline(
        {
            "families": {
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy"],
                }
            },
            "runtime": {"on_error": "collect"},
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    assert result["X"].shape == (4, 2)
    assert np.isnan(result["X"]).all()
    assert result["failures"]


def test_bands_collect_short_window_resolution_failures():
    X = np.random.default_rng(7).normal(size=(3, 2, 8))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power", "relative_power"],
                }
            },
            "runtime": {"on_error": "collect"},
        }
    )

    result = pipe.extract(X=X, sfreq=160.0, channel_names=["C3", "C4"])

    assert result["X"].shape == (3, 20)
    assert any(
        failure["exception_type"] == "BandResolutionError"
        for failure in result["failures"]
    )


def test_warn_policy_emits_aggregate_warning():
    X = np.random.default_rng(23).normal(size=(3, 2, 8))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
            "runtime": {"on_error": "warn"},
        }
    )

    with pytest.warns(UserWarning, match="Collected"):
        result = pipe.extract(X=X, sfreq=160.0, channel_names=["C3", "C4"])

    assert result["failures"]


def test_raise_policy_reraises_runtime_failure():
    X = np.random.default_rng(24).normal(size=(3, 2, 8))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
            "runtime": {"on_error": "raise"},
        }
    )

    with pytest.raises(ValueError, match="does not overlap"):
        pipe.extract(X=X, sfreq=160.0, channel_names=["C3", "C4"])


def test_complexity_collects_nonfinite_output_as_nan():
    X = np.ones((2, 2, 16), dtype=float)
    pipe = DescriptorPipeline(
        {
            "families": {
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy"],
                }
            },
            "runtime": {"on_error": "collect"},
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    assert np.isnan(result["X"]).all()
    assert result["failures"]


def test_complexity_raise_policy_reraises_nonfinite_output():
    X = np.ones((2, 2, 16), dtype=float)
    pipe = DescriptorPipeline(
        {
            "families": {
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy"],
                }
            },
            "runtime": {"on_error": "raise"},
        }
    )

    with pytest.raises(ValueError, match="non-finite"):
        pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])


def test_constant_signal_parametric_skip_collects_failures():
    X = np.zeros((3, 2, 128), dtype=float)
    pipe = DescriptorPipeline(
        {
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic"],
                }
            },
            "runtime": {"on_error": "collect"},
        }
    )
    result = pipe.extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    assert result["failures"]
    assert np.isnan(result["X"]).all()


def test_missing_antropy_dependency_has_clear_install_hint(monkeypatch):
    def _raise_import_error(self):
        raise ImportError(
            "antropy is required for complexity descriptor extraction. "
            "Install it with 'pip install coco-pipe[descriptors]'."
        )

    monkeypatch.setattr(
        ComplexityDescriptorExtractor,
        "_load_antropy",
        _raise_import_error,
    )
    pipe = DescriptorPipeline(
        {
            "families": {
                "complexity": {
                    "enabled": True,
                    "backend": "antropy",
                    "measures": ["sample_entropy"],
                }
            },
        }
    )

    with pytest.raises(ImportError, match=r"coco-pipe\[descriptors\]"):
        pipe.extract(X=[[[1.0] * 32]], sfreq=128.0, channel_names=["Cz"])


def test_multi_family_scale_smoke():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 4, 256))
    pipe = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power", "relative_power"],
                },
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic", "peak_summary"],
                },
                "complexity": {
                    "enabled": True,
                    "measures": ["sample_entropy", "perm_entropy"],
                },
            },
            "runtime": {"obs_chunk": 8},
        }
    )
    result = pipe.extract(
        X=X,
        sfreq=256.0,
        channel_names=["Fz", "Cz", "Pz", "Oz"],
    )

    assert result["X"].shape[0] == 24
    assert result["X"].shape[1] == len(result["descriptor_names"])


def test_multi_family_parallel_matches_sequential():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(12, 3, 128))
    channel_names = ["Fz", "Cz", "Pz"]
    base_config = {
        "families": {
            "bands": {
                "enabled": True,
                "outputs": ["absolute_power", "relative_power"],
            },
            "complexity": {
                "enabled": True,
                "measures": ["sample_entropy", "hjorth_mobility"],
            },
        },
    }
    sequential = DescriptorPipeline(
        {
            **base_config,
            "runtime": {"execution_backend": "sequential", "n_jobs": 1},
        }
    ).extract(X=X, sfreq=128.0, channel_names=channel_names)
    parallel = DescriptorPipeline(
        {
            **base_config,
            "runtime": {"execution_backend": "joblib", "n_jobs": 2},
        }
    ).extract(X=X, sfreq=128.0, channel_names=channel_names)

    assert sequential["descriptor_names"] == parallel["descriptor_names"]
    assert np.allclose(sequential["X"], parallel["X"], equal_nan=True)


def test_parametric_parallel_matches_sequential():
    rng = np.random.default_rng(13)
    t = np.linspace(0, 1, 128, endpoint=False)
    X = rng.normal(scale=0.05, size=(6, 3, 128))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += np.sin(2 * np.pi * 6 * t)

    sequential = DescriptorPipeline(
        {
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic", "fit_quality"],
                }
            },
            "runtime": {"execution_backend": "sequential", "n_jobs": 1},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])
    parallel = DescriptorPipeline(
        {
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic", "fit_quality"],
                }
            },
            "runtime": {"execution_backend": "joblib", "n_jobs": 2},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert sequential["descriptor_names"] == parallel["descriptor_names"]
    assert np.allclose(sequential["X"], parallel["X"], equal_nan=True)


def test_multi_chunk_row_order_matches_unchunked():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(18, 3, 128))
    config = {
        "families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}},
    }
    unchunked = DescriptorPipeline(config).extract(
        X=X,
        sfreq=128.0,
        channel_names=["Fz", "Cz", "Pz"],
    )
    chunked = DescriptorPipeline(
        {
            **config,
            "runtime": {"obs_chunk": 5},
        }
    ).extract(
        X=X,
        sfreq=128.0,
        channel_names=["Fz", "Cz", "Pz"],
    )

    assert unchunked["descriptor_names"] == chunked["descriptor_names"]
    assert np.allclose(unchunked["X"], chunked["X"], equal_nan=True)


def test_n_jobs_one_skips_joblib_loading(monkeypatch):
    rng = np.random.default_rng(16)
    X = rng.normal(size=(4, 2, 64))
    real_import = builtins.__import__
    joblib_imports = 0

    def _count_joblib_imports(name, *args, **kwargs):
        nonlocal joblib_imports
        caller = inspect.currentframe().f_back
        caller_name = None if caller is None else caller.f_globals.get("__name__")
        if name == "joblib" and caller_name in {
            "coco_pipe.descriptors.core",
            "coco_pipe.descriptors.extractors._parametric_fit",
        }:
            joblib_imports += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _count_joblib_imports)
    result = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
            "runtime": {"execution_backend": "joblib", "n_jobs": 1},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz"])

    assert result["X"].shape == (4, len(result["descriptor_names"]))
    assert joblib_imports == 0


def test_parametric_parallel_n_jobs_all_cores_smoke():
    rng = np.random.default_rng(17)
    t = np.linspace(0, 4, 512, endpoint=False)
    X = rng.normal(scale=0.05, size=(4, 3, 512))
    freqs = np.fft.rfftfreq(512, 1 / 128.0)
    weights = 1 / (freqs + 1.0)
    for obs_idx in range(4):
        for ch_idx in range(3):
            X[obs_idx, ch_idx, :] = np.fft.irfft(
                np.fft.rfft(X[obs_idx, ch_idx, :]) * weights,
                n=512,
            )

    X[:, 0, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += 1.5 * np.sin(2 * np.pi * 16 * t)
    X[:, 2, :] += 1.0 * np.sin(2 * np.pi * 6 * t)

    result = DescriptorPipeline(
        {
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic"],
                }
            },
            "runtime": {"execution_backend": "joblib", "n_jobs": -1},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert result["X"].shape == (4, 6)


def test_shared_psd_reuses_one_compute_per_batch_for_same_method(monkeypatch):
    rng = np.random.default_rng(19)
    t = np.linspace(0, 1, 128, endpoint=False)
    X = rng.normal(scale=0.05, size=(8, 3, 128))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += np.sin(2 * np.pi * 6 * t)

    real_compute_psd = descriptors_core.compute_psd
    calls: list[tuple[str, float, float]] = []

    def _counted_compute_psd(*args, **kwargs):
        calls.append((kwargs["method"], kwargs["fmin"], kwargs["fmax"]))
        return real_compute_psd(*args, **kwargs)

    monkeypatch.setattr(descriptors_core, "compute_psd", _counted_compute_psd)

    DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "psd_method": "welch",
                    "outputs": ["absolute_power"],
                },
                "parametric": {
                    "enabled": True,
                    "psd_method": "welch",
                    "outputs": ["aperiodic"],
                },
            },
            "runtime": {"obs_chunk": 4},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert len(calls) == 2
    assert all(method == "welch" for method, _, _ in calls)


def test_corrected_bands_and_parametric_share_one_fit_batch_per_psd_group(monkeypatch):
    rng = np.random.default_rng(191)
    t = np.linspace(0, 1, 128, endpoint=False)
    X = rng.normal(scale=0.05, size=(8, 3, 128))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += np.sin(2 * np.pi * 6 * t)

    real_fit_batch = descriptors_core.fit_parametric_batch
    calls = 0

    def _counted_fit_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fit_batch(*args, **kwargs)

    monkeypatch.setattr(descriptors_core, "fit_parametric_batch", _counted_fit_batch)

    result = DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "psd_method": "welch",
                    "outputs": ["corrected_absolute_power"],
                },
                "parametric": {
                    "enabled": True,
                    "psd_method": "welch",
                    "outputs": ["aperiodic"],
                },
            },
            "runtime": {"obs_chunk": 4},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert calls == 2
    assert "band_corr_abs_alpha_ch-Fz" in result["descriptor_names"]


def test_shared_psd_splits_groups_by_method(monkeypatch):
    rng = np.random.default_rng(20)
    t = np.linspace(0, 1, 128, endpoint=False)
    X = rng.normal(scale=0.05, size=(8, 3, 128))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += np.sin(2 * np.pi * 6 * t)

    real_compute_psd = descriptors_core.compute_psd
    calls: list[str] = []

    def _counted_compute_psd(*args, **kwargs):
        calls.append(kwargs["method"])
        return real_compute_psd(*args, **kwargs)

    monkeypatch.setattr(descriptors_core, "compute_psd", _counted_compute_psd)

    DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "psd_method": "welch",
                    "outputs": ["absolute_power"],
                },
                "parametric": {
                    "enabled": True,
                    "psd_method": "multitaper",
                    "outputs": ["aperiodic"],
                },
            },
            "runtime": {"obs_chunk": 4},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert calls == ["welch", "multitaper", "welch", "multitaper"]


def test_shared_union_psd_matches_separate_family_outputs():
    rng = np.random.default_rng(21)
    t = np.linspace(0, 1, 128, endpoint=False)
    X = rng.normal(scale=0.05, size=(6, 3, 128))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += np.sin(2 * np.pi * 6 * t)
    channel_names = ["Fz", "Cz", "Pz"]

    bands_cfg = {
        "families": {
            "bands": {
                "enabled": True,
                "psd_method": "welch",
                "fmin": 1.0,
                "fmax": 30.0,
                "bands": {
                    "delta": [1.0, 4.0],
                    "theta": [4.0, 8.0],
                    "alpha": [8.0, 13.0],
                    "beta": [13.0, 30.0],
                },
                "outputs": ["absolute_power", "relative_power"],
            }
        },
    }
    param_cfg = {
        "families": {
            "parametric": {
                "enabled": True,
                "psd_method": "welch",
                "freq_range": [1.0, 45.0],
                "outputs": ["aperiodic", "fit_quality"],
            }
        },
    }
    combined_cfg = {
        "families": {
            **bands_cfg["families"],
            **param_cfg["families"],
        },
    }

    bands_only = DescriptorPipeline(bands_cfg).extract(
        X=X,
        sfreq=128.0,
        channel_names=channel_names,
    )
    param_only = DescriptorPipeline(param_cfg).extract(
        X=X,
        sfreq=128.0,
        channel_names=channel_names,
    )
    combined = DescriptorPipeline(combined_cfg).extract(
        X=X,
        sfreq=128.0,
        channel_names=channel_names,
    )

    band_names = [
        name for name in combined["descriptor_names"] if name.startswith("band_")
    ]
    param_names = [
        name for name in combined["descriptor_names"] if name.startswith("param_")
    ]
    band_indices = [combined["descriptor_names"].index(name) for name in band_names]
    param_indices = [combined["descriptor_names"].index(name) for name in param_names]

    assert band_names == bands_only["descriptor_names"]
    assert param_names == param_only["descriptor_names"]
    assert np.allclose(
        combined["X"][:, band_indices],
        bands_only["X"],
        equal_nan=True,
    )
    assert np.allclose(
        combined["X"][:, param_indices],
        param_only["X"],
        equal_nan=True,
    )


def test_obs_batch_parallel_disables_parametric_inner_joblib(monkeypatch):
    rng = np.random.default_rng(22)
    t = np.linspace(0, 4, 512, endpoint=False)
    X = rng.normal(scale=0.05, size=(6, 3, 512))
    freqs = np.fft.rfftfreq(512, 1 / 128.0)
    weights = 1 / (freqs + 1.0)
    for obs_idx in range(6):
        for ch_idx in range(3):
            X[obs_idx, ch_idx, :] = np.fft.irfft(
                np.fft.rfft(X[obs_idx, ch_idx, :]) * weights,
                n=512,
            )

    X[:, 0, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += 1.5 * np.sin(2 * np.pi * 18 * t)
    X[:, 2, :] += 1.0 * np.sin(2 * np.pi * 6 * t)
    real_import = builtins.__import__
    joblib_imports = 0

    def _count_joblib_imports(name, *args, **kwargs):
        nonlocal joblib_imports
        caller = inspect.currentframe().f_back
        caller_name = None if caller is None else caller.f_globals.get("__name__")
        if name == "joblib" and caller_name in {
            "coco_pipe.descriptors.core",
            "coco_pipe.descriptors.extractors._parametric_fit",
        }:
            joblib_imports += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _count_joblib_imports)
    result = DescriptorPipeline(
        {
            "families": {
                "parametric": {
                    "enabled": True,
                    "outputs": ["aperiodic"],
                }
            },
            "runtime": {
                "execution_backend": "joblib",
                "n_jobs": 2,
                "obs_chunk": 2,
            },
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    exponent_indices = [
        idx for idx, name in enumerate(result["descriptor_names"]) if "exponent" in name
    ]
    offset_indices = [
        idx for idx, name in enumerate(result["descriptor_names"]) if "offset" in name
    ]

    assert exponent_indices
    assert offset_indices
    assert np.all(result["X"][:, exponent_indices] > 0)
    assert np.all(np.isfinite(result["X"][:, offset_indices]))
    assert joblib_imports >= 1


def test_single_psd_group_uses_psd_level_n_jobs(monkeypatch):
    rng = np.random.default_rng(23)
    X = rng.normal(size=(4, 3, 128))
    calls: list[int | None] = []
    real_compute_psd = descriptors_core.compute_psd

    def _counted_compute_psd(*args, **kwargs):
        calls.append(kwargs["n_jobs"])
        return real_compute_psd(*args, **kwargs)

    monkeypatch.setattr(descriptors_core, "compute_psd", _counted_compute_psd)
    DescriptorPipeline(
        {
            "families": {
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power"],
                }
            },
            "runtime": {"execution_backend": "joblib", "n_jobs": 2},
        }
    ).extract(X=X, sfreq=128.0, channel_names=["Fz", "Cz", "Pz"])

    assert calls == [2]


def test_validation_edge_cases_runtime():
    from coco_pipe.descriptors.configs import DescriptorConfig
    from coco_pipe.descriptors.validation import validate_runtime_inputs

    config = DescriptorConfig(families={"bands": {"enabled": True}})
    X = np.zeros((2, 2, 64))

    with pytest.raises(ValueError, match="`sfreq` must be positive"):
        validate_runtime_inputs(config, X=X, sfreq=0, channel_names=["ch1", "ch2"])

    with pytest.raises(ValueError, match="`ids` must align with n_obs=2"):
        validate_runtime_inputs(
            config, X=X, sfreq=100.0, ids=[1, 2, 3], channel_names=["ch1", "ch2"]
        )

    with pytest.raises(
        ValueError, match="`channel_names` must align with n_channels=2"
    ):
        validate_runtime_inputs(config, X=X, sfreq=100.0, channel_names=["ch1"])

    with pytest.raises(ValueError, match="`channel_names` must be passed explicitly"):
        validate_runtime_inputs(config, X=X, sfreq=100.0, channel_names=None)


def test_pool_channels_handles_mixture_of_sensor_and_global_features():
    pipe = DescriptorPipeline({})
    result = {
        "X": np.array([[1.0, 10.0, 20.0], [2.0, 30.0, 40.0]], dtype=float),
        "descriptor_names": ["global", "val_ch-Fz", "val_ch-Cz"],
        "failures": [],
    }
    # Pool Fz, Cz -> 15, 35
    # Result: global=1,2, grouped=15,35
    pooled = pipe.pool_channels(result, {"Group": ["Fz", "Cz"]})
    assert pooled["descriptor_names"] == ["global", "val_chgrp-Group"]
    assert np.allclose(pooled["X"][:, 0], [1.0, 2.0])
    assert np.allclose(pooled["X"][:, 1], [15.0, 35.0])


def test_pool_channels_preserves_failures():
    pipe = DescriptorPipeline({})
    result = {
        "X": np.zeros((2, 2)),
        "descriptor_names": ["a_ch-Fz", "a_ch-Cz"],
        "failures": [{"family": "toy", "message": "boom"}],
    }
    pooled = pipe.pool_channels(result, {"G": ["Fz"]})
    assert pooled["failures"] == result["failures"]


def test_pipeline_instantiation_validates_fit_range_coverage():
    config = {
        "families": {
            "bands": {
                "enabled": True,
                "fmin": 1.0,
                "fmax": 45.0,
                "outputs": ["corrected_absolute_power"],
            },
            "parametric": {
                "enabled": True,
                "freq_range": [2.0, 50.0],  # 2.0 > 1.0, bad
            },
        }
    }
    with pytest.raises(ValueError, match="cover the band PSD window"):
        DescriptorPipeline(config)


def test_pipeline_precision_is_propagated_to_pooled_output():
    pipe = DescriptorPipeline({"precision": "float32"})
    result = {
        "X": np.array([[1.0, 2.0]], dtype=np.float64),
        "descriptor_names": ["a_ch-Fz", "a_ch-Cz"],
        "failures": [],
    }
    pooled = pipe.pool_channels(result, {"G": ["Fz", "Cz"]})
    assert pooled["X"].dtype == np.float32


def test_empty_work_unit_parallel_smoke():
    # 0 signal extractors, 1 PSD group (1 consumer) -> sequential
    pipe = DescriptorPipeline(
        {
            "families": {"bands": {"enabled": True}},
            "runtime": {"n_jobs": 2, "execution_backend": "joblib"},
        }
    )
    X = np.zeros((2, 1, 64))
    result = pipe.extract(X, sfreq=100.0, channel_names=["ch1"])
    assert result["X"].shape[0] == 2
