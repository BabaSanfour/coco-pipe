import tempfile
import warnings
from pathlib import Path

import pandas as pd

from coco_pipe.report.descriptor_qc import (
    generate_descriptor_dataset_report,
    generate_descriptor_subject_report,
)


def test_generate_descriptor_subject_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sub_report.html"

        overview_df = pd.DataFrame(
            [
                {
                    "Subject": "sub-01",
                    "Session": "ses-01",
                    "Condition": "rest",
                }
            ]
        )
        flags_df = pd.DataFrame(
            [
                {
                    "level": "warning",
                    "scope": "subject",
                    "code": "missing_data",
                }
            ]
        )
        failure_summary_df = pd.DataFrame(
            [
                {
                    "group": "param",
                    "value": "FOOOF",
                    "count": 2,
                }
            ]
        )
        feature_missingness_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "scope": "sensor",
                    "sensor": "Fz",
                    "missing_rate": 0.1,
                }
            ]
        )
        family_summary_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "missing_rate": 0.05,
                    "band_abs_negative_rate": 0.0,
                    "band_rel_out_of_range_rate": 0.0,
                    "band_corr_rel_out_of_range_rate": 0.0,
                    "band_ratio_nan_rate": 0.0,
                },
                {
                    "family": "param",
                    "missing_rate": 0.1,
                    "param_r_squared_median": 0.95,
                    "param_r_squared_p05": 0.85,
                    "param_fit_error_median": 0.02,
                    "param_fit_error_p95": 0.05,
                    "param_peak_count_missing_rate": 0.0,
                    "param_alpha_peak_freq_missing_rate": 0.0,
                },
                {
                    "family": "complexity",
                    "missing_rate": 0.0,
                    "missing_rate_max": 0.0,
                    "nonfinite_rate": 0.0,
                },
            ]
        )

        # Write dummy images to avoid missing file issues in _add_images
        fig_dir = Path(tmpdir) / "figs"
        fig_dir.mkdir()
        fig_paths = {}
        for key in [
            "family_missingness",
            "failure_counts_by_family",
            "top_missing_features",
            "param_r_squared_hist",
            "param_fit_error_hist",
        ]:
            img_path = fig_dir / f"{key}.png"
            img_path.write_text("dummy image")
            fig_paths[key] = img_path

        res = generate_descriptor_subject_report(
            out_path,
            overview_df,
            flags_df,
            failure_summary_df,
            feature_missingness_df,
            family_summary_df,
            fig_paths,
        )
        assert res.exists()
        assert out_path.exists()


def test_generate_descriptor_dataset_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "dataset_report.html"

        overview_df = pd.DataFrame([{"n_subjects": 10, "n_sessions": 20}])
        shard_summary_df = pd.DataFrame(
            [
                {
                    "session": "ses-01",
                    "condition": "rest",
                    "qc_status": "pass",
                }
            ]
        )
        flags_df = pd.DataFrame(
            [
                {
                    "level": "warning",
                    "scope": "dataset",
                    "code": "variance_too_low",
                }
            ]
        )
        failure_family_df = pd.DataFrame(
            [
                {
                    "value": "band",
                    "count": 5,
                }
            ]
        )
        failure_channel_df = pd.DataFrame(
            [
                {
                    "value": "Fz",
                    "count": 3,
                }
            ]
        )
        feature_missingness_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "scope": "sensor",
                    "sensor": "Fz",
                    "missing_rate": 0.02,
                }
            ]
        )
        low_variance_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "std": 0.0001,
                }
            ]
        )
        family_summary_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "missing_rate": 0.05,
                }
            ]
        )
        manifest_df = pd.DataFrame([{"merged_at": "2026-06-11"}])
        condition_breakdown_df = pd.DataFrame(
            [
                {
                    "family": "band",
                    "condition": "rest",
                    "n_failures": 0,
                }
            ]
        )

        fig_dir = Path(tmpdir) / "figs"
        fig_dir.mkdir()
        fig_paths = {}
        for key in [
            "shard_status_counts",
            "failure_counts_by_family",
            "failure_counts_by_channel",
            "top_missing_features",
            "low_variance_by_family",
        ]:
            img_path = fig_dir / f"{key}.png"
            img_path.write_text("dummy image")
            fig_paths[key] = img_path

        res = generate_descriptor_dataset_report(
            out_path,
            overview_df,
            shard_summary_df,
            flags_df,
            failure_family_df,
            failure_channel_df,
            feature_missingness_df,
            low_variance_df,
            family_summary_df,
            fig_paths,
            manifest_df=manifest_df,
            condition_breakdown_df=condition_breakdown_df,
        )
        assert res.exists()
        assert out_path.exists()


def test_generate_descriptor_subject_report_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sub_report_edge.html"

        overview_df = pd.DataFrame(
            [{"Subject": "sub-01", "Session": "ses-01", "Condition": "rest"}]
        )
        flags_df = pd.DataFrame()
        failure_summary_df = pd.DataFrame()
        feature_missingness_df = pd.DataFrame()

        # 1. family_summary_df is None
        res = generate_descriptor_subject_report(
            out_path,
            overview_df,
            flags_df,
            failure_summary_df,
            feature_missingness_df,
            None,
            {},
        )
        assert res.exists()

        # 2. family_summary_df lacks 'family' column
        family_summary_df_no_col = pd.DataFrame([{"some_column": 1}])
        res = generate_descriptor_subject_report(
            out_path,
            overview_df,
            flags_df,
            failure_summary_df,
            feature_missingness_df,
            family_summary_df_no_col,
            {},
        )
        assert res.exists()

        # 3. family_summary_df has 'family' but it doesn't match any known
        # family (empty rows for known families)
        family_summary_df_empty_family = pd.DataFrame([{"family": "unknown_family"}])
        res = generate_descriptor_subject_report(
            out_path,
            overview_df,
            flags_df,
            failure_summary_df,
            feature_missingness_df,
            family_summary_df_empty_family,
            {},
        )
        assert res.exists()

        # 4. family_summary_df has all NaN diagnostics
        family_summary_df_nan = pd.DataFrame(
            [
                {
                    "family": "band",
                    "band_abs_negative_rate": float("nan"),
                }
            ]
        )
        res = generate_descriptor_subject_report(
            out_path,
            overview_df,
            flags_df,
            failure_summary_df,
            feature_missingness_df,
            family_summary_df_nan,
            {},
        )
        assert res.exists()


def test_generate_descriptor_subject_report_asset_urls_inline_is_self_contained():
    """``asset_urls='inline'`` must embed assets and suppress the CDN warning."""
    overview_df = pd.DataFrame(
        [{"Subject": "sub-01", "Session": "ses-01", "Condition": "rest"}]
    )
    empty = pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmpdir:
        cdn_path = Path(tmpdir) / "cdn.html"
        inline_path = Path(tmpdir) / "inline.html"

        def _build(out_path, **kwargs):
            return generate_descriptor_subject_report(
                out_path,
                overview_df,
                empty,
                empty,
                empty,
                None,
                {},
                **kwargs,
            )

        # Default path warns about external CDN references.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build(cdn_path)
        assert any("external CDN" in str(w.message) for w in caught)

        # Inline path is self-contained: no CDN warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build(inline_path, asset_urls="inline")
        assert not any("external CDN" in str(w.message) for w in caught)
        assert inline_path.exists()
