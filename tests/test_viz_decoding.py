import matplotlib.pyplot as plt
import pandas as pd
import pytest

from coco_pipe.viz import decoding as viz
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)


def _assert_fig_ax(out, figsize=None):
    assert isinstance(out, tuple)
    assert len(out) == 2
    fig, ax = out
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    if figsize is not None:
        width, height = fig.get_size_inches()
        assert round(width, 1) == figsize[0]
        assert round(height, 1) == figsize[1]
    plt.close(fig)


def test_all_decoding_plots_return_fig_ax_and_respect_figsize():
    result = make_synthetic_result()
    metadata = pd.DataFrame(make_synthetic_feature_metadata())
    coords = metadata.drop_duplicates("Sensor").set_index("Sensor")[["x", "y"]]
    sensor_values = pd.DataFrame(
        {
            "FeatureName": coords.index,
            "Importance": [0.1, 0.2, 0.3, 0.4][: len(coords)],
        }
    )

    calls = [
        lambda: viz.plot_confusion_matrix(result, figsize=(3, 3)),
        lambda: viz.plot_roc_curve(result, figsize=(3, 3)),
        lambda: viz.plot_pr_curve(result, figsize=(3, 3)),
        lambda: viz.plot_calibration_curve(result, figsize=(3, 3)),
        lambda: viz.plot_fold_score_dispersion(result, figsize=(3, 3)),
        lambda: viz.plot_temporal_score_curve(result, figsize=(3, 3)),
        lambda: viz.plot_temporal_generalization_matrix(
            result, model="model_1", metric="generalization_accuracy", figsize=(3, 3)
        ),
        lambda: viz.plot_temporal_statistical_assessment(
            result, model="model_1", metric="temporal_accuracy", figsize=(3, 3)
        ),
        lambda: viz.plot_null_interval_summary(result, figsize=(3, 3)),
        lambda: viz.plot_training_history(result, figsize=(3, 3)),
        lambda: viz.plot_decoding_scores(result, figsize=(3, 3)),
        lambda: viz.plot_model_comparison(result, figsize=(3, 3)),
        lambda: viz.plot_fit_diagnostics(result, figsize=(3, 3)),
        lambda: viz.plot_probability_diagnostics(result, figsize=(3, 3)),
        lambda: viz.plot_subject_diagnostics(result, figsize=(3, 3)),
        lambda: viz.plot_group_summary(result, figsize=(3, 3)),
        lambda: viz.plot_search_results(result, figsize=(3, 3)),
        lambda: viz.plot_feature_importance(result, figsize=(3, 3)),
        lambda: viz.plot_feature_stability(result, figsize=(3, 3)),
        lambda: viz.plot_feature_scores(result, figsize=(3, 3)),
        lambda: viz.plot_sensor_feature_heatmap(
            result, feature_metadata=metadata, figsize=(3, 3)
        ),
        lambda: viz.plot_sensor_feature_profile(
            result, feature_metadata=metadata, sensor="E1", figsize=(3, 3)
        ),
    ]
    for call in calls:
        _assert_fig_ax(call(), figsize=(3, 3))
    pytest.importorskip("mne")
    _assert_fig_ax(
        viz.plot_decoding_topomap(
            sensor_values, value="Importance", coords=coords, figsize=(3, 3)
        ),
        figsize=(3, 3),
    )
    _assert_fig_ax(
        viz.plot_feature_sensor_profile(
            result,
            feature_metadata=metadata,
            feature_family="spectral",
            coords=coords,
            figsize=(3, 3),
        ),
        figsize=(3, 3),
    )


def test_decoding_plots_raise_on_missing_data():
    with pytest.raises(ValueError):
        viz.plot_confusion_matrix(pd.DataFrame(columns=["Model"]))
    with pytest.raises(ValueError, match="Subject"):
        viz.plot_subject_diagnostics(
            pd.DataFrame({"Model": ["m"], "y_true": [1], "y_pred": [1]})
        )
    with pytest.raises(ValueError, match="requires either info or coords"):
        viz.plot_decoding_topomap(
            pd.DataFrame({"FeatureName": ["E1"], "Value": [1.0]}), value="Value"
        )
    with pytest.raises(ValueError, match="Sensor"):
        viz.plot_sensor_feature_heatmap(
            make_synthetic_result(),
            feature_metadata=pd.DataFrame({"FeatureName": ["F1"]}),
        )


def test_temporal_single_panel_plots_require_single_model_metric():
    result = make_synthetic_result()
    with pytest.raises(ValueError, match="requires a single model/metric selection"):
        viz.plot_temporal_generalization_matrix(result)
    with pytest.raises(ValueError, match="requires a single model/metric selection"):
        viz.plot_temporal_statistical_assessment(result)

    _assert_fig_ax(
        viz.plot_temporal_generalization_matrix(
            result, model="model_1", metric="generalization_accuracy"
        )
    )
    _assert_fig_ax(
        viz.plot_temporal_statistical_assessment(
            result, model="model_1", metric="temporal_accuracy"
        )
    )


def test_mean_curve_paths_accept_dataframes():
    curve_df = pd.DataFrame(
        {
            "Model": ["LR"] * 4,
            "Fold": [0, 0, 1, 1],
            "FPR": [0, 1, 0, 1],
            "TPR": [0, 1, 0, 1],
            "Recall": [0, 1, 0, 1],
            "Precision": [1, 0, 1, 0],
            "Class": [0, 0, 0, 0],
        }
    )
    _assert_fig_ax(viz.plot_roc_curve(curve_df, mean_only=True))
    _assert_fig_ax(viz.plot_pr_curve(curve_df, mean_only=True))

    calibration_df = pd.DataFrame(
        {
            "Model": ["LR"] * 4,
            "Fold": [0, 0, 1, 1],
            "MeanPredictedProbability": [0.1, 0.9, 0.2, 0.8],
            "FractionPositive": [0.0, 1.0, 0.1, 0.9],
            "Class": [0, 0, 0, 0],
        }
    )
    _assert_fig_ax(viz.plot_calibration_curve(calibration_df, mean_only=True))
