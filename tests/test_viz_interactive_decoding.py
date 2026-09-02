import pandas as pd
import plotly.graph_objects as go

from coco_pipe.viz import interactive as iviz
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)


def test_all_interactive_decoding_plots_return_figure():
    result = make_synthetic_result()
    pd.DataFrame(make_synthetic_feature_metadata())

    calls = [
        lambda: iviz.plot_confusion_matrix(result),
        lambda: iviz.plot_roc_curve(result),
        lambda: iviz.plot_pr_curve(result),
        lambda: iviz.plot_calibration_curve(result),
        lambda: iviz.plot_fold_score_dispersion(result),
        lambda: iviz.plot_temporal_score_curve(result),
        lambda: iviz.plot_temporal_generalization_matrix(
            result, model="model_1", metric="generalization_accuracy"
        ),
        lambda: iviz.plot_temporal_statistical_assessment(
            result, model="model_1", metric="temporal_accuracy"
        ),
        lambda: iviz.plot_null_interval_summary(result),
        lambda: iviz.plot_training_history(result),
        lambda: iviz.plot_decoding_scores(result),
        lambda: iviz.plot_model_comparison(result),
        lambda: iviz.plot_fit_diagnostics(result),
        lambda: iviz.plot_probability_diagnostics(result),
        lambda: iviz.plot_subject_diagnostics(result),
        lambda: iviz.plot_group_summary(result),
        lambda: iviz.plot_search_results(result),
        lambda: iviz.plot_feature_stability(result),
        lambda: iviz.plot_feature_scores(result),
        lambda: iviz.plot_feature_importance(result),
    ]

    for call in calls:
        fig = call()
        assert isinstance(fig, go.Figure)


def test_interactive_feature_importance_signed_colors_by_sign():
    df = pd.DataFrame(
        {
            "Model": ["SVM"] * 3,
            "FeatureName": ["f1", "f2", "f3"],
            "Mean": [0.5, -0.3, 0.1],
        }
    )
    fig = iviz.plot_feature_importance(df, signed=True)
    # signed=True paints a per-bar color list (diverging by sign)
    assert isinstance(fig.data[0].marker.color, (list, tuple))
    # single color when not signed
    plain = iviz.plot_feature_importance(df)
    assert isinstance(plain.data[0].marker.color, str)


def test_interactive_feature_importance_mapping_and_absolute():
    fig = iviz.plot_feature_importance({"a": -0.9, "b": 0.2}, absolute=True)
    assert isinstance(fig, go.Figure)
    # absolute ranks by magnitude, so the strongest feature is present
    assert "a" in fig.data[0].y


def test_interactive_feature_importance_dataframe_missing_columns_raises():
    import pytest

    with pytest.raises(ValueError):
        iviz.plot_feature_importance(pd.DataFrame({"Other": [1, 2]}))


def test_interactive_feature_importance_delegates_non_numeric():
    payload = [
        {
            "Method": "PCA",
            "Feature": "x",
            "Dimension": "PC1",
            "Analysis": "loadings",
            "Value": 0.5,
        },
        {
            "Method": "PCA",
            "Feature": "y",
            "Dimension": "PC1",
            "Analysis": "loadings",
            "Value": 0.2,
        },
    ]
    fig = iviz.plot_feature_importance(payload)
    assert isinstance(fig, go.Figure)
