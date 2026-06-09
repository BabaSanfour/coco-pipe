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
    ]

    for call in calls:
        fig = call()
        assert isinstance(fig, go.Figure)
