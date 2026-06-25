import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from coco_pipe.viz._utils import (
    _auto_select_single_model,
    _coerce_series,
    _importance_with_metadata,
    _moving_average_1d,
    _prediction_accuracy,
    _records_from_interpretation_payload,
    _scalar_decoding_scores,
    _scalar_metrics,
    _single_method,
    _single_model_metric,
    coerce_decoding_frame,
    coerce_reduction_frame,
    coerce_sensor_layout,
    filter_metric_frame,
    finalize_axes,
    get_figure,
    info_from_montage,
    prepare_component_loadings_frame,
    prepare_confusion_matrix,
    prepare_curve_group_data,
    prepare_decoding_curve_frame,
    prepare_decoding_score_data,
    prepare_eigenvalue_curves,
    prepare_embedding_frame,
    prepare_feature_score_series,
    prepare_feature_scores,
    prepare_feature_stability_series,
    prepare_fit_diagnostics_frame,
    prepare_fold_score_data,
    prepare_interpretation_frame,
    prepare_loss_history,
    prepare_metrics_frame,
    prepare_model_comparison_frame,
    prepare_null_interval_frame,
    prepare_prediction_accuracy_scores,
    prepare_probability_diagnostics_summary,
    prepare_regression_prediction_data,
    prepare_search_results_frame,
    prepare_shepard_distances,
    prepare_streamline_grid,
    prepare_streamline_inputs,
    prepare_temporal_generalization_matrix,
    prepare_temporal_score_curve_frame,
    prepare_temporal_statistical_frame,
    prepare_training_history_artifacts,
    prepare_trajectory_data,
    prepare_trajectory_metric_series,
    prepare_trajectory_separation_series,
    require_columns,
    require_non_empty,
    select_dimensions,
    select_reduction_rows,
    select_rows,
)


def test_moving_average_1d():
    arr = np.array([1, 2, 3, 4, 5])
    assert len(_moving_average_1d(arr, 3)) == 3
    assert _moving_average_1d(arr, 1).tolist() == arr.tolist()

    with pytest.raises(ValueError):
        _moving_average_1d(np.array([[1, 2], [3, 4]]), 2)

    with pytest.raises(ValueError):
        _moving_average_1d(arr, 0)

    with pytest.raises(ValueError):
        _moving_average_1d(arr, 10)


def test_finalize_axes():
    fig, ax = plt.subplots()
    finalize_axes(
        ax,
        title="T",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        legend=True,
        legend_title="L",
        xtick_rotation=90,
        xtick_ha="right",
        tick_nbins=5,
        title_fontsize=12,
        title_fontweight="bold",
        title_pad=10,
        label_fontsize=10,
        label_pad=5,
        tick_labelsize=8,
        tick_length=4,
        grid=True,
        show_spines=["bottom", "left"],
    )
    assert ax.get_title() == "T"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)

    fig, ax = plt.subplots()
    finalize_axes(ax, show_spines="top")
    plt.close(fig)


def test_select_dimensions():
    data = np.random.randn(10, 4)
    res = select_dimensions(data, [0, 2])
    assert res.shape == (10, 2)

    with pytest.raises(ValueError):
        select_dimensions(data, [0])  # Not 2 or 3 dims

    with pytest.raises(ValueError):
        select_dimensions(np.random.randn(10), [0, 1])  # Not 2D

    with pytest.raises(ValueError):
        select_dimensions(data, [0, 5])  # Invalid cols


def test_get_figure():
    fig, ax = plt.subplots()
    res_fig, _res_ax = get_figure(ax, None, (5, 5))
    assert res_fig is fig

    f2, _a2 = get_figure(None, (4, 4), (5, 5))
    assert f2.get_figwidth() == 4

    _f3, a3 = get_figure(None, None, (5, 5), projection="polar")
    assert a3.name == "polar"


def test_coerce_decoding_frame():
    df = pd.DataFrame({"A": [1]})
    assert coerce_decoding_frame(df).equals(df)

    class Dummy:
        def get_df(self):
            return df

    assert coerce_decoding_frame(Dummy(), "get_df").equals(df)

    with pytest.raises(TypeError):
        coerce_decoding_frame(Dummy(), "missing")

    with pytest.raises(TypeError):
        coerce_decoding_frame(Dummy(), None)

    class BadDummy:
        def get_df(self):
            return "not a df"

    with pytest.raises(TypeError):
        coerce_decoding_frame(BadDummy(), "get_df")


def test_coerce_reduction_frame():
    df = pd.DataFrame({"A": [1]})
    assert coerce_reduction_frame(df).equals(df)

    class Dummy:
        def get_df(self):
            return df

    assert coerce_reduction_frame(Dummy(), "get_df").equals(df)

    with pytest.raises(TypeError):
        coerce_reduction_frame(Dummy(), "missing")

    class BadDummy:
        def get_df(self):
            return "not a df"

    with pytest.raises(TypeError):
        coerce_reduction_frame(BadDummy(), "get_df")


def test_require_columns():
    df = pd.DataFrame({"A": [1]})
    require_columns(df, ["A"], "ctx")
    with pytest.raises(ValueError):
        require_columns(df, ["B"], "ctx")


def test_require_non_empty():
    df = pd.DataFrame({"A": [1]})
    require_non_empty(df, "ctx")
    with pytest.raises(ValueError):
        require_non_empty(pd.DataFrame(), "ctx")


def test_scalar_metrics():
    df = pd.DataFrame(
        {"Method": ["M"], "Metric": ["A"], "Value": [1.0], "Scope": ["global"]}
    )
    res = _scalar_metrics(df, "ctx")
    assert not res.empty

    df2 = pd.DataFrame({"Method": ["M"], "Metric": ["A"], "Value": [1.0]})
    res2 = _scalar_metrics(df2, "ctx")
    assert not res2.empty


def test_single_method():
    df = pd.DataFrame({"Method": ["M1"]})
    res = _single_method(df, "ctx")
    assert not res.empty

    with pytest.raises(ValueError):
        _single_method(pd.DataFrame({"Method": ["M1", "M2"]}), "ctx")

    assert _single_method(pd.DataFrame({"A": [1]}), "ctx").equals(
        pd.DataFrame({"A": [1]})
    )


def test_select_rows():
    df = pd.DataFrame({"Model": ["A", "B"], "Metric": ["X", "Y"]})
    res = select_rows(df, model="A")
    assert len(res) == 1


def test_select_reduction_rows():
    df = pd.DataFrame({"Method": ["A", "B"], "Metric": ["X", "Y"]})
    res = select_reduction_rows(df, method="A")
    assert len(res) == 1


def test_coerce_series():
    s = _coerce_series([1, 2])
    assert len(s) == 2

    s2 = _coerce_series({"a": 1, "b": 2})
    assert len(s2) == 2

    s3 = _coerce_series(pd.Series([1, 2]))
    assert len(s3) == 2


def test_coerce_sensor_layout():
    coords = np.array([[1, 2], [3, 4]])
    layout = coerce_sensor_layout(coords=coords)
    assert len(layout.names) == 2
    assert layout.positions.shape == (2, 2)

    layout2 = coerce_sensor_layout(coords={"A": [1, 2], "B": [3, 4]})
    assert layout2.names == ["A", "B"]

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "Sensor": ["A", "B"]})
    layout3 = coerce_sensor_layout(coords=df)
    assert layout3.names == ["A", "B"]


def test_info_from_montage():
    pytest.importorskip("mne")
    info = info_from_montage(["Cz", "Pz", "Fz", "NotAChannel"])
    # Unknown channels are dropped; montage positions are attached.
    assert info.ch_names == ["Cz", "Pz", "Fz"]
    layout = coerce_sensor_layout(info=info)
    assert set(layout.names) == {"Cz", "Pz", "Fz"}
    assert layout.positions.shape == (3, 2)


def test_prepare_embedding_frame():
    emb = np.random.randn(10, 2)
    df = prepare_embedding_frame(emb)
    assert "x" in df.columns

    df2 = prepare_embedding_frame(emb, labels=[1] * 10, metadata={"meta": [2] * 10})
    assert "Label" in df2.columns
    assert "meta" in df2.columns

    with pytest.raises(ValueError):
        prepare_embedding_frame(np.random.randn(10))

    with pytest.raises(ValueError):
        prepare_embedding_frame(emb, dimensions=1)

    with pytest.raises(ValueError):
        prepare_embedding_frame(emb, dimensions=3)


def test_prepare_metrics_frame():
    df = prepare_metrics_frame({"A": 1.0, "B": 2.0})
    assert len(df) == 2

    df2 = prepare_metrics_frame([{"Method": "M", "Metric": "A", "Value": 1.0}])
    assert len(df2) == 1

    with pytest.raises(TypeError):
        prepare_metrics_frame(None)


def test_prepare_interpretation_frame():
    df = prepare_interpretation_frame({"feat1": 1.0, "feat2": 2.0})
    assert len(df) == 2

    df2 = prepare_interpretation_frame({"analysis": {"my_analysis": {"feat": 1.0}}})
    assert len(df2) == 1

    with pytest.raises(TypeError):
        prepare_interpretation_frame(None)


def test_prepare_feature_scores():
    s = prepare_feature_scores({"feat1": 1.0, "feat2": 2.0})
    assert s["feat2"] == 2.0

    df = pd.DataFrame(
        {"Method": ["M"], "Analysis": ["A"], "Feature": ["feat"], "Value": [1.0]}
    )
    s2 = prepare_feature_scores(df)
    assert s2["feat"] == 1.0


def test_filter_metric_frame():
    df = pd.DataFrame(
        {
            "Method": ["A", "B", "A"],
            "Metric": ["m1", "m2", "m1"],
            "Scope": ["global", "global", "local"],
            "Value": [1.0, 2.0, 3.0],
        }
    )
    assert len(filter_metric_frame(df, metric="m1")) == 2
    assert len(filter_metric_frame(df, scope="local")) == 1
    assert len(filter_metric_frame(df, method="A")) == 2
    assert len(filter_metric_frame(df, method=["A", "B"])) == 3


def test_prepare_trajectory_metric_series_dict():
    series = {"A": [1, 2, 3], "B": [4, 5, 6]}
    df = prepare_trajectory_metric_series(series)
    assert len(df) == 6

    df2 = prepare_trajectory_metric_series(series, times=[0, 1, 2])
    assert (df2["Time"] == [0, 1, 2, 0, 1, 2]).all()

    with pytest.raises(ValueError):
        prepare_trajectory_metric_series({})
    with pytest.raises(ValueError):
        prepare_trajectory_metric_series({"A": [1, 2], "B": [1]})
    with pytest.raises(ValueError):
        prepare_trajectory_metric_series({"A": [1, 2]}, times=[0])


def test_prepare_trajectory_metric_series_1d():
    arr = np.array([1.0, 2.0, 3.0])
    df = prepare_trajectory_metric_series(arr)
    assert len(df) == 3

    df2 = prepare_trajectory_metric_series(arr, times=[10, 20, 30])
    assert df2["Time"].tolist() == [10, 20, 30]

    with pytest.raises(ValueError):
        prepare_trajectory_metric_series(arr, times=[0])
    with pytest.raises(ValueError):
        prepare_trajectory_metric_series(np.array([]))


def test_prepare_trajectory_metric_series_2d():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = prepare_trajectory_metric_series(arr, labels=["A", "B", "C"])
    assert len(df) == 9

    df2 = prepare_trajectory_metric_series(arr)
    assert "Metric" in df2["Series"].values

    with pytest.raises(ValueError):
        prepare_trajectory_metric_series(arr, labels=["A"])
    with pytest.raises(ValueError):
        prepare_trajectory_metric_series(arr, times=[0])
    with pytest.raises(ValueError):
        prepare_trajectory_metric_series(np.random.randn(2, 3, 4))


def test_scalar_decoding_scores():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "Metric": ["acc"],
            "Value": [0.9],
            "Time": [None],
        }
    )
    res = _scalar_decoding_scores(df, "ctx")
    assert len(res) == 1

    df_with_time = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "Metric": ["acc"],
            "Value": [0.9],
            "Time": [0.5],
        }
    )
    with pytest.raises(ValueError):
        _scalar_decoding_scores(df_with_time, "ctx")


def test_single_model_metric():
    df = pd.DataFrame({"Model": ["M"], "Metric": ["acc"], "Value": [0.9]})
    assert len(_single_model_metric(df, "ctx")) == 1

    df_multi = pd.DataFrame(
        {"Model": ["M1", "M2"], "Metric": ["acc", "f1"], "Value": [0.9, 0.8]}
    )
    with pytest.raises(ValueError):
        _single_model_metric(df_multi, "ctx")


def test_prediction_accuracy():
    group = pd.DataFrame({"y_true": [1, 0, 1], "y_pred": [1, 0, 0]})
    assert _prediction_accuracy(group) == pytest.approx(2 / 3)


def test_prepare_confusion_matrix():
    df = pd.DataFrame(
        {
            "TrueLabel": ["A", "A", "B", "B"],
            "PredictedLabel": ["A", "B", "A", "B"],
            "Value": [5, 1, 2, 7],
        }
    )
    cm = prepare_confusion_matrix(df)
    assert cm.shape == (2, 2)
    assert cm.loc["A", "A"] == 5


def test_prepare_curve_group_data():
    df = pd.DataFrame(
        {
            "Model": ["M1"] * 4,
            "Fold": [0, 0, 1, 1],
            "FPR": [0.0, 1.0, 0.0, 1.0],
            "TPR": [0.0, 0.8, 0.0, 0.9],
        }
    )
    records_mean = prepare_curve_group_data(df, "FPR", "TPR", mean_only=True)
    assert len(records_mean) == 1
    assert records_mean[0]["kind"] == "mean"

    records_fold = prepare_curve_group_data(df, "FPR", "TPR", mean_only=False)
    assert len(records_fold) == 2
    assert all(r["kind"] == "fold" for r in records_fold)

    df_no_fold = pd.DataFrame(
        {
            "Model": ["M1"] * 2,
            "FPR": [0.0, 1.0],
            "TPR": [0.0, 0.8],
        }
    )
    records_no_fold = prepare_curve_group_data(
        df_no_fold, "FPR", "TPR", mean_only=False
    )
    assert len(records_no_fold) == 1


def test_prepare_decoding_curve_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "FPR": [0.5],
            "TPR": [0.9],
        }
    )
    res = prepare_decoding_curve_frame(df, "get_roc_curves", ["FPR", "TPR"], "ROC")
    assert len(res) == 1


def test_prepare_fold_score_data():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "Metric": ["acc"],
            "Value": [0.9],
        }
    )
    res = prepare_fold_score_data(df)
    assert len(res) == 1


def test_auto_select_single_model():
    df1 = pd.DataFrame({"Model": ["M1", "M1"]})
    assert _auto_select_single_model(df1, None) == "M1"

    df2 = pd.DataFrame({"Model": ["M1", "M2"]})
    assert _auto_select_single_model(df2, None) is None

    assert _auto_select_single_model(df1, "explicit") == "explicit"

    df3 = pd.DataFrame({"A": [1]})
    assert _auto_select_single_model(df3, None) is None


def test_prepare_temporal_score_curve_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Metric": ["acc"],
            "Time": [0.5],
            "Mean": [0.9],
        }
    )
    res = prepare_temporal_score_curve_frame(df)
    assert len(res) == 1

    # No temporal rows
    df_no_time = pd.DataFrame(
        {
            "Model": ["M"],
            "Metric": ["acc"],
            "Time": [None],
            "Mean": [0.9],
        }
    )
    with pytest.raises(ValueError, match="No rows available"):
        prepare_temporal_score_curve_frame(df_no_time)

    # Hint for generalization matrix
    df_gen = pd.DataFrame(
        {
            "Model": ["M"],
            "Metric": ["acc"],
            "Time": [None],
            "Mean": [0.9],
            "TrainTime": [0.1],
            "TestTime": [0.2],
        }
    )
    with pytest.raises(ValueError, match="generalisation-matrix"):
        prepare_temporal_score_curve_frame(df_gen)


def test_prepare_temporal_generalization_matrix():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 4,
            "Metric": ["acc"] * 4,
            "TrainTime": [0.1, 0.1, 0.2, 0.2],
            "TestTime": [0.1, 0.2, 0.1, 0.2],
            "Mean": [0.9, 0.8, 0.7, 0.95],
        }
    )
    matrix, _first = prepare_temporal_generalization_matrix(df)
    assert matrix.shape == (2, 2)


def test_prepare_temporal_statistical_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Metric": ["acc"],
            "Time": [0.5],
            "Observed": [0.9],
        }
    )
    res = prepare_temporal_statistical_frame(df)
    assert len(res) == 1


def test_prepare_null_interval_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Metric": ["acc"],
            "Observed": [0.9],
        }
    )
    res = prepare_null_interval_frame(df)
    assert len(res) == 1


def test_prepare_training_history_artifacts():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Key": ["training"],
            "ArtifactType": ["history"],
            "Value": [0.5],
        }
    )
    res = prepare_training_history_artifacts(df)
    assert len(res) == 1


def test_prepare_decoding_score_data():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "Metric": ["acc"],
            "Value": [0.9],
        }
    )

    class Dummy:
        def get_detailed_scores(self):
            return df

    res = prepare_decoding_score_data(Dummy())
    assert len(res) == 1


def test_prepare_model_comparison_frame():
    df = pd.DataFrame(
        {
            "Difference": [0.1, 0.2],
            "ModelA": ["ref", "ref"],
            "ModelB": ["M1", "M2"],
        }
    )
    res = prepare_model_comparison_frame(df)
    assert len(res) == 2


def test_prepare_fit_diagnostics_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"],
            "Fold": [0],
            "TotalTime": [1.5],
        }
    )

    class Dummy:
        def get_fit_diagnostics(self):
            return df

    _frame, data = prepare_fit_diagnostics_frame(Dummy())
    assert len(data) == 1


def test_prepare_probability_diagnostics_summary():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 2,
            "Metric": ["brier", "brier"],
            "Value": [0.1, 0.2],
        }
    )

    class Dummy:
        def get_probability_diagnostics(self):
            return df

    res = prepare_probability_diagnostics_summary(Dummy())
    assert len(res) == 1


def test_prepare_prediction_accuracy_scores():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 4,
            "Subject": ["S1", "S1", "S2", "S2"],
            "y_true": [1, 0, 1, 1],
            "y_pred": [1, 0, 0, 1],
        }
    )

    class Dummy:
        def get_predictions(self):
            return df

    res = prepare_prediction_accuracy_scores(Dummy(), ["Subject"])
    assert len(res) == 2

    with pytest.raises(ValueError):
        prepare_prediction_accuracy_scores(Dummy(), [])


def test_prepare_regression_prediction_data():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 3,
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.1, 2.2, 2.9],
        }
    )

    class Dummy:
        def get_predictions(self):
            return df

    y_true, _y_pred = prepare_regression_prediction_data(Dummy())
    assert len(y_true) == 3

    df_nan = pd.DataFrame(
        {
            "Model": ["M"],
            "y_true": ["not_a_number"],
            "y_pred": ["also_nan"],
        }
    )

    class Dummy2:
        def get_predictions(self):
            return df_nan

    with pytest.raises(ValueError, match="No numeric"):
        prepare_regression_prediction_data(Dummy2())


def test_prepare_search_results_frame():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 3,
            "Rank": [1, 2, 3],
            "MeanTestScore": [0.9, 0.8, 0.7],
        }
    )

    class Dummy:
        def get_search_results(self):
            return df

    res = prepare_search_results_frame(Dummy(), top_n=2)
    assert len(res) == 2

    with pytest.raises(ValueError):
        prepare_search_results_frame(Dummy(), top_n=0)


def test_prepare_feature_stability_series():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 3,
            "FeatureName": ["f1", "f2", "f1"],
            "SelectionFrequency": [0.8, 0.5, 0.9],
        }
    )

    class Dummy:
        def get_feature_stability(self):
            return df

    res = prepare_feature_stability_series(Dummy())
    assert len(res) == 2


def test_prepare_feature_score_series():
    df = pd.DataFrame(
        {
            "Model": ["M"] * 2,
            "FeatureName": ["f1", "f2"],
            "Score": [0.8, 0.5],
        }
    )

    class Dummy:
        def get_feature_scores(self):
            return df

    res = prepare_feature_score_series(Dummy())
    assert len(res) == 2

    with pytest.raises(ValueError):
        prepare_feature_score_series(Dummy(), top_n=0)


def test_prepare_trajectory_data():
    X = np.random.randn(5, 10, 3)
    traj, _t, _labels, _values, _dims = prepare_trajectory_data(X)
    assert traj.shape == (5, 10, 2)

    _traj3, _, _, _, d = prepare_trajectory_data(X, dimensions=3)
    assert d == 3

    with pytest.raises(ValueError):
        prepare_trajectory_data(np.random.randn(5, 10))
    with pytest.raises(ValueError):
        prepare_trajectory_data(X, dimensions=4)
    with pytest.raises(ValueError):
        prepare_trajectory_data(np.random.randn(5, 10, 1), dimensions=2)
    with pytest.raises(ValueError):
        prepare_trajectory_data(X, downsample=0)
    with pytest.raises(ValueError):
        prepare_trajectory_data(X, times=np.arange(5))

    _traj_labels, _, labels_out, _, _ = prepare_trajectory_data(X, labels=["A"] * 5)
    assert labels_out is not None
    with pytest.raises(ValueError):
        prepare_trajectory_data(X, labels=["A"] * 3)

    vals = np.random.randn(5, 10)
    _traj_v, _, _, v, _ = prepare_trajectory_data(X, values=vals)
    assert v is not None
    with pytest.raises(ValueError):
        prepare_trajectory_data(X, values=np.random.randn(3, 10))

    # smooth
    traj_s, _t_s, _, _, _ = prepare_trajectory_data(X, smooth_window=3)
    assert traj_s.shape[1] < 10

    with pytest.raises(ValueError):
        prepare_trajectory_data(X, smooth_window=0)

    # downsample
    traj_d, _t_d, _, _, _ = prepare_trajectory_data(X, downsample=2)
    assert traj_d.shape[1] == 5

    # smooth + values + downsample
    traj_sv, _, _, v_sv, _ = prepare_trajectory_data(
        X, smooth_window=3, values=vals, downsample=2
    )
    assert v_sv.shape[1] == traj_sv.shape[1]


def test_prepare_loss_history():
    losses = prepare_loss_history([1.0, 0.5, 0.3])
    assert len(losses) == 3

    with pytest.raises(ValueError):
        prepare_loss_history([])
    with pytest.raises(ValueError):
        prepare_loss_history([1.0], scope="invalid")

    prepare_loss_history([1.0], scope="train")
    prepare_loss_history([1.0], scope="val")


def test_prepare_eigenvalue_curves():
    records = prepare_eigenvalue_curves([3.0, 2.0, 1.0])
    assert len(records) == 1
    assert records[0]["label"] == "Individual"

    records2 = prepare_eigenvalue_curves({"A": [3, 2, 1], "B": [2, 1, 0.5]})
    assert len(records2) == 2

    records3 = prepare_eigenvalue_curves(
        np.array([[3, 2, 1], [2.5, 1.5, 0.5]]), max_components=2
    )
    assert len(records3[0]["mean"]) == 2

    with pytest.raises(ValueError):
        prepare_eigenvalue_curves([1], max_components=0)
    with pytest.raises(ValueError):
        prepare_eigenvalue_curves({})
    with pytest.raises(ValueError):
        prepare_eigenvalue_curves(np.random.randn(2, 3, 4))
    with pytest.raises(ValueError):
        prepare_eigenvalue_curves(np.empty((2, 0)))


def test_prepare_shepard_distances():
    d_h, _d_l, corr = prepare_shepard_distances(
        np.zeros(1),
        np.zeros(1),
        distances={"original": [1, 2, 3], "embedded": [1.1, 2.1, 2.9]},
    )
    assert len(d_h) == 3
    assert abs(corr) <= 1.0

    with pytest.raises(ValueError):
        prepare_shepard_distances(
            np.zeros(1), np.zeros(1), distances={"original": [1, 2], "embedded": [1]}
        )
    with pytest.raises(ValueError):
        prepare_shepard_distances(
            np.zeros(1),
            np.zeros(1),
            distances={"original": [np.nan], "embedded": [np.nan]},
        )


def test_prepare_streamline_inputs():
    X = np.random.randn(10, 2)
    V = np.random.randn(10, 2)
    pts, _vecs = prepare_streamline_inputs(X, V)
    assert pts.shape == (10, 2)

    with pytest.raises(ValueError):
        prepare_streamline_inputs(np.random.randn(10, 3), V)
    with pytest.raises(ValueError):
        prepare_streamline_inputs(X, np.random.randn(5, 2))


def test_prepare_streamline_grid():
    X = np.random.randn(50, 2)
    V = np.random.randn(50, 2)
    Xi, _Yi, _Ui, _Vi = prepare_streamline_grid(X, V, grid_density=5)
    assert Xi.shape == (5, 5)

    with pytest.raises(ValueError):
        prepare_streamline_grid(X, V, grid_density=1)


def test_prepare_trajectory_separation_series():
    sep = {("A", "B"): [1, 2, 3], "C": [4, 5, 6]}
    records = prepare_trajectory_separation_series(sep)
    assert len(records) == 2
    assert "A vs B" in records[0]["label"] or "A vs B" in records[1]["label"]

    records2 = prepare_trajectory_separation_series(sep, top_n=1)
    assert len(records2) == 1

    with pytest.raises(ValueError):
        prepare_trajectory_separation_series({})
    with pytest.raises(ValueError):
        prepare_trajectory_separation_series({"A": []})
    with pytest.raises(ValueError):
        prepare_trajectory_separation_series({"A": [1, 2]}, times=[0])
    with pytest.raises(ValueError):
        prepare_trajectory_separation_series(sep, top_n=0)


def test_prepare_component_loadings_frame():
    comp = np.random.randn(5, 3)
    df = prepare_component_loadings_frame(comp)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["Component 1", "Component 2", "Component 3"]

    df2 = prepare_component_loadings_frame(
        comp, feature_names=["a", "b", "c", "d", "e"]
    )
    assert list(df2.index) == ["a", "b", "c", "d", "e"]

    df3 = prepare_component_loadings_frame(comp, n_components=2)
    assert df3.shape == (5, 2)

    with pytest.raises(ValueError):
        prepare_component_loadings_frame(None)
    with pytest.raises(ValueError):
        prepare_component_loadings_frame(np.random.randn(5))
    with pytest.raises(ValueError):
        prepare_component_loadings_frame(comp, n_components=0)
    with pytest.raises(ValueError):
        prepare_component_loadings_frame(comp, feature_names=["a", "b"])


def test_importance_with_metadata():
    imp_df = pd.DataFrame(
        {
            "FeatureName": ["f1", "f2"],
            "Mean": [0.5, 0.3],
        }
    )
    meta_df = pd.DataFrame(
        {
            "FeatureName": ["f1", "f2"],
            "x": [1, 2],
            "y": [3, 4],
        }
    )

    class Dummy:
        def get_feature_importances(self):
            return imp_df

    merged = _importance_with_metadata(Dummy(), meta_df)
    assert "_ImportanceValue" in merged.columns
    assert len(merged) == 2


def test_records_from_interpretation_payload():
    payload = {
        "correlation": {"dim1": {"feat1": 0.5, "feat2": 0.3}},
        "importance": {"feat1": 0.9, "feat2": 0.7},
    }
    records = _records_from_interpretation_payload(payload, "method")
    assert len(records) == 4

    # Empty nested
    records2 = _records_from_interpretation_payload(
        {"correlation": {"dim1": "not_a_mapping"}}, "m"
    )
    assert len(records2) == 0


def test_prepare_metrics_frame_wide():
    """Test wide-format metrics with melt."""
    df = pd.DataFrame(
        {
            "Method": ["A", "B"],
            "accuracy": [0.9, 0.8],
            "f1": [0.85, 0.75],
        }
    )
    res = prepare_metrics_frame(df)
    assert "Metric" in res.columns
    assert len(res) == 4  # 2 methods x 2 metrics


def test_prepare_metrics_frame_to_frame():
    """Test metrics from an object with to_frame()."""

    class MetricsObj:
        def to_frame(self):
            return pd.DataFrame(
                {
                    "Method": ["M"],
                    "Metric": ["acc"],
                    "Value": [0.9],
                }
            )

    res = prepare_metrics_frame(MetricsObj())
    assert len(res) == 1


def test_prepare_metrics_frame_edge_cases():
    with pytest.raises(TypeError):
        prepare_metrics_frame("invalid_type")

    # dict with non-numeric and ignored keys
    res = prepare_metrics_frame(
        {"n_iter_": 5, "valid": 0.9, "bool_val": True, "text": "abc"}
    )
    assert len(res) == 1
    assert res.iloc[0]["Metric"] == "valid"

    # empty df
    res2 = prepare_metrics_frame(pd.DataFrame(columns=["Method", "Metric", "Value"]))
    assert res2.empty

    # wide without Method column
    res3 = prepare_metrics_frame(pd.DataFrame({"acc": [0.9]}))
    assert len(res3) >= 1


def test_prepare_embedding_frame_continuous_label():
    emb = np.random.randn(10, 3)
    df = prepare_embedding_frame(
        emb, labels=[1.0] * 10, dimensions=3, label_kind="continuous"
    )
    assert "z" in df.columns

    with pytest.raises(ValueError):
        prepare_embedding_frame(emb, labels=[1] * 10, label_kind="invalid")
    with pytest.raises(ValueError):
        prepare_embedding_frame(emb, labels=[1] * 5)
    with pytest.raises(TypeError):
        prepare_embedding_frame(emb, metadata="not_a_mapping")
    with pytest.raises(ValueError):
        prepare_embedding_frame(emb, metadata={"col": [1] * 5})


def test_prepare_interpretation_frame_records_key():
    payload = {
        "records": [
            {"Method": "M", "Analysis": "A", "Feature": "f1", "Value": 0.5},
        ]
    }
    df = prepare_interpretation_frame(payload)
    assert len(df) == 1


def test_prepare_interpretation_frame_correlation_key():
    payload = {
        "correlation": {"dim1": {"feat": 0.5}},
    }
    df = prepare_interpretation_frame(payload)
    assert len(df) == 1


def test_prepare_feature_scores_edge_cases():
    df = pd.DataFrame(
        {
            "Method": ["M", "M"],
            "Analysis": ["A", "B"],
            "Feature": ["f1", "f2"],
            "Value": [0.5, 0.3],
        }
    )
    with pytest.raises(ValueError, match="analysis"):
        prepare_feature_scores(df)

    df2 = pd.DataFrame(
        {
            "Method": ["M1", "M2"],
            "Analysis": ["A", "A"],
            "Feature": ["f1", "f2"],
            "Value": [0.5, 0.3],
        }
    )
    with pytest.raises(ValueError, match="method"):
        prepare_feature_scores(df2)

    df3 = pd.DataFrame(
        {
            "Method": ["M"],
            "Analysis": ["A"],
            "Feature": ["f1"],
            "Value": [0.5],
            "Dimension": ["d1"],
        }
    )
    prepare_feature_scores(df3, dimension="d1")

    df_multi_dim = pd.DataFrame(
        {
            "Method": ["M", "M"],
            "Analysis": ["A", "A"],
            "Feature": ["f1", "f2"],
            "Value": [0.5, 0.3],
            "Dimension": ["d1", "d2"],
        }
    )
    with pytest.raises(ValueError, match="dimension"):
        prepare_feature_scores(df_multi_dim)

    with pytest.raises(TypeError):
        prepare_feature_scores(42)

    with pytest.raises(ValueError, match="No interpretation"):
        prepare_feature_scores(
            pd.DataFrame(
                {
                    "Method": [],
                    "Analysis": [],
                    "Feature": [],
                    "Value": [],
                }
            )
        )


def test_coerce_sensor_layout_with_names():
    coords = np.array([[1, 2], [3, 4], [5, 6]])
    layout = coerce_sensor_layout(coords=coords, names=["A", "B", "C"])
    assert layout.names == ["A", "B", "C"]

    with pytest.raises(ValueError):
        coerce_sensor_layout(coords=coords, names=["A"])
    with pytest.raises(ValueError):
        coerce_sensor_layout(coords=None)

    # FeatureName column
    df = pd.DataFrame({"x": [1], "y": [2], "FeatureName": ["F1"]})
    layout2 = coerce_sensor_layout(coords=df)
    assert layout2.names == ["F1"]

    # DataFrame with requested names
    df3 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "Sensor": ["A", "B"]})
    layout3 = coerce_sensor_layout(coords=df3, names=["A"])
    assert layout3.names == ["A"]

    # Index-based fallback
    df4 = pd.DataFrame({"x": [1], "y": [2]})
    layout4 = coerce_sensor_layout(coords=df4)
    assert len(layout4.names) == 1

    # df missing x/y
    with pytest.raises(ValueError):
        coerce_sensor_layout(coords=pd.DataFrame({"a": [1]}))

    # Mapping with names filter
    coords_map = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    layout5 = coerce_sensor_layout(coords=coords_map, names=["A", "C"])
    assert len(layout5.names) == 2

    # Empty after filter
    with pytest.raises(ValueError):
        coerce_sensor_layout(coords={"A": [1, 2]}, names=["Z"])

    # Bad array shape
    with pytest.raises(ValueError):
        coerce_sensor_layout(coords=np.array([[1]]))
