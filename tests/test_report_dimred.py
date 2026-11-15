"""
Tests for Dim-Red Components
"""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io.quality import QCResult
from coco_pipe.report.core import Report
from coco_pipe.report.dim_reduction import (
    add_reduction_components,
    add_reduction_coranking,
    add_reduction_diagnostics,
    add_reduction_embedding,
    add_reduction_interpretation,
    add_reduction_metrics,
    add_reduction_trajectory,
    add_reduction_trajectory_separation,
    make_reduction_report,
)
from coco_pipe.report.elements import (
    DownloadAssetElement,
    ImageElement,
    MetricsTableElement,
    PlotlyElement,
    TableElement,
    TabsElement,
)

FEATURES = ["a", "b", "c", "d", "e"]


def _scored_pca():
    """A PCA reduction of 30 x 5 data, scored and interpreted."""
    X = np.random.default_rng(0).normal(size=(30, 5))
    reduction = DimReduction("PCA", n_components=2)
    X_emb = reduction.fit_transform(X)
    reduction.score(X_emb, X=X)
    reduction.interpret(
        X, X_emb=X_emb, analyses=["correlation"], feature_names=FEATURES
    )
    return reduction, X, X_emb


def _scored_trajectory(times):
    """A PCA reduction scored on 6 trajectories of 8 time points."""
    rng = np.random.default_rng(1)
    reduction = DimReduction("PCA", n_components=2).fit(rng.normal(size=(30, 5)))
    X_emb = rng.normal(size=(6, 8, 2))
    reduction.score(X_emb, labels=np.array([0, 1] * 3), times=times)
    return reduction, X_emb


class _Summary:
    """A reduction known only by its summary."""

    def __init__(self, summary, components=None):
        self.summary = summary
        self.components = components

    def get_summary(self):
        return self.summary

    def get_components(self):
        return self.components


def _plots(section):
    return [
        child.figure for child in section.children if isinstance(child, PlotlyElement)
    ]


def _titles(report):
    return [section.title for section in report.children]


def test_reduction_report_renders_qc_result():
    qc_result = QCResult(
        n_rows_entering_qc=12,
        n_dropped_nan_inf=1,
        n_dropped_extreme=1,
        n_obs_in=10,
        n_obs_out=9,
        n_subjects_in=10,
        n_subjects_out=9,
        family_qc=pd.DataFrame({"family": ["band"], "n_features": [4]}),
    )

    report = make_reduction_report([], qc_result=qc_result)
    html = report.render()

    assert "Data Quality (QC)" in html
    assert "QC Funnel Summary" in html
    assert "Family-Level Quality Summary" in html
    assert "N Rows Entering Qc" in html


def test_add_reduction_shows_the_scores_and_diagnostics_of_a_reduction():
    reduction, _, X_emb = _scored_pca()
    summary = reduction.get_summary()

    rep = Report("DimRed Test")
    labels = np.array([0, 1] * 15)
    assert rep.add_reduction(reduction, "My PCA", X_emb=X_emb, labels=labels) is rep

    (section,) = rep.children
    assert section.title == "My PCA"
    (table,) = [child for child in section.children if isinstance(child, TableElement)]
    assert table.title == "Quality Metrics"
    assert table.data["trustworthiness"] == summary["metrics"]["trustworthiness"]
    assert table.data["n_components_"] == 2

    embedding, metrics, scree, coranking = _plots(section)
    assert embedding.layout.title.text == "My PCA Embedding"
    points = np.concatenate([trace.x for trace in embedding.data])
    np.testing.assert_allclose(np.sort(points), np.sort(X_emb[:, 0]))
    assert metrics.layout.title.text == "Metric Details"
    np.testing.assert_allclose(
        scree.data[0].y, summary["diagnostics"]["explained_variance_ratio_"]
    )
    np.testing.assert_array_equal(
        coranking.data[0].z, summary["diagnostics"]["coranking_matrix_"]
    )
    assert "My PCA" in rep.render()


def test_add_reduction_plots_the_trajectory_diagnostics():
    reduction, X_emb = _scored_trajectory(None)

    rep = Report()
    rep.add_reduction(reduction, X_emb=X_emb)
    trajectory, _, _, *plots = _plots(rep.children[0])
    assert trajectory.layout.title.text == "PCA Trajectory"
    # one plot per per-time series, then the class separation
    assert [plot.layout.title.text for plot in plots] == [
        "Trajectory Speed",
        "Trajectory Acceleration",
        "Trajectory Curvature",
        "Trajectory Turning Angle",
        "Trajectory Dispersion",
        "Trajectory Path Length",
        "Trajectory Displacement",
        "Trajectory Separation",
    ]
    np.testing.assert_array_equal(plots[0].data[0].x, np.arange(8))


def test_add_reduction_plots_trajectory_series_on_their_time_axis():
    times = np.arange(8) * 10.0
    speed = np.random.default_rng(3).random((6, 8))
    reduction = _Summary(
        {"diagnostics": {"trajectory_times_": times, "trajectory_speed_": speed}}
    )

    rep = Report()
    # without explicit times, the series follow the scored time axis
    rep.add_reduction(reduction, "Scored")
    (series,) = _plots(rep.children[0])
    np.testing.assert_array_equal(series.data[0].x, times)
    rep.add_reduction(reduction, "Explicit", times=times + 5)
    (series,) = _plots(rep.children[1])
    np.testing.assert_array_equal(series.data[0].x, times + 5)


def test_add_reduction_plots_turning_angles_on_the_inner_time_points():
    times = np.arange(8) * 10.0
    reduction, X_emb = _scored_trajectory(times)

    rep = Report()
    rep.add_reduction(reduction, X_emb=X_emb)
    plots = {plot.layout.title.text: plot for plot in _plots(rep.children[0])}
    np.testing.assert_array_equal(plots["Trajectory Speed"].data[0].x, times)
    for title in ("Trajectory Curvature", "Trajectory Turning Angle"):
        np.testing.assert_array_equal(plots[title].data[0].x, times[1:-1])


def test_add_reduction_reads_the_summary_contract():
    rep = Report("Summary Test")
    with pytest.raises(TypeError, match="must implement get_summary"):
        rep.add_reduction(object())
    with pytest.raises(TypeError, match="must return a dictionary"):
        rep.add_reduction(_Summary("not a dict"))
    with pytest.raises(ValueError, match="2D embedding or 3D trajectory"):
        rep.add_reduction(_Summary({}), X_emb=np.zeros((2, 2, 2, 2)))

    # a partial summary is named after its method and shows what it has
    loss = {"method": "AE", "diagnostics": {"loss_history_": [3.0, 2.0, 1.0]}}
    rep.add_reduction(_Summary(loss))
    (section,) = rep.children
    assert section.title == "AE"
    (figure,) = _plots(section)
    np.testing.assert_array_equal(figure.data[0].y, [3.0, 2.0, 1.0])


def test_add_comparison_summarises_methods():
    scores = {"PCA": (0.9, 0.1, 0.8), "UMAP": (0.95, 0.05, 0.85)}
    rows = [
        {"Method": method, "Metric": metric, "Value": value, "ScopeValue": "A"}
        for method, values in scores.items()
        for metric, value in zip(("Trust", "Loss", "Continuity"), values, strict=True)
    ]
    rep = Report()
    rep.add_comparison(pd.DataFrame(rows))
    (section,) = rep.children
    assert section.title == "Method Comparison"
    (table,) = [c for c in section.children if isinstance(c, MetricsTableElement)]
    assert table.data.loc["UMAP", "Trust"] == 0.95
    # three metrics of two methods on one scope add a radar chart
    titles = [figure.layout.title.text for figure in _plots(section)]
    assert titles[:2] == ["Metric Heatmap", "Metric Details"]
    assert len(titles) == 3

    wide = pd.DataFrame(
        {"Method": ["PCA", "ISO"], "Trustworthiness": [0.8, 0.6]}
    ).set_index("Method")
    rep.add_comparison(wide, name="Wide")
    assert len(_plots(rep.children[1])) == 2

    with pytest.raises(ValueError, match="No comparison metrics"):
        rep.add_comparison(pd.DataFrame())


def test_make_reduction_report_sections_of_a_reduction():
    reduction, _, X_emb = _scored_pca()
    report = make_reduction_report(
        [reduction],
        embeddings=[X_emb],
        sections=[
            "overview",
            "embedding",
            "metrics",
            "coranking",
            "components",
        ],
    )
    assert _titles(report) == [
        "PCA Overview",
        "PCA Embedding",
        "PCA Quality Metrics",
        "PCA Co-Ranking Matrix",
        "PCA Component Loadings",
    ]
    overview = report.children[0].children[0].data
    assert dict(zip(overview["Key"], overview["Value"], strict=True))["method"] == "PCA"
    download = report.children[2].children[0]
    assert isinstance(download, DownloadAssetElement)
    exported = pd.read_csv(StringIO(download.data))
    assert exported["metric"].tolist() == [
        record["metric"] for record in reduction.get_summary()["metric_records"]
    ]
    assert "PCA Embedding" in report.render()


def test_make_reduction_report_pairs_the_embedding_with_its_shepard_diagram():
    X = np.random.default_rng(2).normal(size=(20, 4))
    X_emb = X[:, :2]

    def first_section(diagnostics, sections=("embedding",)):
        report = make_reduction_report(
            [_Summary({"method": "M", "diagnostics": diagnostics})],
            embeddings=[X_emb],
            sections=list(sections),
        )
        return report.children[0]

    (tabs,) = first_section({"X_orig": X}).children
    assert isinstance(tabs, TabsElement)
    assert list(tabs.tabs) == ["Embedding", "Shepard Diagram"]
    # an original that does not align leaves the embedding alone
    (image,) = first_section({"X_orig": X[:5]}).children
    assert isinstance(image, ImageElement)
    assert image.caption == "Embedding"
    # without the embedding section, diagnostics show the Shepard diagram
    diagnostics = first_section({"X_orig_": X}, sections=("diagnostics",))
    assert diagnostics.title == "M Diagnostics"
    assert diagnostics.children[0].caption == "Shepard diagram"


def test_make_reduction_report_takes_components_from_a_mapping():
    reduction = _Summary({"method": "M"}, components={"components": np.ones((2, 3))})
    report = make_reduction_report([reduction], sections=["components"])
    assert _titles(report) == ["M Component Loadings"]


def test_make_reduction_report_shows_each_analysis_of_the_interpretation():
    reduction, X, X_emb = _scored_pca()
    reduction.interpret(
        X,
        X_emb=X_emb,
        analyses=["correlation", "perturbation"],
        feature_names=FEATURES,
        random_state=0,
    )

    report = make_reduction_report([reduction], sections=["interpretation"])
    (section,) = report.children
    assert section.title == "PCA Interpretation"
    correlation, perturbation = section.children
    assert [correlation.caption, perturbation.caption] == [
        "correlation",
        "perturbation",
    ]
    # correlations are a feature x dimension heatmap
    axes = correlation.src.axes[0]
    assert sorted(label.get_text() for label in axes.get_yticklabels()) == FEATURES
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "Dimension 1",
        "Dimension 2",
    ]

    # the summary's analysis payload shows the same analyses
    report = Report()
    report.add_reduction_interpretation(reduction.get_summary()["interpretation"])
    (section,) = report.children
    assert [image.caption for image in section.children] == [
        "correlation",
        "perturbation",
    ]


def test_make_reduction_report_interactive_warning():
    with pytest.warns(UserWarning, match="not yet implemented"):
        make_reduction_report([], interactive=True)


def test_make_reduction_report_mismatched_embeddings():
    reduction, _, X_emb = _scored_pca()
    with pytest.raises(ValueError, match="must align with `reductions`"):
        make_reduction_report([reduction], embeddings=[X_emb, X_emb])


def test_reduction_report_empty_edge_cases():
    class EmptyReduction:
        def get_summary(self):
            return {}

    reduction = EmptyReduction()
    report = Report("Empty")

    # Should not raise any errors, just return self
    report.add_reduction_overview(reduction)
    report.add_reduction_embedding(None)
    report.add_reduction_metrics(reduction)
    report.add_reduction_diagnostics(None, None)
    report.add_reduction_interpretation({})
    report.add_reduction_coranking(None)
    report.add_reduction_components(None)
    report.add_reduction_trajectory(None)
    report.add_reduction_trajectory_separation({})
    assert len(report.children) == 0


def test_add_reduction_interpretation_ranks_the_features():
    X = np.random.default_rng(0).normal(size=(30, 5))
    reduction = DimReduction("PCA", n_components=2)
    X_emb = reduction.fit_transform(X)
    result = reduction.interpret(
        X,
        X_emb=X_emb,
        analyses=["perturbation"],
        feature_names=FEATURES,
        random_state=0,
    )
    scores = result["analysis"]["perturbation"]

    report = Report("Interp")
    report.add_reduction_interpretation(result["records"], analysis="perturbation")
    (section,) = report.children
    assert section.title == "Interpretation"
    (image,) = section.children
    assert image.caption == "perturbation"
    axes = image.src.axes[0]
    # one bar per feature, the most important first
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert [label.get_text() for label in axes.get_yticklabels()] == ranked
    np.testing.assert_allclose(
        [bar.get_width() for bar in axes.patches], [scores[name] for name in ranked]
    )


def test_add_reduction_components_plots_the_features_on_the_rows():
    reduction, _, _ = _scored_pca()
    components = reduction.get_components()
    assert components.shape == (2, len(FEATURES))

    report = Report()
    report.add_reduction_components(components, feature_names=FEATURES)
    (section,) = report.children
    (image,) = section.children
    axes = image.src.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == FEATURES
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "Component 1",
        "Component 2",
    ]


def test_section_builders_add_one_section_each():
    reduction, X, X_emb = _scored_pca()
    coranking = reduction.get_summary()["diagnostics"]["coranking_matrix_"]
    trajectory, X_3d = _scored_trajectory(np.arange(8))
    diagnostics = trajectory.get_summary()["diagnostics"]

    report = Report("Full")
    report.add_reduction_overview(reduction)
    report.add_reduction_embedding(X_emb)
    report.add_reduction_metrics(reduction)
    report.add_reduction_diagnostics(X, X_emb)
    report.add_reduction_coranking(coranking)
    report.add_reduction_components(reduction.get_components())
    report.add_reduction_trajectory(X_3d)
    report.add_reduction_trajectory_separation(diagnostics["trajectory_separation_"])

    assert _titles(report) == [
        "Overview",
        "Embedding",
        "Quality Metrics",
        "Diagnostics",
        "Co-Ranking Matrix",
        "Component Loadings",
        "Trajectory",
        "Trajectory Separation",
    ]
    for section in report.children[1:]:
        assert any(isinstance(child, ImageElement) for child in section.children)


_RECORDS = [{"method": "PCA", "metric": "trust", "value": 0.9, "scope": "global"}]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda rep: add_reduction_embedding(rep, np.zeros((5, 2)), labels=[0, 1]),
            id="embedding",
        ),
        pytest.param(
            lambda rep: add_reduction_metrics(
                rep, _Summary({"metric_records": _RECORDS}), metric="missing"
            ),
            id="metrics",
        ),
        pytest.param(
            lambda rep: add_reduction_diagnostics(
                rep, np.zeros((10, 5)), np.zeros((8, 2))
            ),
            id="diagnostics",
        ),
        pytest.param(
            lambda rep: add_reduction_interpretation(
                rep, [{"feature": "a", "value": 1.0}], analysis="loadings"
            ),
            id="interpretation",
        ),
        pytest.param(
            lambda rep: add_reduction_interpretation(rep, "not a payload"),
            id="interpretation_payload",
        ),
        pytest.param(
            lambda rep: add_reduction_coranking(rep, np.ones((3, 4))),
            id="coranking",
        ),
        pytest.param(
            lambda rep: add_reduction_components(rep, np.ones(3)),
            id="components",
        ),
        pytest.param(
            lambda rep: add_reduction_trajectory(rep, np.ones((5, 2))),
            id="trajectory",
        ),
        pytest.param(
            lambda rep: add_reduction_trajectory_separation(
                rep, {"A-B": np.ones(5)}, times=np.arange(3)
            ),
            id="trajectory_separation",
        ),
    ],
)
def test_section_builders_skip_data_they_cannot_plot(call):
    """A section whose data cannot be plotted is left out of the report."""
    rep = Report("Test")
    assert call(rep) is rep
    assert rep.children == []
