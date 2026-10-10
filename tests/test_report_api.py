import numpy as np
import pandas as pd
import pytest

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io.quality import QCResult
from coco_pipe.io.structures import DataContainer
from coco_pipe.report.api import (
    from_bids,
    from_container,
    from_decoding_sweep,
    from_embeddings,
    from_experiment_result,
    from_experiment_results,
    from_foundation_sweep,
    from_head_to_head,
    from_reductions,
    from_tabular,
    merge_reports,
)
from coco_pipe.report.core import Report
from tests.fixtures.synthetic_result import make_synthetic_result

ASSETS = {"plotly": "", "tailwind": "", "pako": ""}


def _container():
    X = np.random.default_rng(0).normal(size=(12, 3))
    return DataContainer(
        X, dims=("obs", "feature"), coords={"feature": ["a", "b", "c"]}
    )


def _saved_result(path):
    make_synthetic_result(n_models=1, n_times=1).save(path / "result.joblib")
    return str(path)


def test_from_container_saves_the_overview(tmp_path):
    path = tmp_path / "container.html"
    report = from_container(
        _container(), title="Demo", asset_urls=ASSETS, output_path=path
    )
    assert report.title == "Demo"
    assert "Dimensions" in path.read_text(encoding="utf-8")
    assert len(from_container(_container(), raw_preview=False).children) < len(
        report.children
    )


def test_dataset_reports_load_their_source(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path / "bids", "01")
    bids = from_bids(tmp_path / "bids", task="rest", suffix="epo")
    assert bids.title == "BIDS Report: rest"

    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 0.1, 0.2]}).to_csv(
        tmp_path / "data.tsv", sep="\t", index=False
    )
    tabular = from_tabular(tmp_path / "data.tsv")
    assert tabular.title == "Tabular Report: data.tsv"

    np.save(tmp_path / "emb.npy", np.ones((4, 3)))
    embeddings = from_embeddings(tmp_path / "emb.npy")
    assert embeddings.title == "Embedding Report: emb.npy"

    for report in (bids, tabular, embeddings):
        assert "Dimensions" in report.render()


def test_from_reductions_adds_the_data_and_qc(tmp_path):
    container = _container()
    reducer = DimReduction("PCA", n_components=2)
    embedding = reducer.fit_transform(container.X)
    reducer.score(embedding, X=container.X)
    qc = QCResult(n_obs_in=13, n_obs_out=12, n_subjects_in=4, n_subjects_out=4)

    path = tmp_path / "reductions.html"
    report = from_reductions(
        [reducer],
        container=container,
        embeddings=[embedding],
        qc_result=qc,
        asset_urls=ASSETS,
        output_path=path,
    )
    html = path.read_text(encoding="utf-8")
    assert "Dimensions" in html
    assert "PCA" in html
    assert len(report.children) > len(from_reductions([reducer]).children)


def test_from_experiment_result_renders(tmp_path):
    path = tmp_path / "decoding.html"
    from_experiment_result(
        make_synthetic_result(n_models=1, n_times=1),
        title="Decoding",
        sections="compact",
        asset_urls=ASSETS,
        output_path=path,
    )
    assert "Decoding" in path.read_text(encoding="utf-8")


def test_from_experiment_results_compares_labelled_results(tmp_path):
    items = [
        ({"scope": scope}, make_synthetic_result(n_models=1, n_times=1))
        for scope in ("EO", "EC")
    ]
    path = tmp_path / "many.html"
    from_experiment_results(items, by=("scope",), asset_urls=ASSETS, output_path=path)
    html = path.read_text(encoding="utf-8")
    assert "EO" in html and "EC" in html


def test_sweep_reports(tmp_path):
    common = {"status": "success", "target": "adhd", "primary_metric": 0.7}
    classical = from_decoding_sweep(
        [
            {
                **common,
                "scope": "EO",
                "analysis_mode": "flat",
                "selection_mode": "baseline",
                "unit_name": unit,
                "model": "logreg",
                "balanced_accuracy_mean": 0.7,
                "output_dir": _saved_result(tmp_path / unit),
            }
            for unit in ("Fz", "Cz")
        ],
        asset_urls=ASSETS,
    )
    assert classical.children

    foundation = from_foundation_sweep(
        [
            {
                **common,
                "condition": "EO",
                "model_key": "labram",
                "train_mode": "linear_probe",
                "primary_metric_name": "balanced_accuracy",
                "output_dir": _saved_result(tmp_path / "labram"),
            }
        ],
        asset_urls=ASSETS,
    )
    assert "Linear Probe Leaderboard" in foundation.render()

    signature = {
        "scope": "EO",
        "condition": "EO",
        "target": "adhd",
        "primary_metric_name": "balanced_accuracy",
        "cv_strategy": "stratified_group_kfold",
        "effective_n_splits": 5,
        "cv_random_state": 42,
        "cohort_signature": "same-samples",
    }
    frame = pd.DataFrame(
        [
            {
                "comparison_family": "descriptor_flat_baseline",
                "model": "logreg",
                "primary_metric": 0.6,
                **signature,
            },
            {
                "comparison_family": "foundation_linear_probe",
                "model_key": "labram",
                "primary_metric": 0.7,
                **signature,
            },
        ]
    )
    head_to_head = from_head_to_head(
        frame,
        baseline_family="descriptor_flat_baseline",
        title="Head-to-Head",
        asset_urls=ASSETS,
    )
    assert "Paired Delta vs descriptor_flat_baseline" in head_to_head.render()


def test_merge_reports_prefixes_sections_with_their_report():
    first = from_container(_container(), title="Cohort A", raw_preview=False)
    second = from_container(_container(), title="Cohort B", raw_preview=False)

    merged = merge_reports(first, second, title="Both")

    assert merged.title == "Both"
    titles = [section.title for section in merged.children]
    assert titles[0].startswith("Cohort A — ")
    assert titles[1].startswith("Cohort B — ")
    assert len({section.id for section in merged.children}) == 2

    with pytest.raises(ValueError, match="at least two"):
        merge_reports(Report())
