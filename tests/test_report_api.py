from unittest.mock import MagicMock, patch

import pytest

from coco_pipe.io.quality import QCResult
from coco_pipe.report.api import (
    from_bids,
    from_container,
    from_embeddings,
    from_experiment_result,
    from_experiment_results,
    from_reductions,
    from_tabular,
    merge_reports,
)
from coco_pipe.report.core import Report


@patch("coco_pipe.report.core.Report")
def test_from_container(MockReport):
    mock_report = MockReport.return_value
    mock_container = MagicMock()

    rep = from_container(
        mock_container,
        title="Test Container",
        config={"key": "val"},
        raw_preview=True,
        theme="notebook",
        output_path="out.html",
    )

    MockReport.assert_called_once_with(
        title="Test Container", config={"key": "val"}, theme="notebook", asset_urls=None
    )
    mock_report.add_container.assert_called_once_with(mock_container)
    mock_report.add_raw_preview.assert_called_once_with(mock_container)
    mock_report.save.assert_called_once_with("out.html")
    assert rep == mock_report


@patch("coco_pipe.report.api.from_container")
@patch("coco_pipe.io.dataset.BIDSDataset")
def test_from_bids(MockBIDSDataset, mock_from_container):
    mock_ds = MockBIDSDataset.return_value
    mock_container = MagicMock()
    mock_ds.load.return_value = mock_container
    mock_from_container.return_value = "report_mock"

    rep = from_bids(
        root="/fake/bids",
        task="motor",
        theme="poster",
        raw_preview=False,
        output_path="bids.html",
        extra_arg="extra",
    )

    MockBIDSDataset.assert_called_once_with(
        root="/fake/bids", task="motor", extra_arg="extra"
    )
    mock_ds.load.assert_called_once()
    mock_from_container.assert_called_once()

    args, kwargs = mock_from_container.call_args
    assert args[0] == mock_container
    assert kwargs["title"] == "BIDS Report: motor"
    assert kwargs["config"]["run_params"]["source"] == "BIDS"
    assert kwargs["theme"] == "poster"
    assert kwargs["raw_preview"] is False
    assert kwargs["output_path"] == "bids.html"

    assert rep == "report_mock"


@patch("coco_pipe.report.api.from_container")
@patch("coco_pipe.io.dataset.TabularDataset")
def test_from_tabular(MockTabularDataset, mock_from_container):
    mock_ds = MockTabularDataset.return_value
    mock_container = MagicMock()
    mock_ds.load.return_value = mock_container

    from_tabular("data.csv", output_path="tab.html")
    MockTabularDataset.assert_called_once_with(path="data.csv")
    mock_ds.load.assert_called_once()

    _args, kwargs = mock_from_container.call_args
    assert kwargs["title"] == "Tabular Report: data.csv"
    assert kwargs["config"]["run_params"]["source"] == "Tabular"


@patch("coco_pipe.report.api.from_container")
@patch("coco_pipe.io.dataset.EmbeddingDataset")
def test_from_embeddings(MockEmbeddingDataset, mock_from_container):
    mock_ds = MockEmbeddingDataset.return_value
    mock_container = MagicMock()
    mock_ds.load.return_value = mock_container

    from_embeddings("embs/", output_path="emb.html")
    MockEmbeddingDataset.assert_called_once_with(path="embs/")

    _args, kwargs = mock_from_container.call_args
    assert kwargs["title"] == "Embedding Report: embs"
    assert kwargs["config"]["run_params"]["source"] == "Embeddings"


@patch("coco_pipe.report.dim_reduction.make_reduction_report")
def test_from_reductions_no_container(mock_make):
    mock_report = MagicMock()
    mock_make.return_value = mock_report

    rep = from_reductions(["pca", "tsne"], title="Reductions")

    mock_make.assert_called_once()
    assert rep == mock_report


@patch("coco_pipe.report.dim_reduction.make_reduction_report")
def test_from_reductions_forwards_qc_result(mock_make):
    mock_make.return_value = MagicMock()
    qc_result = QCResult(
        n_obs_in=10,
        n_obs_out=9,
        n_subjects_in=5,
        n_subjects_out=5,
    )

    from_reductions(["pca"], qc_result=qc_result)

    assert mock_make.call_args.kwargs["qc_result"] is qc_result


@patch("coco_pipe.report.dim_reduction.make_reduction_report")
def test_from_reductions_with_container(mock_make):
    mock_report = MagicMock()
    mock_make.return_value = mock_report
    mock_container = MagicMock()

    from_reductions(
        ["pca"], container=mock_container, raw_preview=True, output_path="red.html"
    )

    mock_report.add_container.assert_called_once_with(mock_container)
    mock_report.add_raw_preview.assert_called_once_with(mock_container)
    mock_report.save.assert_called_once_with("red.html")


@patch("coco_pipe.report.decoding.make_decoding_report")
def test_from_experiment_result(mock_make):
    mock_report = MagicMock()
    mock_make.return_value = mock_report

    from_experiment_result("result_mock", title="Dec")
    mock_make.assert_called_once()
    assert mock_make.call_args[0][0] == "result_mock"
    assert mock_make.call_args[1]["title"] == "Dec"


@patch("coco_pipe.report.decoding.make_decoding_report")
def test_from_experiment_result_forwards_qc_result(mock_make):
    mock_make.return_value = MagicMock()
    qc_result = QCResult(
        n_obs_in=10,
        n_obs_out=8,
        n_subjects_in=5,
        n_subjects_out=4,
    )

    from_experiment_result("result_mock", qc_result=qc_result)

    assert mock_make.call_args.kwargs["qc_result"] is qc_result


@patch("coco_pipe.report.decoding.make_decoding_report")
def test_from_experiment_result_forwards_composition_options(mock_make):
    mock_make.return_value = MagicMock()
    coords = {"Fp1": (-0.2, 0.4)}
    section_options = {"cv": {"metric": "accuracy"}}

    from_experiment_result(
        "result_mock",
        coords=coords,
        sections="compact",
        verbose=False,
        on_error="placeholder",
        section_options=section_options,
    )

    kwargs = mock_make.call_args.kwargs
    assert kwargs["coords"] is coords
    assert kwargs["on_error"] == "placeholder"
    assert kwargs["section_options"] is section_options


@patch("coco_pipe.report.decoding_comparison.make_experiment_results_report")
def test_from_experiment_results(mock_make):
    mock_make.return_value = MagicMock()
    items = [({"scope": "EO"}, "result.joblib")]

    from_experiment_results(items, by=("scope",), title="Many")

    mock_make.assert_called_once()
    assert mock_make.call_args.args[0] == items
    assert mock_make.call_args.kwargs["by"] == ("scope",)
    assert mock_make.call_args.kwargs["title"] == "Many"


def test_merge_reports_errors():
    with pytest.raises(ValueError, match="requires at least two Report objects"):
        merge_reports(Report())


def test_merge_reports():
    rep1 = Report(title="Report 1")
    sec1 = MagicMock()
    sec1.title = "Section A"
    rep1.children = [sec1]

    rep2 = Report(title="Report 2")
    sec2 = MagicMock()
    sec2.title = ""
    rep2.children = [sec2]

    merged = merge_reports(rep1, rep2, title="Merged Title")

    assert merged.title == "Merged Title"
    assert len(merged.children) == 2

    # Assert section titles were prefixed
    assert merged.children[0].title == "Report 1 — Section A"
    assert merged.children[0].id == "report-1-section-a"

    assert merged.children[1].title == "Report 2"
    assert merged.children[1].id == "report-2"
