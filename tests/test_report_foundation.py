from coco_pipe.report.foundation import (
    make_foundation_decoding_report,
    make_foundation_embedding_report,
)
from tests.fixtures.synthetic_result import make_synthetic_result


def test_foundation_embedding_report_renders_offline(tmp_path, monkeypatch):
    asset_cache = tmp_path / "assets"
    asset_cache.mkdir()
    for name in ("plotly", "tailwind", "pako"):
        (asset_cache / f"{name}.js").write_text(
            f"window.{name}=true;",
            encoding="utf-8",
        )
    monkeypatch.setenv("COCO_PIPE_REPORT_ASSET_CACHE", str(asset_cache))
    path = tmp_path / "report.html"
    report = make_foundation_embedding_report(
        [
            {
                "subject": "01",
                "model_key": "cbramod",
                "window_count": 10,
                "embedding_shape": [10, 200],
                "status": "success",
            }
        ],
        output_path=str(path),
    )
    assert path.exists()
    assert "Extraction Overview" in report.render()
    html = path.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in html
    assert "window.plotly=true;" in html


def test_foundation_embedding_report_renders_nested_channel_adaptation():
    report = make_foundation_embedding_report(
        [
            {
                "model_key": "labram",
                "status": "success",
                "channel_adaptation": {
                    "interpolated_channels": ["Fpz"],
                    "zero_filled_channels": [],
                    "dropped_channels": [],
                },
            }
        ],
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )

    assert "Fpz" in report.render()


def test_foundation_embedding_report_empty():
    report = make_foundation_embedding_report(
        [], asset_urls={"plotly": "", "tailwind": "", "pako": ""}
    )
    assert "No embedding extraction records were produced" in report.render()


def test_foundation_decoding_report(tmp_path):
    records = []
    for model_key in ("labram", "cbramod"):
        output_dir = tmp_path / f"{model_key}_linear_probe"
        make_synthetic_result(n_models=1, n_times=1).save(output_dir / "result.joblib")
        records.append(
            {
                "status": "success",
                "condition": "EO",
                "target": "adhd",
                "model_key": model_key,
                "train_mode": "linear_probe",
                "primary_metric": 0.7,
                "primary_metric_name": "balanced_accuracy",
                "output_dir": str(output_dir),
            }
        )

    path = tmp_path / "decoding.html"
    report = make_foundation_decoding_report(
        records,
        capability_records=[
            {"model_key": "labram", "train_mode": "linear_probe", "status": "ok"}
        ],
        dataset_name="DemoDS",
        output_path=str(path),
        asset_urls={"plotly": "", "tailwind": "", "pako": ""},
    )
    assert path.exists()
    html = report.render()
    assert "Scientific Overview" in html
    assert "Foundation Capability Matrix" in html
    assert "Linear Probe Leaderboard" in html
    assert "Training-Mode Comparison" in html
    assert "Per-Result Diagnostics" in html


def test_foundation_decoding_report_empty_records():
    report = make_foundation_decoding_report(
        [], asset_urls={"plotly": "", "tailwind": "", "pako": ""}
    )
    assert "No foundation decoding units were produced." in report.render()
