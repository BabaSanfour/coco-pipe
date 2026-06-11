from coco_pipe.report.foundation import (
    make_foundation_decoding_report,
    make_foundation_embedding_report,
)


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
    class DummyResult:
        pass

    from unittest.mock import patch

    with patch("coco_pipe.report.foundation.make_decoding_report") as mock_make:
        from coco_pipe.report.core import Report

        mock_report = Report("Dummy")
        mock_make.return_value = mock_report

        path = tmp_path / "decoding.html"
        report = make_foundation_decoding_report(
            DummyResult(),
            capability_records=[{"model": "test", "capability": "yes"}],
            output_path=str(path),
        )
        assert path.exists()
        assert "Foundation Capability Matrix" in report.render()
