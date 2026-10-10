from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from coco_pipe.report._assets import (
    _cache_dir,
    _download_to,
    get_vendored_contents,
    vendor_assets,
)


def test_cache_dir_no_env(monkeypatch):
    monkeypatch.delenv("COCO_PIPE_REPORT_ASSET_CACHE", raising=False)
    assert _cache_dir() == Path.home() / ".cache" / "coco-pipe" / "report-assets"


def test_download_to_exception(tmp_path):
    target = tmp_path / "test.js"

    with (
        patch("urllib.request.urlopen", side_effect=URLError("Mock error")),
        pytest.raises(URLError),
    ):
        _download_to(target, "http://example.com")

    # temporary file should be unlinked
    assert not (tmp_path / "test.js.part").exists()


def test_vendor_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("COCO_PIPE_REPORT_ASSET_CACHE", str(tmp_path))

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value.read.return_value = b"test"

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        # First pass
        vendor_assets(force=False)
        assert mock_open.call_count == 3  # plotly, tailwind, pako

        # Second pass (exists, no force)
        mock_open.reset_mock()
        vendor_assets(force=False)
        assert mock_open.call_count == 0

        vendor_assets(force=True)
        assert mock_open.call_count == 3


def test_get_vendored_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("COCO_PIPE_REPORT_ASSET_CACHE", str(tmp_path))

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value.read.return_value = b"test_content"

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        # Not cached, will download
        contents = get_vendored_contents()
        assert contents["plotly"] == "test_content"
        assert mock_open.call_count == 3

        # Cached, won't download
        mock_open.reset_mock()
        contents2 = get_vendored_contents()
        assert contents2["plotly"] == "test_content"
        assert mock_open.call_count == 0
