
import pytest
from coco_pipe.report.provenance import get_environment_info

def test_get_environment_info():
    info = get_environment_info()
    assert isinstance(info, dict)
    assert "timestamp_utc" in info
    assert "git_hash" in info
    assert "python_version" in info
    assert "versions" in info
    assert "numpy" in info["versions"]
    assert "pandas" in info["versions"]
