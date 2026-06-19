from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest

from coco_pipe.utils import (
    get_environment_info,
    get_git_revision_hash,
    get_package_version,
    import_optional_dependency,
    resolve_n_jobs,
    run_task_batch,
    slug,
    stable_hash,
)


def test_stable_hash():
    assert stable_hash({"b": 2, "a": 1}, length=16) == stable_hash(
        {"a": 1, "b": 2}, length=16
    )
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    with pytest.raises(ValueError, match="between 1 and 64"):
        stable_hash({}, length=0)


def test_import_optional_dependency():
    # Test successful import
    assert import_optional_dependency(lambda: "module", "Feature", "lib") == "module"

    # Test ImportError
    with pytest.raises(ImportError, match="demo-lib is required for DemoReducer"):
        import_optional_dependency(
            lambda: (_ for _ in ()).throw(ImportError("missing")),
            feature="DemoReducer",
            dependency="demo-lib",
            install_hint="pip install demo-lib",
        )

    # Test RuntimeError
    with pytest.raises(
        RuntimeError, match="demo-lib failed to initialize for DemoReducer"
    ):
        import_optional_dependency(
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            feature="DemoReducer",
            dependency="demo-lib",
        )


@patch("subprocess.run")
def test_get_git_revision_hash_success(mock_run):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "1a2b3c4\n"
    mock_run.return_value = mock_result

    assert get_git_revision_hash() == "1a2b3c4"
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_get_git_revision_hash_failure(mock_run):
    # Test non-zero exit
    mock_result = MagicMock()
    mock_result.returncode = 128
    mock_run.return_value = mock_result
    assert get_git_revision_hash() == "Unknown"

    # Test exception (e.g. git not found)
    mock_run.side_effect = FileNotFoundError()
    assert get_git_revision_hash() == "Unknown"


@patch("importlib.metadata.version")
def test_get_package_version_success(mock_version):
    mock_version.return_value = "1.2.3"
    assert get_package_version("pandas") == "1.2.3"


@patch("importlib.metadata.version")
def test_get_package_version_missing(mock_version):
    mock_version.side_effect = PackageNotFoundError()
    assert get_package_version("missing-lib") == "Unknown"


@patch("coco_pipe.utils.get_git_revision_hash")
@patch("coco_pipe.utils.get_package_version")
def test_get_environment_info(mock_get_version, mock_get_git):
    mock_get_git.return_value = "deadbeef"

    # Return "v1" for coco-pipe, and version string for others
    def version_side_effect(pkg):
        return f"{pkg}_v1"

    mock_get_version.side_effect = version_side_effect

    custom_packages = {"pkg1": "dist1", "pkg2": "dist2"}

    info = get_environment_info(cwd=".", packages=custom_packages)

    assert "timestamp_utc" in info
    assert "UTC" in info["timestamp_utc"]
    assert "os_platform" in info
    assert "python_version" in info
    assert "command" in info
    assert info["git_hash"] == "deadbeef"
    assert info["coco_pipe_version"] == "coco-pipe_v1"

    versions = info["versions"]
    assert versions["pkg1"] == "dist1_v1"
    assert versions["pkg2"] == "dist2_v1"


def test_slug():
    # Regular slugification
    assert slug("hello world") == "hello-world"
    assert slug("hello_world-1.0=2") == "hello_world-1.0=2"
    # Strips leading/trailing punctuation and handles empty case
    assert slug("!!!") == "none"
    assert slug("") == "none"
    assert slug("a" * 100, max_len=10) == "a" * 10


def test_resolve_n_jobs():
    import os

    # n_jobs = -1
    assert resolve_n_jobs(-1) == max(os.cpu_count() or 1, 1)
    # n_jobs >= 1
    assert resolve_n_jobs(4) == 4
    # ValueError on < 1 (except -1)
    with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
        resolve_n_jobs(0)
    with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
        resolve_n_jobs(-2)


def test_run_task_batch():
    # Empty tasks
    assert run_task_batch([], lambda x: x, 4) == []
    # Sequential execution (max_workers = 1)
    assert run_task_batch([1, 2, 3], lambda x: x * 2, 1) == [2, 4, 6]
    # Parallel execution (max_workers > 1)
    assert run_task_batch([1, 2, 3], lambda x: x * 2, 2) == [2, 4, 6]
