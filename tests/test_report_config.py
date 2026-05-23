from unittest.mock import patch

from coco_pipe.report.config import ProvenanceConfig, ReportConfig


def test_provenance_config_defaults():
    prov = ProvenanceConfig()
    assert prov.source == "Unknown"
    assert prov.git_hash == "Unknown"
    assert "UTC" in prov.timestamp_utc
    assert prov.command is None
    assert prov.python_version is None
    assert prov.os_platform is None
    assert prov.coco_pipe_version == "Unknown"
    assert prov.versions == {}


@patch("coco_pipe.report.config.get_environment_info")
def test_provenance_config_from_env(mock_get_env):
    mock_get_env.return_value = {
        "timestamp_utc": "2023-01-01 12:00:00 UTC",
        "os_platform": "Linux",
        "python_version": "3.10.0",
        "command": "pytest",
        "git_hash": "deadbeef",
        "coco_pipe_version": "0.1.0",
        "versions": {"numpy": "1.24.0"},
    }

    prov = ProvenanceConfig.from_env(source="BIDS Dataset", custom_field="Custom Value")

    assert prov.source == "BIDS Dataset"
    assert prov.timestamp_utc == "2023-01-01 12:00:00 UTC"
    assert prov.os_platform == "Linux"
    assert prov.python_version == "3.10.0"
    assert prov.command == "pytest"
    assert prov.git_hash == "deadbeef"
    assert prov.coco_pipe_version == "0.1.0"
    assert prov.versions == {"numpy": "1.24.0"}
    assert prov.custom_field == "Custom Value"  # extra="allow" handles this


@patch("coco_pipe.report.config.get_environment_info")
def test_report_config_defaults(mock_get_env):
    # Setup mock to return empty env to test default factory execution
    mock_get_env.return_value = {"git_hash": "mocked_hash"}

    report = ReportConfig()
    assert report.title.startswith("CoCo Analysis Report (")
    assert report.author is None
    assert report.description is None
    assert report.run_params == {}

    # Provenance should be automatically generated via from_env factory
    assert isinstance(report.provenance, ProvenanceConfig)
    assert report.provenance.git_hash == "mocked_hash"
    assert report.provenance.source == "Unknown"


def test_report_config_custom():
    prov = ProvenanceConfig(source="Manual")
    report = ReportConfig(
        title="Custom Report",
        author="John Doe",
        description="A nice report.",
        provenance=prov,
        run_params={"alpha": 0.05},
        extra_key="extra_value",
    )

    assert report.title == "Custom Report"
    assert report.author == "John Doe"
    assert report.description == "A nice report."
    assert report.provenance.source == "Manual"
    assert report.run_params == {"alpha": 0.05}
    assert report.extra_key == "extra_value"
