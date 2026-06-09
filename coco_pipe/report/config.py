"""
Pydantic schemas for report configuration and provenance metadata.
"""

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from coco_pipe.utils import get_environment_info


class ProvenanceConfig(BaseModel):
    """Runtime provenance attached to generated reports."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(
        "Unknown", description="Source of the data (BIDS, Tabular, etc.)"
    )
    git_hash: str = Field("Unknown", description="Git commit hash of the code.")
    timestamp_utc: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        description="Execution timestamp.",
    )
    command: str | None = Field(None, description="Command line arguments used.")
    python_version: str | None = Field(None, description="Python version.")
    os_platform: str | None = Field(None, description="Operating system.")
    coco_pipe_version: str = Field(
        "Unknown", description="Installed CoCo Pipe package version."
    )
    versions: dict[str, str] = Field(
        default_factory=dict, description="Package versions."
    )

    @classmethod
    def from_env(cls, source: str = "Unknown", **kwargs: Any) -> "ProvenanceConfig":
        """
        Build a ProvenanceConfig by capturing the current runtime environment.

        Parameters
        ----------
        source : str
            Description of the data source.
        **kwargs : Any
            Additional metadata to override or append to the environment info.

        Returns
        -------
        ProvenanceConfig
            A new instance populated with runtime metrics.
        """
        env_info = get_environment_info()
        data = {"source": source, **env_info, **kwargs}
        return cls(**data)


class ReportConfig(BaseModel):
    """User-facing configuration attached to a report."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(
        default_factory=lambda: (
            f"CoCo Analysis Report "
            f"({dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')})"
        ),
        description="Title of the report.",
    )
    author: str | None = Field(None, description="Author of the report.")
    description: str | None = Field(None, description="Brief description.")
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig.from_env, description="Execution metadata."
    )
    run_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used in the analysis run."
    )
