"""
Configuration Schemas for Report
================================

Pydantic models for validting report configuration and metadata.

Classes
-------
ProvenanceConfig
    Capture environment and execution metadata.
ReportConfig
    Main configuration for the report generation.

Author: Antigravity
Date: 2026-01-17
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

class ProvenanceConfig(BaseModel):
    """Configuration for execution provenance."""
    source: str = Field("Unknown", description="Source of the data (BIDS, Tabular, etc.)")
    git_hash: str = Field("Unknown", description="Git commit hash of the code.")
    timestamp_utc: str = Field(..., description="Execution timestamp.")
    command: Optional[str] = Field(None, description="Command line arguments used.")
    python_version: Optional[str] = Field(None, description="Python version.")
    os_platform: Optional[str] = Field(None, description="Operating System.")
    versions: Dict[str, str] = Field(default_factory=dict, description="Package versions.")
    
    class Config:
        extra = "allow" # Allow extra fields like 'root', 'task', etc.

class ReportConfig(BaseModel):
    """
    Configuration for the Report object.
    """
    title: str = Field("CoCo Analysis Report", description="Title of the report.")
    author: Optional[str] = Field(None, description="Author of the report.")
    description: Optional[str] = Field(None, description="Brief description.")
    
    # Nested provenance info
    provenance: Optional[ProvenanceConfig] = Field(None, description="Execution metadata.")
    
    # Generic config storage for run parameters
    run_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters used in the analysis run.")

    class Config:
        extra = "allow"
