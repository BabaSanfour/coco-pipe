#!/usr/bin/env python3
"""
Report Generation Demo
======================

Demonstrates the V2.1 Reporting API capabilities:
1.  Auto-Visualization (Raw Data Scroller)
2.  Auto-Provenance (Run Info)
3.  Quality Checks (Missingness, Flatlines)
4.  Scalability (Global Compressed Store)

Usage:
    python examples/demo_report.py
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from coco_pipe.io.structures import DataContainer
from coco_pipe.report import from_container

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_data() -> DataContainer:
    """Create a synthetic dataset with some quality issues."""
    n_samples = 100
    n_features = 50
    
    # 1. Signals (Sine waves with noise)
    t = np.linspace(0, 10, n_features)
    X = np.sin(t) + np.random.randn(n_samples, n_features) * 0.5
    
    # Inject Issue: Dead Channel (Flatline)
    X[:, 10] = 0.0 
    
    # Inject Issue: Outlier
    X[50, 0] = 100.0
    
    # Inject Issue: Missing Data
    X[20:30, 5] = np.nan
    
    # Metadata
    ids = [f"sub-{i:03d}" for i in range(n_samples)]
    coords = {
        "group": ["patient" if i % 2 == 0 else "control" for i in range(n_samples)],
        "age": np.random.randint(20, 80, n_samples).tolist()
    }
    
    return DataContainer(
        X=X,
        dims=("obs", "time"),
        ids=ids,
        coords=coords
    )

def main():
    logger.info("Generating dummy data...")
    container = generate_dummy_data()
    
    logger.info("Generating Report (V2.1)...")
    # This single line triggers:
    # - Quality Checks (will flag missing data, flatline, outlier)
    # - Provenance Capture (Git hash, OS, etc.)
    # - Interactive Visualization (Raw Data Scroller)
    report = from_container(
        container, 
        title="CoCo V2.1 Demo Report",
        config={"demo_mode": True, "notes": "Synthetic data with injected faults."}
    )
    
    output_path = Path("examples/outputs/demo_report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving report to {output_path}...")
    report.save(output_path)
    logger.info("Done! Open the report to see V2.1 features in action.")

if __name__ == "__main__":
    main()
