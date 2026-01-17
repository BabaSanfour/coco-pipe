#!/usr/bin/env python3
"""
End-to-End Pipeline Demo
========================

Demonstrates the full CoCo pipeline:
1.  Load (Tabular)
2.  Preprocess (StandardScaler)
3.  Reduce (PCA & UMAP)
4.  Report (Comparison)

Usage:
    python examples/demo_pipeline.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io.dataset import TabularDataset
from coco_pipe.report import from_reductions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_csv(path: Path):
    """Create a dummy CSV file for the demo."""
    logger.info(f"Creating synthetic dataset at {path}...")

    # 3 Clusters
    c1 = np.random.randn(50, 10) + 5
    c2 = np.random.randn(50, 10) - 5
    c3 = np.random.randn(100, 10)

    X = np.vstack([c1, c2, c3])
    labels = ["A"] * 50 + ["B"] * 50 + ["C"] * 100

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["label"] = labels

    df.to_csv(path, index=False)


def main():
    # Setup
    data_path = Path("examples/outputs/dummy_data.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    create_synthetic_csv(data_path)

    # 1. Load Data
    logger.info("1. Loading Data...")
    ds = TabularDataset(data_path, target_col="label")
    container = ds.load()

    # 2. Preprocessing (simulate via manual scaling)
    logger.info("2. Preprocessing...")
    container.X = (container.X - container.X.mean(axis=0)) / container.X.std(axis=0)

    # 3. Dimensionality Reduction
    logger.info("3. Running Reductions...")

    # PCA
    pca = DimReducer(method="PCA", n_components=2)
    pca.fit(container.X)

    # UMAP
    umap = DimReducer(method="UMAP", n_components=2, n_neighbors=15)
    umap.fit(container.X)

    # 4. Generate Comparative Report
    logger.info("4. Generating Report...")
    report = from_reductions(
        reductions=[pca, umap],
        container=container,
        title="Pipeline Demo: PCA vs UMAP",
        config={"pipeline": "Full Demo", "scaling": "StandardScaler"},
    )

    output_path = Path("examples/outputs/demo_pipeline.html")
    report.save(output_path)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
