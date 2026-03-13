#!/usr/bin/env python3
"""
Test script for coco_pipe EEG processing capabilities.

This script:
1. Downloads a small open EEG BIDS dataset
2. Loads it using coco_pipe's BIDS EEG loading functionality
3. Optionally runs dimensionality reduction or ML pipeline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import openneuro

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io import load_data

# Define constants
DATASET_NAME = "ds002778"  # Parkinson's disease EEG dataset
TEST_DATA_DIR = Path("test_data/bids_eeg")
BIDS_ROOT = TEST_DATA_DIR  # Was TEST_DATA_DIR / DATASET_NAME
SUBJECT = "pd6"  # Example subject
SESSION = "off"  # off medication session
TASK = "rest"  # resting state task


def download_dataset():
    """Download the example BIDS dataset if it doesn't exist."""
    if not BIDS_ROOT.exists() or not any(BIDS_ROOT.iterdir()):
        print(f"Downloading {DATASET_NAME} dataset to {BIDS_ROOT}...")
        # The include parameter specifies to only download data for a single subject
        openneuro.download(
            dataset=DATASET_NAME,
            target_dir=str(TEST_DATA_DIR),
            include=[f"sub-{SUBJECT}"],
        )
        print("Download complete!")


def load_eeg_data():
    """Load the EEG data into a DataContainer."""
    print(f"Loading EEG data for subject {SUBJECT}...")
    container = load_data(
        BIDS_ROOT,
        mode="bids",
        subjects=SUBJECT,
        session=SESSION,
        task=TASK,
        datatype="eeg",
        suffix="eeg",
        loading_mode="continuous",
    )
    print(f"Successfully loaded EEG data with shape: {container.X.shape}")
    return container


def run_dim_reduction(container, method="PCA", n_components=2):
    """Run dimensionality reduction on the EEG data."""
    print(f"Running {method} dimensionality reduction...")

    data = container.X
    if data.ndim == 3:
        data = data[0]
    elif data.ndim != 2:
        raise ValueError(f"Expected 2D or 3D EEG data, got shape {data.shape}.")

    # Reshape to (n_samples, n_features) if needed
    # For simplicity, we'll flatten across channels
    n_channels, n_times = data.shape
    data_reshaped = data.reshape(n_channels, -1).T

    reducer = DimReduction(method=method, n_components=n_components)
    reduced_data = reducer.fit_transform(data_reshaped)

    print(f"Original data shape: {data.shape}")
    print(f"Reshaped data shape: {data_reshaped.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")

    # Plot the results if 2D or 3D
    if n_components in (2, 3):
        plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
            plt.xlabel(f"{method} Dimension 1")
            plt.ylabel(f"{method} Dimension 2")
        else:  # 3D
            ax = plt.figure().add_subplot(111, projection="3d")
            ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.7
            )
            ax.set_xlabel(f"{method} Dimension 1")
            ax.set_ylabel(f"{method} Dimension 2")
            ax.set_zlabel(f"{method} Dimension 3")

        plt.title(f"{method} Reduction of EEG Data")
        plt.tight_layout()
        plt.savefig(f"{method.lower()}_reduction.png")
        plt.close()
        print(f"Plot saved as {method.lower()}_reduction.png")

    return reduced_data


def main():
    """Main function to run the entire pipeline."""
    # Step 1: Download the dataset
    download_dataset()

    # Step 2: Load the EEG data
    container = load_eeg_data()

    # Optional: Basic visualization of the EEG data
    preview = container.X[0] if container.X.ndim == 3 else container.X
    plt.figure(figsize=(12, 4))
    plt.plot(preview[0])
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("EEG preview (first channel)")
    plt.savefig("eeg_visualization.png")
    plt.close()
    print("EEG visualization saved as eeg_visualization.png")

    # Step 3: Run dimensionality reduction
    run_dim_reduction(container, method="PCA", n_components=2)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
