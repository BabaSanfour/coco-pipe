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

from coco_pipe.dim_reduction import DimReductionPipeline

# Import coco_pipe functionality
from coco_pipe.io.load import load

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
    """Load the EEG data using coco_pipe's load function."""
    print(f"Loading EEG data for subject {SUBJECT}...")
    # Load the data using our load function from coco_pipe
    raw = load(
        type="eeg",
        data_path=str(BIDS_ROOT),
        subjects=SUBJECT,
        session=SESSION,
        task=TASK,
        datatype="eeg",
        suffix="eeg",
        verbose=True,  # Show detailed loading information
    )
    print(f"Successfully loaded EEG data: {raw}")
    return raw


def run_dim_reduction(raw, method="PCA", n_components=2):
    """Run dimensionality reduction on the EEG data."""
    print(f"Running {method} dimensionality reduction...")

    # Extract the EEG data as a numpy array
    data = raw.get_data()

    # Reshape to (n_samples, n_features) if needed
    # For simplicity, we'll flatten across channels
    n_channels, n_times = data.shape
    data_reshaped = data.reshape(n_channels, -1).T

    # Initialize the dimensionality reduction pipeline
    DimReductionPipeline(
        type="eeg",
        method=method,
        data_path=str(BIDS_ROOT),
        task=TASK,
        run=SESSION,  # Using session as run since that's what our dataset has
        n_components=n_components,
    )

    # For this quick test, we'll create a small wrapper to avoid needing to save files
    # This is a simplified version compared to the full pipeline
    from coco_pipe.dim_reduction.reducer import DimReducer

    reducer = DimReducer(method=method, n_components=n_components)
    reduced_data = reducer.fit_transform(data_reshaped)

    print(f"Original data shape: {data.shape}")
    print(f"Reshaped data shape: {data_reshaped.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")

    # Plot the results if 2D or 3D
    if n_components in (2, 3):
        plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
            plt.xlabel(f"{method} Component 1")
            plt.ylabel(f"{method} Component 2")
        else:  # 3D
            ax = plt.figure().add_subplot(111, projection="3d")
            ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.7
            )
            ax.set_xlabel(f"{method} Component 1")
            ax.set_ylabel(f"{method} Component 2")
            ax.set_zlabel(f"{method} Component 3")

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
    raw = load_eeg_data()

    # Optional: Basic visualization of the EEG data
    raw.plot(duration=10, n_channels=10, scalings="auto", show=False)
    plt.savefig("eeg_visualization.png")
    plt.close()
    print("EEG visualization saved as eeg_visualization.png")

    # Step 3: Run dimensionality reduction
    run_dim_reduction(raw, method="PCA", n_components=2)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
