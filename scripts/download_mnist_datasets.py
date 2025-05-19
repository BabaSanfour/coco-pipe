#!/usr/bin/env python3
"""
Download MNIST and Fashion-MNIST datasets and save them in the test_data/dim_reduction folder.
These datasets will be used for comparing different dimension reduction methods.
"""
import os
import logging
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Define the output directory
OUTPUT_DIR = Path("test_data/dim_reduction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_and_save_dataset(dataset_name, file_prefix):
    """
    Download a dataset from OpenML and save it to the output directory.
    
    Args:
        dataset_name (str): Name of the dataset to download
        file_prefix (str): Prefix for the saved files
    """
    logger.info(f"Downloading {dataset_name} dataset...")
    
    # Download dataset
    try:
        data = fetch_openml(dataset_name, as_frame=False)
        X = data.data
        y = data.target
        
        # Save data and target as NPY files
        data_path = OUTPUT_DIR / f"{file_prefix}_data.npy"
        labels_path = OUTPUT_DIR / f"{file_prefix}_labels.npy"
        
        logger.info(f"Saving {dataset_name} dataset (shape {X.shape}) to {data_path}")
        np.save(data_path, X)
        
        logger.info(f"Saving {dataset_name} labels (shape {y.shape}) to {labels_path}")
        np.save(labels_path, y)
        
        # Create a metadata file
        meta_path = OUTPUT_DIR / f"{file_prefix}_meta.json"
        with open(meta_path, 'w') as f:
            f.write('{\n')
            f.write(f'  "dataset": "{dataset_name}",\n')
            f.write(f'  "n_samples": {X.shape[0]},\n')
            f.write(f'  "n_features": {X.shape[1]},\n')
            f.write(f'  "data_file": "{data_path.name}",\n')
            f.write(f'  "labels_file": "{labels_path.name}"\n')
            f.write('}\n')
        
        logger.info(f"Saved metadata to {meta_path}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error downloading {dataset_name}: {e}")
        return None, None

def main():
    """Download and save MNIST and Fashion-MNIST datasets."""
    logger.info("Starting download of datasets...")
    
    # Download MNIST
    mnist_X, mnist_y = download_and_save_dataset("mnist_784", "mnist")
    
    # Download Fashion-MNIST
    fashion_X, fashion_y = download_and_save_dataset("Fashion-MNIST", "fashion_mnist")
    
    if mnist_X is not None and fashion_X is not None:
        logger.info("Successfully downloaded all datasets.")
    else:
        logger.warning("Some datasets could not be downloaded. Check the logs for details.")

if __name__ == "__main__":
    main() 