#!/usr/bin/env python3
"""
Compare different dimension reduction methods on MNIST and Fashion-MNIST datasets.

This script creates grid visualizations comparing different dimension reduction methods
(PCA, t-SNE, UMAP, and PaCMAP) with different parameter settings for MNIST and Fashion-MNIST.
"""
import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from coco_pipe.dim_reduction import DimReducer
from coco_pipe.dim_reduction.config import METHODS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Define directories
DATA_DIR = Path("test_data/dim_reduction")
OUTPUT_DIR = Path("examples/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define class names for datasets
MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_dataset(prefix):
    """
    Load a dataset from test_data/dim_reduction.
    
    Args:
        prefix (str): Dataset prefix (mnist or fashion_mnist)
        
    Returns:
        tuple: (X, y, class_names)
    """
    data_path = DATA_DIR / f"{prefix}_data.npy"
    labels_path = DATA_DIR / f"{prefix}_labels.npy"
    
    if not data_path.exists() or not labels_path.exists():
        logger.error(f"Dataset files not found: {data_path} or {labels_path}")
        logger.error("Run scripts/download_mnist_datasets.py first to download the datasets")
        return None, None, None
    
    logger.info(f"Loading {prefix} dataset...")
    X = np.load(data_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True)
    
    # Assign class names based on dataset
    if prefix == "mnist":
        class_names = MNIST_CLASSES
    elif prefix == "fashion_mnist":
        class_names = FASHION_MNIST_CLASSES
    else:
        class_names = [str(i) for i in range(len(np.unique(y)))]
    
    return X, y, class_names

def apply_dim_reduction(X, method, n_components=2, **kwargs):
    """
    Apply a dimension reduction method to the data.
    
    Args:
        X (np.ndarray): Input data
        method (str): Dimension reduction method (PCA, TSNE, UMAP, PACMAP)
        n_components (int): Number of components in the reduced space
        **kwargs: Additional keyword arguments for the reducer
        
    Returns:
        tuple: (X_reduced, elapsed_time)
    """
    logger.info(f"Applying {method} with params {kwargs} to reduce to {n_components} dimensions...")
    
    # Initialize DimReducer
    reducer = DimReducer(
        method=method,
        n_components=n_components,
        save_path=None,  # Don't save the reducer
        **kwargs
    )
    
    # Apply dimension reduction with timing
    start_time = time.time()
    X_reduced = reducer.fit_transform(X)
    elapsed_time = time.time() - start_time
    
    logger.info(f"{method} completed in {elapsed_time:.2f} seconds")
    return X_reduced, elapsed_time

def visualize_grid(results, y, class_names, dataset_name):
    """
    Create a grid visualization of dimension reduction results with different parameters.
    
    Args:
        results (dict): Dictionary mapping method names to lists of 
                        (X_reduced, param_str, elapsed_time) tuples
        y (np.ndarray): Labels
        class_names (list): Names of classes
        dataset_name (str): Name of the dataset
    """
    # Get methods and number of parameter variations for each
    methods = list(results.keys())
    n_methods = len(methods)
    n_params = max(len(results[method]) for method in methods)
    
    # Create figure with a grid of subplots
    fig, axes = plt.subplots(n_params, n_methods, figsize=(n_methods * 4, n_params * 4))
    
    # If only one row, make axes 2D
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    # Convert y to integers for coloring
    y_int = y.astype(int) if not np.issubdtype(y.dtype, np.integer) else y
    
    # Loop through methods (columns)
    for col, method in enumerate(methods):
        method_results = results[method]
        
        # Loop through parameters for this method (rows)
        for row, (X_reduced, param_str, _) in enumerate(method_results):
            ax = axes[row, col]
            
            # Plot with different colors for each class
            scatter = ax.scatter(
                X_reduced[:, 0], X_reduced[:, 1], 
                c=y_int, cmap='tab10', alpha=0.5, s=1
            )
            
            # Set titles
            if row == 0:
                ax.set_title(f"{method}", fontsize=14, pad=10)
            
            # Add parameter information
            ax.set_xlabel(param_str, fontsize=10)
            
            # Remove ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Fill empty subplots if any
    for method in methods:
        n_results = len(results[method])
        for row in range(n_results, n_params):
            ax = axes[row, methods.index(method)]
            ax.axis('off')
    
    plt.suptitle(f"Dimension Reduction Comparison - {dataset_name}", fontsize=18, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_grid_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Grid visualization saved to {output_path}")
    
    # Also save as SVG for high-quality reproduction
    svg_path = OUTPUT_DIR / f"{dataset_name.lower()}_grid_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()
    
    # Create a performance comparison table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the table
    table_data = []
    for method in methods:
        for X_reduced, param_str, elapsed_time in results[method]:
            table_data.append([method, param_str, f"{elapsed_time:.2f}s"])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Parameters", "Computation Time"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title(f"Performance Comparison - {dataset_name}", fontsize=16, pad=20)
    
    # Save the table
    table_path = OUTPUT_DIR / f"{dataset_name.lower()}_performance_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    """
    Main function to run the dimension reduction comparison.
    
    This function:
    1. Loads both MNIST and Fashion-MNIST datasets
    2. Applies dimension reduction with multiple methods and parameters
    3. Visualizes the results in a grid layout
    """
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    n_components = 2
    n_samples = 5000  # Use a smaller subset for faster computation
    
    # Define methods and parameters to test
    method_params = {
        "PCA": [
            ({}, "Default"),
            ({"svd_solver": "randomized"}, "Randomized SVD"),
            ({"whiten": True}, "Whitened")
        ],
        "TSNE": [
            ({"perplexity": 10, "n_iter": 1000}, "perplexity=10"),
            ({"perplexity": 30, "n_iter": 1000}, "perplexity=30"),
            ({"perplexity": 50, "n_iter": 1000}, "perplexity=50")
        ],
        "UMAP": [
            ({"n_neighbors": 10, "min_dist": 0.1}, "n_neighbors=10"),
            ({"n_neighbors": 20, "min_dist": 0.1}, "n_neighbors=20"),
            ({"n_neighbors": 40, "min_dist": 0.1}, "n_neighbors=40")
        ],
        "PACMAP": [
            ({"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2.0}, "n_neighbors=10"),
            ({"n_neighbors": 20, "MN_ratio": 0.5, "FP_ratio": 2.0}, "n_neighbors=20"),
            ({"n_neighbors": 40, "MN_ratio": 0.5, "FP_ratio": 2.0}, "n_neighbors=40")
        ]
    }
    
    # Process both datasets
    for dataset_name, prefix in [("MNIST", "mnist"), ("Fashion-MNIST", "fashion_mnist")]:
        # Load dataset
        X, y, class_names = load_dataset(prefix)
        
        if X is None:
            continue
        
        # Use a subset of samples for faster computation
        if len(X) > n_samples:
            logger.info(f"Using {n_samples} samples from {len(X)} total")
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Results dictionary to store all outputs
        results = {method: [] for method in method_params}
        
        # Apply dimension reduction with each method and parameter set
        for method, param_sets in method_params.items():
            for params, param_str in param_sets:
                try:
                    X_reduced, elapsed_time = apply_dim_reduction(
                        X, 
                        method, 
                        n_components=n_components,
                        **params
                    )
                    results[method].append((X_reduced, param_str, elapsed_time))
                    
                except Exception as e:
                    logger.error(f"Error applying {method} with {param_str} to {dataset_name}: {e}")
        
        # Visualize results
        if any(results.values()):
            visualize_grid(results, y, class_names, dataset_name)
            
            # Print performance comparison
            logger.info(f"\n{dataset_name} Performance Comparison:")
            for method, method_results in results.items():
                for _, param_str, elapsed_time in method_results:
                    logger.info(f"  {method} ({param_str}): {elapsed_time:.2f} seconds")
        else:
            logger.error(f"No successful dimension reduction methods for {dataset_name}")

if __name__ == "__main__":
    main() 