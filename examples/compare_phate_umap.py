#!/usr/bin/env python3
"""
Compare PHATE and UMAP dimensionality reduction methods on MNIST and Fashion-MNIST datasets.

This script creates visualizations comparing PHATE and UMAP with different parameter settings,
showing how they perform on the same datasets.
"""
import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import phate
import umap
import warnings

# Suppress the deprecation warning
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

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

def apply_phate(X, param_str, knn=5, t='auto'):
    """
    Apply PHATE dimensionality reduction.
    
    Args:
        X (np.ndarray): Input data
        param_str (str): Parameter string for labeling
        knn (int): Number of nearest neighbors
        t (str or int): Diffusion time scale parameter
        
    Returns:
        tuple: (X_reduced, elapsed_time)
    """
    logger.info(f"Applying PHATE with parameters: {param_str}...")
    
    # Initialize PHATE
    phate_operator = phate.PHATE(
        n_components=2,
        knn=knn,
        t=t
    )
    
    # Apply PHATE with timing
    start_time = time.time()
    X_reduced = phate_operator.fit_transform(X)
    elapsed_time = time.time() - start_time
    
    logger.info(f"PHATE completed in {elapsed_time:.2f} seconds")
    return X_reduced, elapsed_time

def apply_umap(X, param_str, n_neighbors=15, min_dist=0.1):
    """
    Apply UMAP dimensionality reduction.
    
    Args:
        X (np.ndarray): Input data
        param_str (str): Parameter string for labeling
        n_neighbors (int): Number of nearest neighbors
        min_dist (float): Minimum distance in the embedding
        
    Returns:
        tuple: (X_reduced, elapsed_time)
    """
    logger.info(f"Applying UMAP with parameters: {param_str}...")
    
    # Initialize UMAP - set random_state=None to allow n_jobs parallelism
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=None,  # Allow parallelism
        n_jobs=-1  # Use all available cores
    )
    
    # Apply UMAP with timing
    start_time = time.time()
    X_reduced = umap_reducer.fit_transform(X)
    elapsed_time = time.time() - start_time
    
    logger.info(f"UMAP completed in {elapsed_time:.2f} seconds")
    return X_reduced, elapsed_time

def visualize_comparison(results, y, class_names, dataset_name):
    """
    Create a visualization comparing PHATE and UMAP results with different parameters.
    
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
    fig, axes = plt.subplots(n_params, n_methods, figsize=(n_methods * 5, n_params * 5))
    
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
                c=y_int, cmap='tab10', alpha=0.5, s=2
            )
            
            # Set titles
            if row == 0:
                ax.set_title(f"{method}", fontsize=14, pad=10)
            
            # Add parameter information
            ax.set_xlabel(param_str, fontsize=10)
            
            # Remove ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Create a legend
    handles, labels = scatter.legend_elements(prop="colors")
    legend = fig.legend(handles, class_names, loc="upper right", title="Classes",
                       bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle(f"PHATE vs UMAP - {dataset_name}", fontsize=18, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_phate_umap_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison visualization saved to {output_path}")
    
    # Also save as SVG for high-quality reproduction
    svg_path = OUTPUT_DIR / f"{dataset_name.lower()}_phate_umap_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()
    
    # Create a performance comparison table
    fig, ax = plt.subplots(figsize=(10, 4))
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
    table_path = OUTPUT_DIR / f"{dataset_name.lower()}_phate_umap_performance.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance table saved to {table_path}")
    
    plt.close()

def main():
    """
    Main function to run the comparison between PHATE and UMAP.
    """
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    n_samples = 5000  # Use a smaller subset for faster computation
    
    # Define parameter sets for each method
    phate_params = [
        (5, 'auto', "knn=5"),
        (10, 'auto', "knn=10"),
        (20, 'auto', "knn=20")
    ]
    
    umap_params = [
        (10, 0.1, "n_neighbors=10"),
        (20, 0.1, "n_neighbors=20"),
        (40, 0.1, "n_neighbors=40")
    ]
    
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
        results = {
            "PHATE": [],
            "UMAP": []
        }
        
        # Apply PHATE with different parameters
        for knn, t, param_str in phate_params:
            try:
                X_reduced, elapsed_time = apply_phate(X, param_str, knn=knn, t=t)
                results["PHATE"].append((X_reduced, param_str, elapsed_time))
            except Exception as e:
                logger.error(f"Error applying PHATE with {param_str} to {dataset_name}: {e}")
        
        # Apply UMAP with different parameters
        for n_neighbors, min_dist, param_str in umap_params:
            try:
                X_reduced, elapsed_time = apply_umap(X, param_str, n_neighbors=n_neighbors, min_dist=min_dist)
                results["UMAP"].append((X_reduced, param_str, elapsed_time))
            except Exception as e:
                logger.error(f"Error applying UMAP with {param_str} to {dataset_name}: {e}")
        
        # Create visualizations
        if all(results.values()):
            visualize_comparison(results, y, class_names, dataset_name)
            
            # Print performance comparison
            logger.info(f"\n{dataset_name} Performance Comparison:")
            for method, method_results in results.items():
                for _, param_str, elapsed_time in method_results:
                    logger.info(f"  {method} ({param_str}): {elapsed_time:.2f} seconds")
        else:
            logger.error(f"No successful dimension reduction methods for {dataset_name}")

if __name__ == "__main__":
    main() 