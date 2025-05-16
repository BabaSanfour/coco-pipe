#!/usr/bin/env python3
"""
Simple comparison of dimension reduction methods on MNIST and Fashion-MNIST datasets.

This script creates a visualization comparing PCA, t-SNE, UMAP, and PaCMAP
with a single parameter setting each for MNIST and Fashion-MNIST.
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from coco_pipe.dim_reduction import DimReducer

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
    """Load a dataset from test_data/dim_reduction."""
    data_path = DATA_DIR / f"{prefix}_data.npy"
    labels_path = DATA_DIR / f"{prefix}_labels.npy"
    
    if not data_path.exists() or not labels_path.exists():
        logger.error(f"Dataset files not found: {data_path} or {labels_path}")
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
    """Apply a dimension reduction method to the data."""
    method_display = method
    param_display = ""
    if kwargs:
        param_display = f"({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        
    logger.info(f"Applying {method} {param_display}...")
    
    # Initialize DimReducer
    reducer = DimReducer(
        method=method,
        n_components=n_components,
        save_path=None,
        **kwargs
    )
    
    # Apply dimension reduction with timing
    start_time = time.time()
    X_reduced = reducer.fit_transform(X)
    elapsed_time = time.time() - start_time
    
    logger.info(f"{method} completed in {elapsed_time:.2f} seconds")
    return X_reduced, elapsed_time, f"{method_display}{param_display}"

def visualize_comparison(dataset_name, X, y, class_names, n_samples=5000):
    """Create a visualization comparing different dimension reduction methods."""
    # Use a subset for faster computation
    if len(X) > n_samples:
        logger.info(f"Using {n_samples} samples from {len(X)} total")
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Methods to compare with their parameters
    methods = [
        ("PCA", {}),
        ("TSNE", {"perplexity": 30, "n_iter": 1000}),
        ("UMAP", {"n_neighbors": 15, "min_dist": 0.1}),
        ("PACMAP", {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2.0})
    ]
    
    # Run dimension reduction for each method
    results = []
    for method, params in methods:
        try:
            X_reduced, elapsed_time, method_name = apply_dim_reduction(
                X, method, n_components=2, **params
            )
            results.append((X_reduced, method_name, elapsed_time))
        except Exception as e:
            logger.error(f"Error applying {method}: {e}")
    
    if not results:
        logger.error("No successful dimension reduction methods")
        return
    
    # Create the visualization
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(n_methods * 4, 4))
    
    # Convert y to integers for coloring
    y_int = y.astype(int) if not np.issubdtype(y.dtype, np.integer) else y
    
    for i, (X_reduced, method_name, _) in enumerate(results):
        ax = axes[i] if n_methods > 1 else axes
        
        # Plot with different colors for each class
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1], 
            c=y_int, cmap='tab10', alpha=0.5, s=1
        )
        
        # Set titles
        ax.set_title(method_name, fontsize=12)
        
        # Remove ticks for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f"{dataset_name} - Dimension Reduction Comparison", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_simple_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    # Also save as SVG for high-quality reproduction
    svg_path = OUTPUT_DIR / f"{dataset_name.lower()}_simple_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()
    
    # Print performance comparison
    logger.info(f"\n{dataset_name} Performance Comparison:")
    for _, method_name, elapsed_time in results:
        logger.info(f"  {method_name}: {elapsed_time:.2f} seconds")

def main():
    """Run dimension reduction comparison on MNIST and Fashion-MNIST datasets."""
    # Process MNIST dataset
    X, y, class_names = load_dataset("mnist")
    if X is not None:
        visualize_comparison("MNIST", X, y, class_names)
    
    # Process Fashion-MNIST dataset
    X, y, class_names = load_dataset("fashion_mnist")
    if X is not None:
        visualize_comparison("Fashion-MNIST", X, y, class_names)

if __name__ == "__main__":
    main() 