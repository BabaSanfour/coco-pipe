"""
Dynamics Visualization: Velocity Embeddings
===========================================

Computes velocity vector fields on low-dimensional embeddings based on 
high-dimensional transition probabilities. Inspired by scVelo.

Functions
---------
compute_velocity_fields
    Main entry point. Computes V_emb from X and X_emb.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-08
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def compute_velocity_fields(X: np.ndarray, 
                            X_emb: np.ndarray, 
                            delta_t: int = 1, 
                            n_neighbors: int = 30,
                            sigma: float = 0.1) -> np.ndarray:
    """
    Compute velocity vectors in embedding space.

    Parameters
    ----------
    X : np.ndarray
        High-dimensional data (time-series). Assumed to be ordered by time.
    X_emb : np.ndarray
        Low-dimensional embedding.
    delta_t : int
        Time step lag. v_i = x_{i+dt} - x_i.
    n_neighbors : int
        Number of neighbors for projection.
    sigma : float
        Kernel width for transition probability.

    Returns
    -------
    V_emb : np.ndarray
        Velocity vectors in embedding space. Shape (n_samples, n_components).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(100, 10) # Time series
    >>> X_emb = np.random.rand(100, 2)
    >>> V = compute_velocity_fields(X, X_emb, delta_t=1)
    """
    n_samples = X.shape[0]
    
    # 1. Calculate High-Dimensional Velocity
    # v_i = x_{i+1} - x_i
    V_high = np.zeros_like(X)
    V_high[:-delta_t] = X[delta_t:] - X[:-delta_t]
    
    # 2. Find Neighbors (High D)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    
    V_emb = np.zeros_like(X_emb)
    
    # Precompute norms for cosine similarity
    # We want similarity between V_high[i] and (X[j] - X[i])
    
    # Iterate (could be vectorized but keeping explicit for clarity)
    for i in range(n_samples - delta_t):
        idx_neighbors = indices[i]
        
        # Displacements to neighbors
        D_high = X[idx_neighbors] - X[i]
        v_i = V_high[i]
        
        # Cosine correlation
        numer = D_high @ v_i
        norm_v = np.linalg.norm(v_i)
        norm_d = np.linalg.norm(D_high, axis=1)
        
        if norm_v < 1e-9:
             continue
             
        corr = numer / (norm_v * norm_d + 1e-9)
        
        # Transition Probabilities
        probs = np.exp(corr / sigma)
        probs /= np.sum(probs)
        
        # 3. Project to Low-D
        D_low = X_emb[idx_neighbors] - X_emb[i]
        V_emb[i] = probs @ D_low
        
    return V_emb
