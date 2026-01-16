"""
Feature Attribution and Analysis
================================

Methods for explaining and interpreting embedding axes.

Functions
---------
compute_feature_importance
    Compute feature importance using perturbation or gradient (if available).
correlate_features
    Compute Spearman correlation between input features and embedding dimensions.

Author: Hamza Abdelhedi
Date: 2026-01-16
"""

import numpy as np
from typing import Optional, List, Union, Any
from scipy.stats import spearmanr

def correlate_features(X_orig: np.ndarray, 
                       X_emb: np.ndarray, 
                       feature_names: Optional[List[str]] = None) -> dict:
    """
    Compute correlation between original features and embedding dimensions.
    
    Helps interpret non-linear embeddings by identifying which original features
    covary most strongly with the reduced dimensions.

    Parameters
    ----------
    X_orig : np.ndarray
        Original high-dimensional data (N_samples, N_features).
    X_emb : np.ndarray
        Low-dimensional embedding (N_samples, N_components).
    feature_names : list, optional
        Names of the original features. If None, uses "Feature {i}".

    Returns
    -------
    correlations : dict
        Nested dictionary: {
            'Component 1': {'Feature A': 0.8, 'Feature B': -0.1, ...},
            'Component 2': ...
        }
        ordered by magnitude of correlation.
    """
    n_features = X_orig.shape[1]
    n_components = X_emb.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
    results = {}
    
    for j in range(n_components):
        comp_res = {}
        for i in range(n_features):
            # Spearman Rank Correlation (Robust to non-linear monotonic relationships)
            rho, _ = spearmanr(X_orig[:, i], X_emb[:, j])
            comp_res[feature_names[i]] = float(rho)
            
        # Sort by absolute correlation
        sorted_res = dict(sorted(comp_res.items(), key=lambda item: abs(item[1]), reverse=True))
        results[f"Component {j+1}"] = sorted_res
        
    return results


def perturbation_importance(model: Any, 
                            X: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            n_repeats: int = 5) -> dict:
    """
    Compute feature importance by shuffling features.

    Parameters
    ----------
    model : fitted estimator
        Must have a `transform` method.
    X : np.ndarray
        Data.
    feature_names : list, optional
        Names of features.
    n_repeats : int
        Number of shuffles per feature.

    Returns
    -------
    importances : dict
        Dictionary mapping feature names to importance scores. Sums to 1.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> model = PCA(n_components=2).fit(X)
    >>> scores = perturbation_importance(model, X, feature_names=['A', 'B'])
    """
    if not hasattr(model, 'transform'):
        raise ValueError("Model must have a transform method.")
        
    original_emb = model.transform(X)
    n_features = X.shape[1]
    
    scores = np.zeros(n_features)
    
    for f in range(n_features):
        feature_score = 0
        for r in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, f])
            
            emb_permuted = model.transform(X_permuted)
            
            dist = np.mean((original_emb - emb_permuted) ** 2)
            feature_score += dist
            
        scores[f] = feature_score / n_repeats
        
    # Normalize
    scores /= np.sum(scores)
    
    if feature_names is None:
        return {f"Feature {i}": s for i, s in enumerate(scores)}
    
    return {n: s for n, s in zip(feature_names, scores)}


def compute_feature_importance(model: Any, 
                               X: np.ndarray, 
                               method: str = 'perturbation', 
                               **kwargs) -> dict:
    """
    Compute feature importance for a given dimensionality reduction model.

    Parameters
    ----------
    model : fitted estimator
        The dimensionality reduction model (e.g., PCA, UMAP, or TopologicalAE).
    X : np.ndarray
        Input data.
    method : {'perturbation', 'gradient'}, default='perturbation'
        Method to calculate importance.
        - 'perturbation': Model-agnostic. Shuffles features and measures embedding displacement.
        - 'gradient': Model-specific. Computes saliency maps (requires PyTorch model).
    **kwargs : dict
        Additional arguments passed to the specific importance function 
        (e.g., `n_repeats` for perturbation, `feature_names`).

    Returns
    -------
    importances : dict
        Dictionary feature_name -> importance_score.
    
    Examples
    --------
    >>> scores = compute_feature_importance(model, X, method='perturbation')
    """
    if method == 'perturbation':
        return perturbation_importance(model, X, **kwargs)
    elif method == 'gradient':
        if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            # Assume TopoAE or similar PyTorch structure
            return gradient_importance(model, X, **kwargs)
        raise NotImplementedError("Gradient method requires a supported PyTorch model (TopologicalAEReducer).")
    else:
        raise ValueError(f"Unknown method {method}")


def gradient_importance(wrapper: Any, X: np.ndarray, **kwargs) -> dict:
    """
    Compute gradient-based feature importance (Saliency).

    Calculates the mean absolute gradient of the embedding sum with respect 
    to the input features. This estimates how sensitive the embedding is to 
    changes in each feature.

    Parameters
    ----------
    wrapper : Any
        Fitted reducer wrapper containing the PyTorch model (e.g., TopologicalAEReducer).
    X : np.ndarray
        Input data.
    **kwargs : dict
        - feature_names: List of strings.

    Returns
    -------
    importances : dict
        Dictionary feature_name -> importance_score.
    """
    import torch
    
    model = wrapper.model
    device = next(model.parameters()).device
    
    # Check dimensions
    if X.ndim == 2:
        # (N, Features)
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)

    Z = model.encoder(X_tensor)
    
    target = Z.sum()
    target.backward()
    
    grads = X_tensor.grad 
    
    # Mean absolute gradient per feature
    if X.ndim == 2:
        mean_grads = torch.mean(torch.abs(grads), dim=0) # (Features,)
    else:
        mean_grads = torch.mean(torch.abs(grads), dim=0) # (Ch, Time)

    scores = mean_grads.detach().cpu().numpy()
    
    if np.sum(scores) > 0:
        scores /= np.sum(scores) # Normalize
    
    feature_names = kwargs.get('feature_names')
    if feature_names is None:
        if scores.ndim == 1:
            return {f"Feature {i}": s for i, s in enumerate(scores)}
        else:
            # Return raw array for complex shapes if no names
            return {"importance_matrix": scores}
            
    return {n: s for n, s in zip(feature_names, scores.flatten())}
