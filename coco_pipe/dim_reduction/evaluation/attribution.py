"""
Feature Attribution
===================

Methods for explaining embedding axes using feature importance.

Functions
---------
compute_feature_importance
    Compute feature importance using perturbation or gradient (if available).

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-08
"""

import numpy as np
from typing import Optional, List, Union, Any

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
            
            # Measure displacement
            # MSE between original and permuted embedding
            # High displacement = Feature was important for structure
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
    
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    
    # Forward pass
    Z = model.encoder(X_tensor)
    
    # Compute gradients of the sum of embeddings (Saliency)
    n_features = X.shape[1]
    importances = torch.zeros(n_features).to(device)
    
    # We can just backward on the sum of all embeddings
    # grad = d(sum(Z)) / dX
    target = Z.sum()
    target.backward()
    
    grads = X_tensor.grad # (N, Features)
    
    # Mean absolute gradient per feature
    mean_grads = torch.mean(torch.abs(grads), dim=0)
    
    scores = mean_grads.detach().cpu().numpy()
    scores /= np.sum(scores) # Normalize
    
    feature_names = kwargs.get('feature_names')
    if feature_names is None:
        return {f"Feature {i}": s for i, s in enumerate(scores)}
    return {n: s for n, s in zip(feature_names, scores)}
