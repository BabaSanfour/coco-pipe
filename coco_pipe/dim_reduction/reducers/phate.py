import phate
import numpy as np
from typing import Optional
from .base import BaseReducer

class PHATEReducer(BaseReducer):
    """
    PHATE dimensionality reducer.
    
    This class applies Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
    for dimensionality reduction. PHATE is particularly effective for visualizing trajectory structures
    in high-dimensional data.
    
    Parameters:
        n_components (int): The target number of dimensions for projection, default 2.
        n_landmark (int, optional): Number of landmarks to use in fast PHATE, default None.
        t (int or str, optional): Diffusion time scale parameter, 'auto' by default.
        knn (int, optional): Number of nearest neighbors for graph construction, default 5.
        gamma (float, optional): Controls the decay rate of kernel tails, default 1.
        **kwargs: Additional keyword arguments for PHATE.
        
    Attributes:
        params (dict): Dictionary of parameters for the PHATE model.
        model (phate.PHATE): The internal PHATE model instance once initialized and fitted.
        
    Methods:
        fit(X, y=None): Fit the PHATE model on the input data X.
        transform(X): Transform new data X using the trained PHATE model.
        fit_transform(X, y=None): Compute and return the PHATE embedding.
    """
    
    def __init__(self, n_components: int = 2, n_landmark: Optional[int] = None, 
                 t: Optional[int] = 'auto', knn: int = 5, 
                 gamma: float = 1.0, **kwargs):
        # Remove alpha parameter if it exists in kwargs to avoid conflicts
        if 'alpha' in kwargs:
            del kwargs['alpha']
            
        self.params = {
            'n_components': n_components,
            'n_landmark': n_landmark,
            't': t,
            'knn': knn,
            'gamma': gamma,
            **kwargs
        }
        self.model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PHATEReducer":
        """Fit PHATE on X."""
        self.model = phate.PHATE(**self.params)
        self.model.fit(X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the trained PHATE to X."""
        if self.model is None:
            raise RuntimeError("PHATEReducer: call fit before transform")
        return self.model.transform(X)
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute and return the PHATE embedding."""
        self.fit(X)
        return self.model.transform(X) 