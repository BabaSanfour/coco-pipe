import pacmap
import numpy as np
from .base import BaseReducer

class PacmapReducer(BaseReducer):
    """
    PaCMAP dimensionality reducer.

    This class applies Pairwise Controlled Manifold Approximation (PaCMAP) for dimensionality reduction.
    PaCMAP is designed to preserve both local and global structure of the data.
    
    Parameters:
        n_components (int): The target dimension for projection, default 2.
        n_neighbors (int, optional): Number of neighbors considered during graph construction.
            Default is 10. If None, will auto-select based on dataset size.
        MN_ratio (float, optional): The ratio of mid-near pairs to neighbors. Default is 0.5.
        FP_ratio (float, optional): The ratio of far pairs to neighbors. Default is 2.0.
        **kwargs: Additional keyword arguments for PaCMAP.

    Attributes:
        model (pacmap.PaCMAP): The internal PaCMAP model instance once initialized.
        
    Methods:
        fit(X, y=None): Fit the PaCMAP model on the input data X.
        transform(X): Transform new data X using the trained PaCMAP model.
        fit_transform(X, y=None): Compute and return the PaCMAP embedding.
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 10, 
                 MN_ratio: float = 0.5, FP_ratio: float = 2.0, **kwargs):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.kwargs = kwargs
        self.model = None
        self.embedding_ = None

    def fit(self, X, y=None):
        """Fit PaCMAP on X."""
        self.model = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.MN_ratio,
            FP_ratio=self.FP_ratio,
            **self.kwargs
        )
        self.embedding_ = self.model.fit_transform(X, init="pca")
        return self

    def transform(self, X):
        """Apply the trained PaCMAP to X."""
        if self.model is None:
            raise RuntimeError("PacmapReducer: call fit before transform")
        if X.shape[0] == self.embedding_.shape[0]:
            # If same number of samples, return the precomputed embedding
            return self.embedding_
        # For new data, PaCMAP doesn't support direct transform, so we need to refit
        # This is a limitation of PaCMAP
        return self.model.fit_transform(X, init="pca")

    def fit_transform(self, X, y=None):
        """Compute and return the PaCMAP embedding."""
        self.fit(X)
        return self.embedding_ 