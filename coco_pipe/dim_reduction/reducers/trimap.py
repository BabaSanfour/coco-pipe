import trimap
import numpy as np
from .base import BaseReducer

class TrimapReducer(BaseReducer):
    """
    TriMap dimensionality reducer.

    This class applies TriMap (Large-scale Dimensionality Reduction Using Triplets) for dimensionality reduction.
    TriMap is designed to preserve both local and global structure of the data.
    
    Parameters:
        n_components (int): The target dimension for projection, default 2.
        n_inliers (int, optional): Number of nearest neighbors for forming nearest neighbor triplets.
            Default is 10.
        n_outliers (int, optional): Number of outliers for forming nearest neighbor triplets.
            Default is 5.
        n_random (int, optional): Number of random triplets per point. Default is 5.
        distance (str, optional): Distance measure ('euclidean', 'manhattan', 'angular', 'hamming').
            Default is 'euclidean'.
        weight_temp (float, optional): Temperature of the logarithm applied to weights.
            Larger values generate more compact embeddings. Default is 0.5.
        **kwargs: Additional keyword arguments for TriMap.

    Attributes:
        model (trimap.TRIMAP): The internal TriMap model instance once initialized.
        
    Methods:
        fit(X, y=None): Fit the TriMap model on the input data X.
        transform(X): Transform new data X using the trained TriMap model.
        fit_transform(X, y=None): Compute and return the TriMap embedding.
    """
    def __init__(self, n_components=2, n_inliers=10, n_outliers=5, n_random=5, 
                 distance='euclidean', weight_temp=0.5, **kwargs):
        self.n_components = n_components
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.distance = distance
        self.weight_temp = weight_temp
        self.kwargs = kwargs
        self.model = None
        self.embedding_ = None

    def fit(self, X, y=None):
        """Fit TriMap on X."""
        self.model = trimap.TRIMAP(
            n_dims=self.n_components,
            n_inliers=self.n_inliers,
            n_outliers=self.n_outliers,
            n_random=self.n_random,
            distance=self.distance,
            weight_temp=self.weight_temp,
            **self.kwargs
        )
        self.embedding_ = self.model.fit_transform(X)
        return self

    def transform(self, X):
        """Apply the trained TriMap to X."""
        if self.model is None:
            raise RuntimeError("TrimapReducer: call fit before transform")
        
        # TriMap doesn't support direct transform for new data
        # If X is the same data used for fitting, return the precomputed embedding
        if X.shape[0] == self.embedding_.shape[0]:
            return self.embedding_
        
        # For new data, we need to refit the model
        # This is a limitation of TriMap
        return self.model.fit_transform(X)

    def fit_transform(self, X, y=None):
        """Compute and return the TriMap embedding."""
        self.fit(X)
        return self.embedding_ 