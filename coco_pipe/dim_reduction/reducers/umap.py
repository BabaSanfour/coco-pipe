import umap
from typing import Optional
import numpy as np
from coco_pipe.dim_reduction.reducers.base import BaseReducer

class UMAPReducer(BaseReducer):
    """
    UMAP dimensionality reducer.

    This class applies Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction.
    It encapsulates the UMAP model and provides a convenient interface to fit on training data and transform new data.

    Parameters:
        n_components (int): The target number of dimensions for projection.
        **kwargs: Additional keyword arguments to configure the UMAP model. Common kwargs include:
            n_neighbors (int): The number of neighboring points used in local approximations.
            min_dist (float): The effective minimum distance between embedded points.
            metric (str): The distance metric to measure similarity between points.
            random_state (int): Seed for the random number generator to ensure reproducibility.

    Attributes:
        params (dict): Dictionary of parameters and keyword arguments initialized for the UMAP model.
        model (umap.UMAP): The internal UMAP model instance once initialized and fitted.

    Methods:
        fit(X, y=None): Fit the UMAP model on the input data X.
        transform(X): Transform new data X using the trained UMAP model.
    """
    def __init__(self, n_components: int = 2, **kwargs):
        self.params = dict(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "UMAPReducer":
        """Fit UMAP on X."""
        self.model = umap.UMAP(**self.params)
        self.model.fit(X)
        return self

    def transform(self, X):
        """Apply the trained UMAP to X (out-of-sample)."""
        if self.model is None:
            raise RuntimeError("UMAPReducer: call fit before transform")
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(f"Expected {self.model.n_features_in_} features, got {X.shape[1]} features")
        return self.model.transform(X)