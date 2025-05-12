from sklearn.decomposition import PCA
from coco_pipe.dim_reduction.reducers.base import BaseReducer

class PCAReducer(BaseReducer):
    """PCA-based dimensionality reducer."""
    def __init__(self, n_components: int = 2, **kwargs):
        self.n_components = n_components
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y=None):
        """Fit PCA on X."""
        self.model = PCA(n_components=self.n_components, **self.kwargs)
        self.model.fit(X)

    def transform(self, X):
        """Apply the trained PCA to X."""
        if self.model is None:
            raise RuntimeError("PCAReducer: call fit before transform")
        return self.model.transform(X)
