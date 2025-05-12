from sklearn.manifold import TSNE
from coco_pipe.dim_reduction.reducers.base import BaseReducer

class TSNEReducer(BaseReducer):
    """t-SNE dimensionality reducer. Note: no transform for new data."""
    def __init__(self, n_components: int = 2, **kwargs):
        self.params = dict(n_components=n_components, **kwargs)
        self.embedding_ = None

    def fit(self, X, y=None):
        """Compute t-SNE embedding for X."""
        self.embedding_ = TSNE(**self.params).fit_transform(X)

    def transform(self, X):
        """t-SNE does not support out-of-sample transformation by default."""
        raise NotImplementedError(
            "TSNEReducer cannot transform new data. Use fit_transform on your dataset."
        )

    def fit_transform(self, X, y=None):  # override base
        """Compute and return the t-SNE embedding."""
        self.fit(X, y=y)
        return self.embedding_
