from sklearn.manifold import TSNE
from reducers.base import BaseReducer

class TSNEReducer(BaseReducer):
    """
    t-SNE dimensionality reducer.

    This class applies t-distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
    It encapsulates the TSNE model and provides a convenient interface to fit on data.
    Note: t-SNE does not support out-of-sample transformation; use fit_transform on your complete dataset.

    Parameters:
        n_components (int): The target number of dimensions for projection.
        **kwargs: Additional keyword arguments to configure the TSNE model. Common kwargs include:
            perplexity (float): The perplexity parameter related to the effective number of nearest neighbors.
            learning_rate (float): The learning rate for the t-SNE optimization.
            n_iter (int): The maximum number of iterations for optimization.

    Attributes:
        params (dict): Dictionary of parameters and keyword arguments initialized for the TSNE model.
        embedding_ (array-like): The resulting low-dimensional embedding after fitting the model.

    Methods:
        fit_transform(X, y=None): Compute and return the t-SNE embedding.
        save(filename: str): Save the trained UMAP model to a file.
        load(filename: str): Load a trained UMAP model from a file.

    """
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
