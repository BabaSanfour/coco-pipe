from sklearn.decomposition import PCA
from .base import BaseReducer

class PCAReducer(BaseReducer):
    """
    PCA dimensionality reducer.

    This class applies Principal Component Analysis (PCA) for dimensionality reduction.
    It encapsulates the PCA model from scikit-learn and provides a convenient interface to
    fit on training data and transform new data.

    Parameters:
        n_components (int): The target number of principal components for projection.
        **kwargs: Additional keyword arguments to configure the PCA model. These may include
            options for whitening or other PCA-specific parameters as defined in scikit-learn.

    Attributes:
        model (sklearn.decomposition.PCA): The internal PCA model instance once initialized and fitted.

    Methods:
        fit(X, y=None): Fit the PCA model on the input data X.
        transform(X): Transform new data X using the trained PCA model.
        fit_transform(X): Compute and return the UMAP embedding.
        save(filename: str): Save the trained UMAP model to a file.
        load(filename: str): Load a trained UMAP model from a file.
    """
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
