import umap
from coco_pipe.dim_reduction.reducers.base import BaseReducer

class UMAPReducer(BaseReducer):
    """UMAP dimensionality reducer."""
    def __init__(self, n_components: int = 2, **kwargs):
        self.params = dict(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X, y=None):
        """Fit UMAP on X."""
        self.model = umap.UMAP(**self.params)
        self.model.fit(X)

    def transform(self, X):
        """Apply the trained UMAP to X (out-of-sample)."""
        if self.model is None:
            raise RuntimeError("UMAPReducer: call fit before transform")
        return self.model.transform(X)