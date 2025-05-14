# Available methods
METHODS = ["PCA", "TSNE", "UMAP"]

# Dynamically import your reducers here:
from .reducers import PCAReducer, TSNEReducer, UMAPReducer

METHODS_DICT = {
    "PCA":  PCAReducer,
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
}