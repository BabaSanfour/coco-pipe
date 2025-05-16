# Available methods
METHODS = ["PCA", "TSNE", "UMAP", "PACMAP"]

# Dynamically import your reducers here:
from .reducers import PCAReducer, TSNEReducer, UMAPReducer, PacmapReducer

METHODS_DICT = {
    "PCA":  PCAReducer,
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
    "PACMAP": PacmapReducer,
}