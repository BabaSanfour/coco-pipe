# Available methods
METHODS = ["PCA", "TSNE", "UMAP", "PACMAP", "TRIMAP", "PHATE"]

# Dynamically import your reducers here:
from .reducers import PCAReducer, TSNEReducer, UMAPReducer, PacmapReducer, TrimapReducer, PHATEReducer

METHODS_DICT = {
    "PCA":  PCAReducer,
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
    "PACMAP": PacmapReducer,
    "TRIMAP": TrimapReducer,
    "PHATE": PHATEReducer,
}