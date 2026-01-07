# Available methods
METHODS = ["PCA", "TSNE", "UMAP", "PACMAP", "TRIMAP", "PHATE", 
"ISOMAP", "LLE", "MDS", "SPECTRAL", "DMD", "TRCA", "IVIS", "TOPO_AE"]

# Dynamically import your reducers here:
from .reducers.linear import PCAReducer
from .reducers.manifold import IsomapReducer, LLEReducer, MDSReducer, SpectralEmbeddingReducer
from .reducers.spatiotemporal import DMDReducer, TRCAReducer
from .reducers.neural import IVISReducer
from .reducers.topology import TopologicalAEReducer
from .reducers.neighbor import TSNEReducer, UMAPReducer, PacmapReducer, TrimapReducer, PHATEReducer

METHODS_DICT = {
    "PCA": PCAReducer,
    "ISOMAP": IsomapReducer,
    "LLE": LLEReducer,
    "MDS": MDSReducer,
    "SPECTRAL": SpectralEmbeddingReducer,
    "DMD": DMDReducer,
    "TRCA": TRCAReducer,
    "IVIS": IVISReducer,
    "TOPO_AE": TopologicalAEReducer,
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
    "PACMAP": PacmapReducer,
    "TRIMAP": TrimapReducer,
    "PHATE": PHATEReducer,
}