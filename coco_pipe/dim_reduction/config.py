# Available methods
METHODS = ["PCA", "TSNE", "UMAP"]

# Dynamically import your reducers here:
from coco_pipe.dim_reduction.reducers.pca  import PCAReducer
from coco_pipe.dim_reduction.reducers.tsne import TSNEReducer
from coco_pipe.dim_reduction.reducers.umap import UMAPReducer

METHODS_DICT = {
    "PCA":  PCAReducer,
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
}