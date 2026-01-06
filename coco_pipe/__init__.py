"""
Package initializer for the coco_pipe package.
"""

from .ml import (
    MLPipeline,
    pipeline_baseline,
    pipeline_feature_selection,
    pipeline_HP_search,
    pipeline_feature_selection_HP_search,
    pipeline_unsupervised
)

from .dim_reduction import (
    DimReducer,
    METHODS,
    METHODS_DICT,
    PCAReducer,
    TSNEReducer,
    UMAPReducer,
    BaseReducer,
    DimReductionPipeline
)
