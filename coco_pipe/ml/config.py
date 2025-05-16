# coco_pipe/ml/config.py

"""
Global default configuration for ML pipelines.
"""

DEFAULT_CV = {
    "strategy": "stratified",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "n_groups": 1,
}
