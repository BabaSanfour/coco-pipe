# coco_pipe/ml/__init__.py

from .config import DEFAULT_CV
from .base import CrossValidationStrategy, BasePipeline

__all__ = [
    "DEFAULT_CV",
    "CrossValidationStrategy",
    "BasePipeline",
]
