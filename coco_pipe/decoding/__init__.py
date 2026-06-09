"""
Decoding Module
===============

Core module for scientific decoding and machine learning experiments on
electrophysiological and behavioral data.
"""

from .configs import (
    CheckpointConfig,
    ClassicalModelConfig,
    DeviceConfig,
    ExperimentConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    LoRAConfig,
    NeuralFineTuneConfig,
    QuantizationConfig,
    StatisticalAssessmentConfig,
    TemporalDecoderConfig,
    TrainerConfig,
    TrainStageConfig,
)
from .experiment import Experiment
from .registry import (
    EstimatorCapabilities,
    get_capabilities,
    list_capabilities,
    register_estimator,
    register_estimator_spec,
)
from .result import ExperimentResult
from .stats import (
    aggregate_predictions_for_inference,
    binomial_accuracy_test,
    run_statistical_assessment,
)

__all__ = [
    # Configs
    "ExperimentConfig",
    "ClassicalModelConfig",
    "FoundationEmbeddingModelConfig",
    "FrozenBackboneDecoderConfig",
    "NeuralFineTuneConfig",
    "TemporalDecoderConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "DeviceConfig",
    "CheckpointConfig",
    "TrainerConfig",
    "TrainStageConfig",
    "StatisticalAssessmentConfig",
    # Execution
    "Experiment",
    "ExperimentResult",
    # Model Discovery & Metadata
    "register_estimator",
    "register_estimator_spec",
    "get_capabilities",
    "list_capabilities",
    "EstimatorCapabilities",
    # Stats Utilities
    "run_statistical_assessment",
    "binomial_accuracy_test",
    "aggregate_predictions_for_inference",
]
