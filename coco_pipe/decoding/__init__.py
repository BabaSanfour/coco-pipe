"""
Decoding Module
===============

Core module for scientific decoding and machine learning experiments on
electrophysiological and behavioral data.
"""

from ._specs import SignalMetadata
from .configs import (
    ChanceAssessmentConfig,
    CheckpointConfig,
    ClassicalModelConfig,
    CVConfig,
    DeviceConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    LoRAConfig,
    NeuralFineTuneConfig,
    QuantizationConfig,
    ReducerConfig,
    StatisticalAssessmentConfig,
    TemporalDecoderConfig,
    TrainerConfig,
    TrainStageConfig,
)
from .experiment import Experiment
from .persistence import (
    completed_for_config,
    config_hash,
    load_completed_result_records,
    redact_sensitive,
    write_run_status,
)
from .registry import (
    EstimatorCapabilities,
    get_capabilities,
    get_foundation_model_spec,
    list_capabilities,
    list_foundation_models,
    register_estimator,
    register_estimator_spec,
)
from .result import ExperimentResult
from .stats import (
    aggregate_predictions_for_inference,
    benjamini_hochberg,
    binomial_accuracy_test,
    correct_sweep_pvalues,
    run_statistical_assessment,
)
from .targets import prepare_target, safe_group_n_splits

__all__ = [
    # Configs
    "ExperimentConfig",
    "FeatureSelectionConfig",
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
    "ReducerConfig",
    "StatisticalAssessmentConfig",
    "CVConfig",
    "ChanceAssessmentConfig",
    # Execution
    "Experiment",
    "ExperimentResult",
    "SignalMetadata",
    "prepare_target",
    "safe_group_n_splits",
    "config_hash",
    "completed_for_config",
    "load_completed_result_records",
    "redact_sensitive",
    "write_run_status",
    # Model Discovery & Metadata
    "register_estimator",
    "register_estimator_spec",
    "get_capabilities",
    "get_foundation_model_spec",
    "list_capabilities",
    "list_foundation_models",
    "EstimatorCapabilities",
    # Stats Utilities
    "run_statistical_assessment",
    "binomial_accuracy_test",
    "aggregate_predictions_for_inference",
    "benjamini_hochberg",
    "correct_sweep_pvalues",
]
