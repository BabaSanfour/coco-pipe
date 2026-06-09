# API Reference

This page lists the stable public API entry points that should be used from
source code and examples. The modeling API is `coco_pipe.decoding`; older
modeling surfaces are not part of the supported public API.

## Decoding

Use `coco_pipe.decoding` for classification, regression, cross-validation,
feature selection, hyperparameter tuning, temporal decoding, and result
accessors.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.decoding.Experiment
   coco_pipe.decoding.ExperimentConfig
   coco_pipe.decoding.EstimatorSpec
   coco_pipe.decoding.EstimatorCapabilities
   coco_pipe.decoding.SelectorCapabilities
   coco_pipe.decoding.result.ExperimentResult
   coco_pipe.decoding.result.ExperimentResult.to_payload
   coco_pipe.decoding.result.ExperimentResult.save
   coco_pipe.decoding.result.ExperimentResult.load
   coco_pipe.decoding.get_estimator_cls
   coco_pipe.decoding.register_estimator
   coco_pipe.decoding.register_estimator_spec
   coco_pipe.decoding.get_estimator_spec
   coco_pipe.decoding.list_estimator_specs
   coco_pipe.decoding.get_capabilities
   coco_pipe.decoding.list_capabilities
   coco_pipe.decoding.run_statistical_assessment
   coco_pipe.decoding.binomial_accuracy_test
   coco_pipe.decoding.aggregate_predictions_for_inference
```

### Decoding Configs

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.decoding.configs.CVConfig
   coco_pipe.decoding.configs.FeatureSelectionConfig
   coco_pipe.decoding.configs.TuningConfig
   coco_pipe.decoding.configs.CalibrationConfig
   coco_pipe.decoding.configs.StatisticalAssessmentConfig
   coco_pipe.decoding.configs.ClassicalModelConfig
   coco_pipe.decoding.configs.LogisticRegressionConfig
   coco_pipe.decoding.configs.RandomForestClassifierConfig
   coco_pipe.decoding.configs.SVCConfig
   coco_pipe.decoding.configs.LinearSVCConfig
   coco_pipe.decoding.configs.RidgeConfig
   coco_pipe.decoding.configs.RandomForestRegressorConfig
   coco_pipe.decoding.configs.SVRConfig
   coco_pipe.decoding.configs.SlidingEstimatorConfig
   coco_pipe.decoding.configs.GeneralizingEstimatorConfig
```

### Decoding Splitters, Metrics, And Registry

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.decoding.registry.get_estimator_cls
   coco_pipe.decoding.registry.get_estimator_spec
   coco_pipe.decoding.registry.get_capabilities
   coco_pipe.decoding.registry.resolve_estimator_spec
```

## Dimensionality Reduction

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.DimReduction
   coco_pipe.dim_reduction.config.EvaluationConfig
   coco_pipe.dim_reduction.BaseReducer
   coco_pipe.dim_reduction.config.BaseReducerConfig
```

## IO

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.io.DataContainer
   coco_pipe.io.load_data
```

## Reports

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.Report
   coco_pipe.report.core.Report.add_decoding_diagnostics
   coco_pipe.report.core.Report.add_decoding_statistical_assessment
   coco_pipe.report.core.Report.add_decoding_temporal
   coco_pipe.report.config.ReportConfig
```

## Visualization

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.plot_temporal_score_curve
   coco_pipe.viz.plot_temporal_generalization_matrix
   coco_pipe.viz.plot_temporal_statistical_assessment
   coco_pipe.viz.plot_confusion_matrix
   coco_pipe.viz.plot_roc_curve
   coco_pipe.viz.plot_pr_curve
   coco_pipe.viz.plot_calibration_curve
   coco_pipe.viz.plot_fold_score_dispersion
```

## Full Module Index

The generated [AutoAPI module index](autoapi/index) is still available for
lower-level internals and module exploration, but the public modeling API
should be documented and used through `coco_pipe.decoding`.
