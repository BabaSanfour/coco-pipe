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
   coco_pipe.decoding.EstimatorCapabilities
   coco_pipe.decoding.result.ExperimentResult
   coco_pipe.decoding.result.ExperimentResult.to_payload
   coco_pipe.decoding.result.ExperimentResult.save
   coco_pipe.decoding.result.ExperimentResult.load
   coco_pipe.decoding.register_estimator
   coco_pipe.decoding.register_estimator_spec
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

The full user guide lives at {ref}`dim-reduction`. The summaries below list
every public entry point.

### Manager and Registry

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.DimReduction
   coco_pipe.dim_reduction.METHODS
   coco_pipe.dim_reduction.BaseReducer
```

### Reducer Configs

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.config.BaseReducerConfig
   coco_pipe.dim_reduction.config.StochasticReducerConfig
   coco_pipe.dim_reduction.config.PCAConfig
   coco_pipe.dim_reduction.config.IncrementalPCAConfig
   coco_pipe.dim_reduction.config.DaskPCAConfig
   coco_pipe.dim_reduction.config.DaskTruncatedSVDConfig
   coco_pipe.dim_reduction.config.UMAPConfig
   coco_pipe.dim_reduction.config.ParametricUMAPConfig
   coco_pipe.dim_reduction.config.TSNEConfig
   coco_pipe.dim_reduction.config.PacmapConfig
   coco_pipe.dim_reduction.config.TrimapConfig
   coco_pipe.dim_reduction.config.PHATEConfig
   coco_pipe.dim_reduction.config.IsomapConfig
   coco_pipe.dim_reduction.config.LLEConfig
   coco_pipe.dim_reduction.config.MDSConfig
   coco_pipe.dim_reduction.config.SpectralEmbeddingConfig
   coco_pipe.dim_reduction.config.DMDConfig
   coco_pipe.dim_reduction.config.TRCAConfig
   coco_pipe.dim_reduction.config.TopologicalAEConfig
   coco_pipe.dim_reduction.config.IVISConfig
   coco_pipe.dim_reduction.config.EvaluationConfig
   coco_pipe.dim_reduction.config.get_reducer_class
```

### Evaluation and Comparison

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.evaluation.core.evaluate_embedding
   coco_pipe.dim_reduction.evaluation.MethodSelector
   coco_pipe.dim_reduction.trustworthiness
   coco_pipe.dim_reduction.continuity
   coco_pipe.dim_reduction.lcmc
   coco_pipe.dim_reduction.shepard_diagram_data
   coco_pipe.dim_reduction.evaluation.metrics.compute_coranking_matrix
   coco_pipe.dim_reduction.evaluation.metrics.compute_mrre
   coco_pipe.dim_reduction.paired_condition_stats
   coco_pipe.dim_reduction.grouped_condition_stats
```

### Trajectory Geometry

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.trajectory_speed
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_acceleration
   coco_pipe.dim_reduction.trajectory_curvature
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_turning_angle
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_path_length
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_displacement
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_tortuosity
   coco_pipe.dim_reduction.evaluation.geometry.trajectory_dispersion
   coco_pipe.dim_reduction.trajectory_distance_from_center
   coco_pipe.dim_reduction.trajectory_cohesion
   coco_pipe.dim_reduction.trajectory_intra_spread
   coco_pipe.dim_reduction.trajectory_auc_speed
   coco_pipe.dim_reduction.trajectory_separation
```

### Interpretation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.interpret_features
   coco_pipe.dim_reduction.analysis.correlate_features
   coco_pipe.dim_reduction.analysis.perturbation_importance
   coco_pipe.dim_reduction.analysis.gradient_importance
```

### Reducer Catalog

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.PCAReducer
   coco_pipe.dim_reduction.IncrementalPCAReducer
   coco_pipe.dim_reduction.IsomapReducer
   coco_pipe.dim_reduction.LLEReducer
   coco_pipe.dim_reduction.MDSReducer
   coco_pipe.dim_reduction.SpectralEmbeddingReducer
   coco_pipe.dim_reduction.TSNEReducer
```

### Preprocessing Helpers

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.dim_reduction.apply_pca_score_baseline
   coco_pipe.dim_reduction.flip_pc_scores_for_consistency
```

## IO

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.io.DataContainer
   coco_pipe.io.load_data
```

## Reports

The full user guide lives at {ref}`report`. The summaries below list
every public entry point.

### Core Containers

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.Report
   coco_pipe.report.Section
```

### Factories and Assembly

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.from_container
   coco_pipe.report.from_bids
   coco_pipe.report.from_tabular
   coco_pipe.report.from_embeddings
   coco_pipe.report.from_reductions
   coco_pipe.report.from_experiment_result
   coco_pipe.report.merge_reports
   coco_pipe.report.make_decoding_report
   coco_pipe.report.make_reduction_report
```

### Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.config.ReportConfig
   coco_pipe.report.config.ProvenanceConfig
```

### Decoding Section Adders (`coco_pipe.report.decoding`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.decoding.add_decoding_overview
   coco_pipe.report.decoding.add_decoding_summary
   coco_pipe.report.decoding.add_decoding_diagnostics
   coco_pipe.report.decoding.add_decoding_performance
   coco_pipe.report.decoding.add_decoding_temporal
   coco_pipe.report.decoding.add_decoding_statistical_assessment
   coco_pipe.report.decoding.add_decoding_neural_artifacts
   coco_pipe.report.decoding.add_decoding_features
   coco_pipe.report.decoding.add_decoding_topomaps
```

### Dim-Reduction Section Adders (`coco_pipe.report.dim_reduction`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.dim_reduction.add_reduction
   coco_pipe.report.dim_reduction.add_comparison
   coco_pipe.report.dim_reduction.add_reduction_overview
   coco_pipe.report.dim_reduction.add_reduction_embedding
   coco_pipe.report.dim_reduction.add_reduction_metrics
   coco_pipe.report.dim_reduction.add_reduction_diagnostics
   coco_pipe.report.dim_reduction.add_reduction_interpretation
   coco_pipe.report.dim_reduction.add_reduction_coranking
   coco_pipe.report.dim_reduction.add_reduction_components
   coco_pipe.report.dim_reduction.add_reduction_trajectory
   coco_pipe.report.dim_reduction.add_reduction_trajectory_separation
```

### Element Primitives (`coco_pipe.report.elements`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report.elements.Element
   coco_pipe.report.elements.HtmlElement
   coco_pipe.report.elements.ImageElement
   coco_pipe.report.elements.PlotlyElement
   coco_pipe.report.elements.TableElement
   coco_pipe.report.elements.InteractiveTableElement
   coco_pipe.report.elements.MetricsTableElement
   coco_pipe.report.elements.StatCardElement
   coco_pipe.report.elements.CalloutElement
   coco_pipe.report.elements.CodeBlockElement
   coco_pipe.report.elements.MarkdownElement
   coco_pipe.report.elements.BadgeElement
   coco_pipe.report.elements.ProgressBarElement
   coco_pipe.report.elements.TimelineElement
   coco_pipe.report.elements.TabsElement
   coco_pipe.report.elements.AccordionElement
   coco_pipe.report.elements.ColumnsElement
   coco_pipe.report.elements.ContainerElement
   coco_pipe.report.elements.DownloadAssetElement
```

### Data Quality

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.io.quality.CheckResult
   coco_pipe.io.quality.check_missingness
   coco_pipe.io.quality.check_constant_columns
   coco_pipe.io.quality.check_outliers_zscore
   coco_pipe.io.quality.check_flatline
```

### Asset Vendoring (Offline Mode)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.report._assets.get_vendored_contents
   coco_pipe.report._assets.vendor_assets
```

## Visualization

The full user guide lives at {ref}`viz`. The summaries below list every public
plotting entry point.

### Theme & Helpers

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.theme.coco_theme
   coco_pipe.viz.theme.set_coco_theme
   coco_pipe.viz.theme.figure_size
   coco_pipe.viz.theme.save_figure
```

### Plotting Primitives (`coco_pipe.viz.base`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.plot_bar
   coco_pipe.viz.plot_line
   coco_pipe.viz.plot_error_points
   coco_pipe.viz.plot_distribution_groups
   coco_pipe.viz.plot_scatter2d
   coco_pipe.viz.plot_scatter3d
   coco_pipe.viz.plot_heatmap
   coco_pipe.viz.plot_hexbin
   coco_pipe.viz.plot_streamfield
   coco_pipe.viz.plot_topomap
```

### Static Decoding Plots (`coco_pipe.viz.decoding`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.plot_decoding_scores
   coco_pipe.viz.plot_fold_score_dispersion
   coco_pipe.viz.plot_confusion_matrix
   coco_pipe.viz.plot_roc_curve
   coco_pipe.viz.plot_pr_curve
   coco_pipe.viz.plot_calibration_curve
   coco_pipe.viz.plot_probability_diagnostics
   coco_pipe.viz.plot_temporal_score_curve
   coco_pipe.viz.plot_temporal_generalization_matrix
   coco_pipe.viz.plot_temporal_statistical_assessment
   coco_pipe.viz.plot_null_interval_summary
   coco_pipe.viz.plot_model_comparison
   coco_pipe.viz.plot_fit_diagnostics
   coco_pipe.viz.plot_training_history
   coco_pipe.viz.plot_search_results
   coco_pipe.viz.plot_subject_diagnostics
   coco_pipe.viz.plot_group_summary
   coco_pipe.viz.plot_regression_diagnostics
   coco_pipe.viz.plot_feature_importance
   coco_pipe.viz.plot_feature_stability
   coco_pipe.viz.plot_feature_scores
   coco_pipe.viz.plot_decoding_topomap
   coco_pipe.viz.plot_sensor_feature_heatmap
   coco_pipe.viz.plot_sensor_feature_profile
   coco_pipe.viz.plot_feature_sensor_profile
```

### Static Dimensionality-Reduction Plots (`coco_pipe.viz.dim_reduction`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.plot_embedding
   coco_pipe.viz.plot_metrics
   coco_pipe.viz.plot_eigenvalues
   coco_pipe.viz.plot_loss_history
   coco_pipe.viz.plot_shepard_diagram
   coco_pipe.viz.plot_coranking_matrix
   coco_pipe.viz.plot_component_loadings
   coco_pipe.viz.plot_reduction_feature_importance
   coco_pipe.viz.plot_feature_correlation_heatmap
   coco_pipe.viz.plot_trajectory
   coco_pipe.viz.plot_trajectory_separation
   coco_pipe.viz.plot_trajectory_metric_series
   coco_pipe.viz.plot_streamlines
```

### Interactive Plotly Plots (`coco_pipe.viz.interactive`)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   coco_pipe.viz.interactive.decoding.plot_decoding_scores
   coco_pipe.viz.interactive.decoding.plot_fold_score_dispersion
   coco_pipe.viz.interactive.decoding.plot_confusion_matrix
   coco_pipe.viz.interactive.decoding.plot_roc_curve
   coco_pipe.viz.interactive.decoding.plot_pr_curve
   coco_pipe.viz.interactive.decoding.plot_calibration_curve
   coco_pipe.viz.interactive.decoding.plot_probability_diagnostics
   coco_pipe.viz.interactive.decoding.plot_temporal_score_curve
   coco_pipe.viz.interactive.decoding.plot_temporal_generalization_matrix
   coco_pipe.viz.interactive.decoding.plot_temporal_statistical_assessment
   coco_pipe.viz.interactive.decoding.plot_null_interval_summary
   coco_pipe.viz.interactive.decoding.plot_model_comparison
   coco_pipe.viz.interactive.decoding.plot_fit_diagnostics
   coco_pipe.viz.interactive.decoding.plot_training_history
   coco_pipe.viz.interactive.decoding.plot_search_results
   coco_pipe.viz.interactive.decoding.plot_subject_diagnostics
   coco_pipe.viz.interactive.decoding.plot_group_summary
   coco_pipe.viz.interactive.decoding.plot_regression_diagnostics
   coco_pipe.viz.interactive.decoding.plot_feature_stability
   coco_pipe.viz.interactive.decoding.plot_feature_scores
   coco_pipe.viz.interactive.dim_reduction.plot_embedding
   coco_pipe.viz.interactive.dim_reduction.plot_metrics
   coco_pipe.viz.interactive.dim_reduction.plot_eigenvalues
   coco_pipe.viz.interactive.dim_reduction.plot_loss_history
   coco_pipe.viz.interactive.dim_reduction.plot_shepard_diagram
   coco_pipe.viz.interactive.dim_reduction.plot_coranking_matrix
   coco_pipe.viz.interactive.dim_reduction.plot_component_loadings
   coco_pipe.viz.interactive.dim_reduction.plot_feature_importance
   coco_pipe.viz.interactive.dim_reduction.plot_feature_correlation_heatmap
   coco_pipe.viz.interactive.dim_reduction.plot_streamlines
   coco_pipe.viz.interactive.dim_reduction.plot_trajectory
   coco_pipe.viz.interactive.dim_reduction.plot_trajectory_metric_series
   coco_pipe.viz.interactive.dim_reduction.plot_trajectory_separation
   coco_pipe.viz.interactive.dim_reduction.plot_channel_traces
   coco_pipe.viz.interactive.dim_reduction.plot_raw_preview
   coco_pipe.viz.interactive.dim_reduction.plot_radar_comparison
```

## Full Module Index

The generated [AutoAPI module index](autoapi/index) is still available for
lower-level internals and module exploration, but the public modeling API
should be documented and used through `coco_pipe.decoding`.
