# API Reference

The public API of **coco-pipe**. Most workflows use a small set of high-level
entry points listed under **Selected APIs** below; the **Full Module Index**
documents every public and internal symbol, generated directly from the source.

For conceptual background and worked examples, see the {doc}`user_guide`.

---

## Selected APIs

The most commonly used classes and functions, grouped by module.

### Decoding

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.decoding.Experiment
   coco_pipe.decoding.ExperimentConfig
   coco_pipe.decoding.ExperimentResult
   coco_pipe.decoding.get_capabilities
   coco_pipe.decoding.list_capabilities
   coco_pipe.decoding.register_estimator
   coco_pipe.decoding.run_statistical_assessment
```

### Dimensionality Reduction

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.dim_reduction.DimReduction
   coco_pipe.dim_reduction.BaseReducer
   coco_pipe.dim_reduction.METHODS
   coco_pipe.dim_reduction.evaluation.MethodSelector
   coco_pipe.dim_reduction.interpret_features
```

### Reports

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.report.Report
   coco_pipe.report.Section
   coco_pipe.report.from_experiment_result
   coco_pipe.report.from_reductions
   coco_pipe.report.from_container
   coco_pipe.report.merge_reports
```

### Visualization

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.viz.plot_embedding
   coco_pipe.viz.plot_decoding_scores
   coco_pipe.viz.plot_confusion_matrix
   coco_pipe.viz.plot_roc_curve
   coco_pipe.viz.plot_topomap
   coco_pipe.viz.set_coco_theme
   coco_pipe.viz.save_figure
```

### IO and Features

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.io.DataContainer
   coco_pipe.io.load_data
```

### Transforms and Diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   coco_pipe.transforms.LeaceEraser
   coco_pipe.transforms.EuclideanAlign
   coco_pipe.transforms.RiemannAlign
   coco_pipe.transforms.TemporalProcrustesAlignment
   coco_pipe.transforms.make_subject_transform
   coco_pipe.diagnostics.variance_decomposition_report
```

---

## Full Module Index

The complete, auto-generated reference for every public and internal symbol,
organized by module. Jump straight to a module:

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} io
:link: api/coco_pipe/io/index
:link-type: doc

Loading, validation, and the ``DataContainer``.
:::

:::{grid-item-card} descriptors
:link: api/coco_pipe/descriptors/index
:link-type: doc

Signal feature extraction.
:::

:::{grid-item-card} dim_reduction
:link: api/coco_pipe/dim_reduction/index
:link-type: doc

Reducers, evaluation, and trajectories.
:::

:::{grid-item-card} decoding
:link: api/coco_pipe/decoding/index
:link-type: doc

Classical ML and foundation-model decoding.
:::

:::{grid-item-card} viz
:link: api/coco_pipe/viz/index
:link-type: doc

Static and interactive plotting.
:::

:::{grid-item-card} report
:link: api/coco_pipe/report/index
:link-type: doc

Automated HTML reports.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

Full Module Index <api/coco_pipe/index>
```
