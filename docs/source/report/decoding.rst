.. _report-section-decoding:

==================================
Decoding Sections
==================================

:mod:`coco_pipe.report.decoding` provides nine section-adder functions
and a one-shot factory that build report sections from
:class:`~coco_pipe.decoding.result.ExperimentResult` objects (or their
serialized payloads). Every adder is also bound to :class:`Report` as
a fluent method.

---

1. Section Catalog
====================

Each adder reads from a specific :class:`ExperimentResult` accessor
and produces one section. Sections are silently skipped when the
underlying data is absent.

==========================================  =====================================================
Function / method                            Backing accessor
==========================================  =====================================================
:func:`~coco_pipe.report.decoding.add_decoding_overview`         ``result.meta``
:func:`~coco_pipe.report.decoding.add_decoding_summary`          ``result.get_detailed_scores()``
:func:`~coco_pipe.report.decoding.add_decoding_diagnostics`      ``result.get_detailed_scores()`` + ``result.get_splits()``
:func:`~coco_pipe.report.decoding.add_decoding_performance`      ``result.get_roc_curve()`` / ``get_pr_curve()`` / ``get_calibration_curve()``
:func:`~coco_pipe.report.decoding.add_decoding_temporal`         ``result.get_temporal_score_summary()`` / ``get_generalization_matrix()``
:func:`~coco_pipe.report.decoding.add_decoding_statistical_assessment`  ``result.get_statistical_assessment()``
:func:`~coco_pipe.report.decoding.add_decoding_neural_artifacts`  ``result.get_model_artifacts()``
:func:`~coco_pipe.report.decoding.add_decoding_features`         ``result.get_feature_importances()`` / ``get_feature_stability()``
:func:`~coco_pipe.report.decoding.add_decoding_topomaps`         ``result.get_feature_importances()`` + ``info`` / ``coords``
==========================================  =====================================================

Two private adders (``_add_configuration``, ``_add_provenance``,
``_add_confusion_probability``, ``_add_fit_diagnostics``,
``_add_caveats``, ``_add_export_inventory``) handle administrative
sections — they're invoked by
:func:`~coco_pipe.report.decoding.make_decoding_report` but aren't
exposed as part of the fluent API. See the source if you need to
replicate their behavior in custom workflows.

---

2. Quickstart
===============

.. code-block:: python

   from coco_pipe.report import Report

   report = (
       Report(title="Decoding")
       .add_decoding_overview(result)
       .add_decoding_summary(result)
       .add_decoding_performance(result)
       .add_decoding_temporal(result, metric="accuracy")
       .add_decoding_statistical_assessment(result)
       .add_decoding_features(result)
   )
   report.save("decoding.html")

Or all at once via the factory:

.. code-block:: python

   from coco_pipe.report import make_decoding_report

   report = make_decoding_report(result, output_path="decoding.html")

---

3. Each Adder in Detail
=========================

3.1 ``add_decoding_overview``
-------------------------------

One-row context table summarising the experiment: task, sample count,
feature count, observation level, inferential unit, schema version.
Sourced from ``result.meta``.

.. code-block:: python

   report.add_decoding_overview(result, name="Overview")

3.2 ``add_decoding_summary``
------------------------------

Per-model performance summary — mean ± std for every metric across
folds. Includes a :class:`MetricsTableElement` that highlights the
best model per metric.

.. code-block:: python

   report.add_decoding_summary(result, name="Model Performance")

3.3 ``add_decoding_diagnostics``
----------------------------------

Cross-validation diagnostics: per-fold scores, fold sizes, group
counts (when group-based CV), and any
fit/predict warnings (e.g., LinearAlgebra warnings).

.. code-block:: python

   report.add_decoding_diagnostics(result, model="SVM")   # filter to one model

3.4 ``add_decoding_performance``
----------------------------------

ROC, precision-recall, calibration curves — interactive Plotly
versions. Falls back gracefully if the result doesn't carry
probabilities.

.. code-block:: python

   report.add_decoding_performance(result, metric="roc_auc")

3.5 ``add_decoding_temporal``
-------------------------------

For sliding / generalizing decoders: time-resolved score curves and,
when present, the train-time × test-time generalization matrix
heatmap.

.. code-block:: python

   report.add_decoding_temporal(result, metric="accuracy")

3.6 ``add_decoding_statistical_assessment``
---------------------------------------------

The full assessment view: observed score, null distribution histogram,
p-value, max-stat-corrected p-values for temporal decoders.

.. code-block:: python

   report.add_decoding_statistical_assessment(result, metric="accuracy")

3.7 ``add_decoding_neural_artifacts``
---------------------------------------

Loss / metric training curves, learning-rate schedules, and any
artifacts stored by neural-network estimators
(:meth:`ExperimentResult.get_model_artifacts`).

.. code-block:: python

   report.add_decoding_neural_artifacts(result)

3.8 ``add_decoding_features``
-------------------------------

Top-N feature importances, stability across folds, feature scores
when a univariate selector was used. Accepts an optional
``feature_metadata`` DataFrame to annotate features with metadata
(e.g., channel, frequency, descriptor family).

.. code-block:: python

   report.add_decoding_features(result, feature_metadata=meta_df)

3.9 ``add_decoding_topomaps``
-------------------------------

Sensor-level topomaps of feature importance. Requires either an MNE
:class:`~mne.Info` or a coordinate table.

.. code-block:: python

   report.add_decoding_topomaps(
       result,
       feature_metadata=meta_df,
       info=raw.info,
   )

---

4. ``make_decoding_report``
=============================

The factory builds a full report from an :class:`ExperimentResult` in
one call, calling every applicable adder in a stable order.

.. code-block:: python

   from coco_pipe.report import make_decoding_report

   report = make_decoding_report(
       result,
       title="Decoding Report",
       feature_metadata=meta_df,        # optional, enables topomaps + annotation
       info=raw.info,                   # optional, enables topomaps
       sections="default",              # or a custom subset
       theme="paper",
       output_path="decoding.html",
   )

4.1 Sections argument
-----------------------

``sections="default"`` enables every supported section. To narrow:

.. code-block:: python

   make_decoding_report(
       result,
       sections=["overview", "performance", "temporal"],
   )

Valid section names (see :data:`~coco_pipe.report.decoding.DEFAULT_SECTIONS`):

.. code-block:: text

   overview, configuration, provenance, model_summary, cv_summary,
   performance, statistical, confusion_probability, temporal, features,
   fit_diagnostics, caveats, export_inventory, topomaps

Unknown names raise ``ValueError`` at the call site.

4.2 Error tolerance
---------------------

Each section is wrapped in a ``try / except``: if the underlying
accessor returns ``None`` or raises, the section is logged and
skipped, but the rest of the report still renders. This means you can
pass a partial result without preflighting.

---

5. Loading from a Saved Result
================================

:class:`ExperimentResult` supports JSON round-trips
(:meth:`~coco_pipe.decoding.result.ExperimentResult.save` /
:meth:`~coco_pipe.decoding.result.ExperimentResult.load`). Reports
built from a loaded result are byte-identical to ones built from the
in-memory version.

.. code-block:: python

   from coco_pipe.decoding.result import ExperimentResult
   from coco_pipe.report import make_decoding_report

   result = ExperimentResult.load("results/run01.json")
   make_decoding_report(result, output_path="reports/run01.html")

---

6. Compared to the API Factory
================================

:func:`coco_pipe.report.from_experiment_result` is a thin wrapper over
:func:`make_decoding_report` with a slightly different default surface
(see :ref:`report-api`). When in doubt, use
:func:`make_decoding_report` — it accepts every argument the factory
exposes.
