.. _report-sections:

================
Section Builders
================

.. _report-section-decoding:

Decoding Sections
=================

:mod:`coco_pipe.report.decoding` provides nine section-adder functions
and a one-shot factory that build report sections from
:class:`~coco_pipe.decoding.result.ExperimentResult` objects (or their
serialized payloads). Every adder is also bound to :class:`Report` as
a fluent method.

---

1. Section Catalog
------------------

Each adder reads from a specific :class:`ExperimentResult` accessor
and produces one section. Sections are silently skipped when the
underlying data is absent.

.. list-table::
   :header-rows: 1

   * - Function / method
     - Backing accessor
   * - :func:`~coco_pipe.report.decoding.add_decoding_overview`
     - ``result.meta``
   * - :func:`~coco_pipe.report.decoding.add_decoding_summary`
     - ``result.get_detailed_scores()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_diagnostics`
     - ``result.get_detailed_scores()`` + ``result.get_splits()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_performance`
     - ``result.get_roc_curve()`` / ``get_pr_curve()`` / ``get_calibration_curve()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_temporal`
     - ``result.get_temporal_score_summary()`` / ``get_generalization_matrix()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_statistical_assessment`
     - ``result.get_statistical_assessment()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_neural_artifacts`
     - ``result.get_model_artifacts()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_features`
     - ``result.get_feature_importances()`` / ``get_feature_stability()``
   * - :func:`~coco_pipe.report.decoding.add_decoding_topomaps`
     - ``result.get_feature_importances()`` + ``info`` / ``coords``

Two private adders (``_add_configuration``, ``_add_provenance``,
``_add_confusion_probability``, ``_add_fit_diagnostics``,
``_add_caveats``, ``_add_export_inventory``) handle administrative
sections — they're invoked by
:func:`~coco_pipe.report.decoding.make_decoding_report` but aren't
exposed as part of the fluent API. See the source if you need to
replicate their behavior in custom workflows.

---

2. Quickstart
-------------

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
-----------------------

3.1 ``add_decoding_overview``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One-row context table summarising the experiment: task, sample count,
feature count, observation level, inferential unit, schema version.
Sourced from ``result.meta``.

.. code-block:: python

   report.add_decoding_overview(result, name="Overview")

3.2 ``add_decoding_summary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-model performance summary — mean ± std for every metric across
folds. Includes a :class:`MetricsTableElement` that highlights the
best model per metric.

.. code-block:: python

   report.add_decoding_summary(result, name="Model Performance")

3.3 ``add_decoding_diagnostics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-validation diagnostics: per-fold scores, fold sizes, group
counts (when group-based CV), and any
fit/predict warnings (e.g., LinearAlgebra warnings).

.. code-block:: python

   report.add_decoding_diagnostics(result, model="SVM")   # filter to one model

3.4 ``add_decoding_performance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROC, precision-recall, calibration curves — interactive Plotly
versions. Falls back gracefully if the result doesn't carry
probabilities.

.. code-block:: python

   report.add_decoding_performance(result, metric="roc_auc")

3.5 ``add_decoding_temporal``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For sliding / generalizing decoders: time-resolved score curves and,
when present, the train-time × test-time generalization matrix
heatmap.

.. code-block:: python

   report.add_decoding_temporal(result, metric="accuracy")

3.6 ``add_decoding_statistical_assessment``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full assessment view: observed score, null distribution histogram,
p-value, max-stat-corrected p-values for temporal decoders.

.. code-block:: python

   report.add_decoding_statistical_assessment(result, metric="accuracy")

3.7 ``add_decoding_neural_artifacts``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loss / metric training curves, learning-rate schedules, and any
artifacts stored by neural-network estimators
(:meth:`ExperimentResult.get_model_artifacts`).

.. code-block:: python

   report.add_decoding_neural_artifacts(result)

3.8 ``add_decoding_features``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Top-N feature importances, stability across folds, feature scores
when a univariate selector was used. Accepts an optional
``feature_metadata`` DataFrame to annotate features with metadata
(e.g., channel, frequency, descriptor family).

.. code-block:: python

   report.add_decoding_features(result, feature_metadata=meta_df)

3.9 ``add_decoding_topomaps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
---------------------------

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
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

Each section is wrapped in a ``try / except``: if the underlying
accessor returns ``None`` or raises, the section is logged and
skipped, but the rest of the report still renders. This means you can
pass a partial result without preflighting.

---

5. Loading from a Saved Result
------------------------------

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
------------------------------

:func:`coco_pipe.report.from_experiment_result` is a thin wrapper over
:func:`make_decoding_report` with a slightly different default surface
(see :ref:`report-api`). When in doubt, use
:func:`make_decoding_report` — it accepts every argument the factory
exposes.

.. _report-section-dim-reduction:

Dimensionality-Reduction Sections
=================================

:mod:`coco_pipe.report.dim_reduction` provides eleven section-adder
functions and a one-shot factory for building report sections from
:class:`~coco_pipe.dim_reduction.DimReduction` objects, embeddings,
and ``MethodSelector`` comparison frames.

---

1. Section Catalog
------------------

.. list-table::
   :header-rows: 1

   * - Function / method
     - Backing source
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_overview`
     - ``reduction.get_summary()``
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_embedding`
     - Explicit ``X_emb`` (2-D or 3-D)
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_metrics`
     - ``reduction.get_metrics()`` / ``get_summary()["metric_records"]``
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_diagnostics`
     - Shepard diagram (X_orig, X_emb)
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_interpretation`
     - ``get_summary()["interpretation"]``
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_coranking`
     - ``diagnostics["coranking_matrix_"]``
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_components`
     - ``reduction.get_components()``
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_trajectory`
     - 3-D trajectory tensor
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction_trajectory_separation`
     - ``trajectory_separation`` output
   * - :func:`~coco_pipe.report.dim_reduction.add_reduction`
     - Single fully-assembled section
   * - :func:`~coco_pipe.report.dim_reduction.add_comparison`
     - Wide/tidy metric frame or ``MethodSelector``

---

2. Two Levels of Granularity
----------------------------

Two levels of granularity are supported depending on how much
control the caller needs:

- **Coarse** — :meth:`Report.add_reduction` packages an overview,
  embedding/trajectory plot, scalar metrics, metric records,
  diagnostics (loss history, scree, co-ranking matrix, trajectory
  kinematics, separation timecourses), and interpretation into one
  section. One call per reduction.
- **Fine** — call individual ``add_reduction_*`` adders for full
  layout control. Use this when assembling a custom report.

.. code-block:: python

   # Coarse
   report.add_reduction(pca, X_emb=embedding, labels=y)

   # Fine
   (
       report
       .add_reduction_overview(pca, name="PCA Overview")
       .add_reduction_embedding(embedding, labels=y)
       .add_reduction_metrics(pca)
       .add_reduction_coranking(coranking_matrix)
       .add_reduction_components(pca.get_components())
   )

---

3. Each Adder in Detail
-----------------------

3.1 ``add_reduction_overview``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key/value summary from
:meth:`DimReduction.get_summary <coco_pipe.dim_reduction.core.DimReduction.get_summary>`
— method name, n_components, random_state, capability flags, scalar
quality metadata. Skips ``metrics`` / ``metric_records`` /
``diagnostics`` / ``interpretation`` blocks (they have dedicated
adders).

.. code-block:: python

   report.add_reduction_overview(pca, name="PCA Summary")

3.2 ``add_reduction_embedding``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2-D scatter for ``X_emb.shape == (n, 2)``, 3-D scatter for
``(n, 3)``. Optional class labels color the points; optional metadata
dict adds hover annotations.

.. code-block:: python

   report.add_reduction_embedding(
       X_emb, labels=class_ids,
       metadata={"group": groups, "score": scores},
       name="UMAP embedding",
   )

3.3 ``add_reduction_metrics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quality metrics table (downloadable as CSV) plus a metric bar chart
using :func:`coco_pipe.viz.dim_reduction.plot_metrics`. Accepts either
the reducer (sourced from ``get_metrics()`` or the cached metric
records) or a raw scores list.

.. code-block:: python

   report.add_reduction_metrics(pca, metric="trustworthiness")

3.4 ``add_reduction_diagnostics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shepard diagram: original pairwise distances vs. embedded pairwise
distances on a sampled subset. Requires the original ``X_orig`` and
the embedding ``X_emb``.

.. code-block:: python

   report.add_reduction_diagnostics(X_orig, X_emb)

3.5 ``add_reduction_interpretation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Renders ``correlation`` / ``perturbation`` / ``gradient`` results from
:meth:`DimReduction.interpret`. Accepts either the raw
``get_summary()["interpretation"]`` payload or a list of records.

.. code-block:: python

   summary = pca.get_summary()
   report.add_reduction_interpretation(summary["interpretation"])

3.6 ``add_reduction_coranking``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heatmap of a co-ranking matrix (typically
``DimReduction.diagnostics_["coranking_matrix_"]``).

.. code-block:: python

   report.add_reduction_coranking(diagnostics["coranking_matrix_"])

3.7 ``add_reduction_components``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear reducer components heatmap (PCA, IncrementalPCA, DaskPCA,
DaskTruncatedSVD, DMD, TRCA). Pass ``feature_names`` for axis labels.

.. code-block:: python

   report.add_reduction_components(
       pca.get_components(),
       feature_names=channel_names,
   )

3.8 ``add_reduction_trajectory``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Renders a 3-D trajectory tensor of shape
``(n_trajectories, n_times, n_dims)``. Pair with ``times=`` for
labelled x-axis, ``labels=`` for per-trajectory coloring.

.. code-block:: python

   report.add_reduction_trajectory(
       X_3d, times=time_axis, labels=condition_ids,
   )

3.9 ``add_reduction_trajectory_separation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-pair separation timecourses (e.g., distance between condition
centroids over time) as a multi-line plot. Accepts the output of
:func:`~coco_pipe.dim_reduction.trajectory_separation` directly.

.. code-block:: python

   sep = trajectory_separation(traj, condition_ids, method="centroid")
   report.add_reduction_trajectory_separation(sep, times=time_axis, top_n=5)

3.10 ``add_reduction``
~~~~~~~~~~~~~~~~~~~~~~

Full section in one call: overview + embedding/trajectory plot +
scalar metrics + metric records + every applicable diagnostic. Pass
``X_emb`` to enable embedding/trajectory plots; without it the
section still renders scalar metrics, loss curves, scree, and
co-ranking.

.. code-block:: python

   report.add_reduction(
       pca, name="PCA",
       X_emb=embedding,
       labels=condition_ids,
       metadata={"subject": subject_ids},
       times=time_axis,             # only when X_emb is 3-D
   )

3.11 ``add_comparison``
~~~~~~~~~~~~~~~~~~~~~~~

Multi-method comparison: a metric × method table with best-value
highlighting (:class:`MetricsTableElement`), a metric-by-method
heatmap, a bar chart of details, and — when applicable — a radar
chart normalised across metrics.

.. code-block:: python

   from coco_pipe.dim_reduction.evaluation import MethodSelector

   selector = MethodSelector(reducers).collect()
   report.add_comparison(selector.to_frame(), name="Method Comparison")

Raises ``ValueError`` if the metrics frame is empty after
normalisation.

---

4. ``make_reduction_report``
----------------------------

The factory builds a multi-reducer report by calling adders 1-9 for
each reducer.

.. code-block:: python

   from coco_pipe.report import make_reduction_report

   report = make_reduction_report(
       [pca, umap, phate],
       embeddings=[pca_emb, umap_emb, phate_emb],   # required for embedding sections
       labels=class_ids,
       metadata={"subject": subject_ids},
       times=time_axis,                              # required for 3-D embeddings
       sections="default",
       theme="paper",
       output_path="reduction.html",
   )

4.1 Sections argument
~~~~~~~~~~~~~~~~~~~~~

Valid section names (see :data:`DEFAULT_REDUCTION_SECTIONS`):

.. code-block:: text

   overview, embedding, metrics, diagnostics, coranking,
   interpretation, components, trajectory, trajectory_separation

Unknown names raise ``ValueError``. Pass a subset to render only the
sections you need:

.. code-block:: python

   make_reduction_report(
       [pca, umap],
       embeddings=[pca_emb, umap_emb],
       sections=["overview", "embedding", "metrics"],
   )

4.2 Error tolerance
~~~~~~~~~~~~~~~~~~~

Per-section ``try / except`` mirrors the decoding factory:
unavailable sections are logged and skipped, the rest of the report
still renders.

---

5. Combining With Comparison
----------------------------

The natural full-comparison pattern:

.. code-block:: python

   from coco_pipe.report import make_reduction_report
   from coco_pipe.dim_reduction.evaluation import MethodSelector

   selector = MethodSelector(reducers).collect()
   report = make_reduction_report(
       reducers,
       embeddings=embeddings,
       title="PCA vs UMAP vs PHATE",
   )
   report.add_comparison(selector.to_frame())
   report.save("reduction_comparison.html")

---

6. Internal Helpers (Used by Adders)
------------------------------------

Two helpers in :mod:`coco_pipe.report.dim_reduction` are exposed for
custom adders that need the same normalisation contract:

.. list-table::

   * - ``_get_reducer_summary(reducer)``
     - Return a normalised summary dict (always populates ``method``, ``metrics``, ``metric_records``, ``quality_metadata``, ``diagnostics``, ``interpretation``, ``interpretation_records``, ``capabilities``). Raises if the reducer lacks ``get_summary()``.
   * - ``_metrics_summary_table(metrics)``
     - Pivot a tidy metrics frame into the ``method`` × ``metric`` wide table used by :class:`MetricsTableElement`.
   * - ``_trajectory_times(diagnostics, times)``
     - Resolve the trajectory time axis, preferring an explicit ``times=`` argument over ``diagnostics["trajectory_times_"]``.

These are intentionally underscore-prefixed; their signatures may
change. For custom workflows, prefer the public adders.

.. _report-data-quality:

Data-Quality Checks and Findings
================================

:mod:`coco_pipe.io.quality` contains the dataclass-based
quality checks that run automatically when a
:class:`~coco_pipe.io.DataContainer` is passed to
:meth:`Report.add_container <coco_pipe.report.core.Report.add_container>`,
plus the :class:`~coco_pipe.io.quality.CheckResult` type that
:meth:`Section.add_finding <coco_pipe.report.core.Section.add_finding>`
consumes.

---

1. ``CheckResult``
------------------

Every check returns a :class:`CheckResult` (or a list of them):

.. code-block:: python

   @dataclass
   class CheckResult:
       check_name: str           # e.g. "Missingness"
       status: Literal["OK", "WARN", "FAIL"]
       message: str              # human-readable summary
       severity: int             # 0 (info) to 10 (critical)
       metric_name: str | None = None
       metric_value: float | None = None

       @property
       def is_issue(self) -> bool:
           return self.status in {"WARN", "FAIL"}

When a section accumulates findings, its overall status is the worst
of any single finding's status. ``FAIL`` is sticky: once a section
reaches it, a later ``WARN`` does not downgrade.

---

2. Built-in Checks
------------------

All checks accept either a ``pd.DataFrame`` or a NumPy array and
operate on numeric columns only.

.. list-table::
   :header-rows: 1

   * - Check
     - What it does
   * - :func:`~coco_pipe.io.quality.check_missingness`
     - Fraction of NaN values. ``WARN`` ≥ ``threshold_warn`` (default 0.01), ``FAIL`` ≥ ``threshold_fail`` (default 0.20).
   * - :func:`~coco_pipe.io.quality.check_constant_columns`
     - Per-column near-zero variance. Returns one ``CheckResult`` per offending column; ``FAIL`` for any constant numeric column.
   * - :func:`~coco_pipe.io.quality.check_outliers_zscore`
     - Z-score outlier detection. ``WARN`` when outlier fraction exceeds threshold.
   * - :func:`~coco_pipe.io.quality.check_flatline`
     - Detect zero-variance signal arrays (e.g., flatlined EEG channels).

.. code-block:: python

   from coco_pipe.io.quality import (
       check_missingness, check_constant_columns,
       check_flatline, check_outliers_zscore,
   )

   miss = check_missingness(df)
   constants = check_constant_columns(df)
   outliers = check_outliers_zscore(df, threshold=3.0)
   flat = check_flatline(signal_array, threshold=1e-10)

   for result in [miss, *constants, outliers, flat]:
       print(result.status, result.check_name, result.message)

---

3. Automatic Integration via ``add_container``
----------------------------------------------

The most common path: pass a :class:`DataContainer` to a report and
let the checks fire automatically.

.. code-block:: python

   from coco_pipe.report import Report
   from coco_pipe.io import load_data

   container = load_data("scores.csv", mode="tabular", target_col="label")

   report = Report(title="Input QC")
   report.add_container(container)
   report.save("qc.html")

The added section will contain:

- a metadata table (dims, coords, dtype),
- a ``CalloutElement`` per data-quality finding,
- a histogram or scatter preview of the values,
- automatic section-status upgrades to ``WARN`` / ``FAIL`` when
  findings warrant it.

If the container itself can't be inspected (e.g., wrong shape), the
section is skipped with a ``UserWarning`` rather than raising.

---

4. Adding Custom Findings to Any Section
----------------------------------------

A finding doesn't have to come from a built-in check — you can attach
your own at any point.

.. code-block:: python

   from coco_pipe.report import Section
   from coco_pipe.io.quality import CheckResult

   sec = Section(title="Manual QC")
   sec.add_finding(CheckResult(
       check_name="Subject coverage",
       status="WARN",
       message="Only 14 of 20 subjects have all 3 conditions.",
       severity=5,
       metric_name="subjects_with_full_coverage",
       metric_value=14,
   ))

The section is rendered with a colored finding banner per attached
result, the section header shows the status pill, and the sidebar TOC
flags the section with a colored dot.

---

5. Custom Checks
----------------

A custom check is any function returning a :class:`CheckResult` (or a
list of them):

.. code-block:: python

   def check_class_balance(y, *, threshold_warn=0.1):
       counts = pd.Series(y).value_counts(normalize=True)
       min_share = counts.min()
       status = "WARN" if min_share < threshold_warn else "OK"
       return CheckResult(
           check_name="Class balance",
           status=status,
           message=f"Smallest class share: {min_share:.2%}",
           severity=5 if status == "WARN" else 0,
           metric_name="min_class_share",
           metric_value=float(min_share),
       )

Then plug it into your own section adder or call it directly:

.. code-block:: python

   sec.add_finding(check_class_balance(y))

---

6. Severity Conventions
-----------------------

The numeric ``severity`` field is informational (not used to compute
the displayed status). Useful ranges:

.. list-table::
   :header-rows: 1

   * - ``severity``
     - Meaning
   * - 0
     - Informational; no action.
   * - 1-3
     - Minor; worth noting in the report.
   * - 4-6
     - Warning; investigate before publishing.
   * - 7-9
     - Major; the analysis may be invalid.
   * - 10
     - Critical; downstream results should not be trusted.

Pair high severities with ``status="FAIL"`` for the visual loudness
to match.
