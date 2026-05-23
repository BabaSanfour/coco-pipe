.. _report-api:

============================================
Factories and Assembly
============================================

:mod:`coco_pipe.report.api` exposes seven top-level helpers — six
``from_*`` factories that produce a populated :class:`Report` from a
domain object, plus :func:`merge_reports` for combining existing
reports.

These are convenience entry points. Internally they call the same
``add_*`` methods documented in :ref:`report-core`,
:ref:`report-section-decoding`, and
:ref:`report-section-dim-reduction`.

---

1. Factory at a Glance
========================

==========================================  =====================================================
Factory                                      Input                                            Output sections
==========================================  =====================================================
:func:`~coco_pipe.report.from_container`     :class:`~coco_pipe.io.DataContainer`             Data overview + optional raw preview
:func:`~coco_pipe.report.from_bids`          BIDS root directory                              Data overview from loaded container
:func:`~coco_pipe.report.from_tabular`       Path to CSV / parquet / etc.                     Data overview from loaded container
:func:`~coco_pipe.report.from_embeddings`    ``X_emb`` array                                 Standalone embedding section
:func:`~coco_pipe.report.from_reductions`    list of scored :class:`DimReduction`             One section per reducer + optional comparison
:func:`~coco_pipe.report.from_experiment_result`  :class:`ExperimentResult`                  Full decoding report
:func:`~coco_pipe.report.merge_reports`      ``*Report`` arguments                            Sections of every input, prefixed
==========================================  =====================================================

Every factory accepts the standard report kwargs: ``title``,
``config``, ``theme``, ``asset_urls``, ``output_path``.

---

2. Container Factories
========================

2.1 ``from_container``
------------------------

The simplest path: render a :class:`DataContainer` as a report.

.. code-block:: python

   from coco_pipe.io import load_data
   from coco_pipe.report import from_container

   container = load_data("scores.csv", mode="tabular", target_col="label")
   report = from_container(
       container,
       title="Input Inspection",
       raw_preview=True,           # adds an interactive scroller
       output_path="input.html",
   )

The single "Data Overview" section includes dimensions, coordinates,
data-quality findings (missingness, flatline, outliers, constant
columns — see :ref:`report-data-quality`), and a sample histogram.

2.2 ``from_bids``
-------------------

Wraps :func:`coco_pipe.io.load_data` in BIDS mode + ``from_container``.

.. code-block:: python

   from coco_pipe.report import from_bids

   report = from_bids(
       root="/data/bids_root",
       task="resting",
       output_path="bids_overview.html",
   )

2.3 ``from_tabular``
----------------------

Wraps :func:`coco_pipe.io.load_data` in tabular mode + ``from_container``.

.. code-block:: python

   from coco_pipe.report import from_tabular

   report = from_tabular(
       path="data/scores.csv",
       sep=",",
       target_col="label",
       title="Score Table Inspection",
   )

2.4 ``from_embeddings``
-------------------------

Standalone embedding plot — useful when you have an embedding array
but no fitted :class:`DimReduction` to ship alongside.

.. code-block:: python

   from coco_pipe.report import from_embeddings

   report = from_embeddings(
       X_emb=embedding,
       labels=class_ids,
       metadata={"subject": subject_ids},
       title="UMAP embedding",
   )

---

3. Result Factories
=====================

3.1 ``from_reductions``
-------------------------

Builds a multi-reducer report. Pass ``container=`` to inject a "Data
Overview" section, ``embeddings=`` to enable embedding/trajectory
plots, ``raw_preview=True`` to add a raw-data scroller after the data
overview.

.. code-block:: python

   from coco_pipe.report import from_reductions

   report = from_reductions(
       reductions=[pca, umap, phate],
       container=container,
       embeddings=[pca_emb, umap_emb, phate_emb],
       labels=container.y,
       title="PCA vs UMAP vs PHATE",
       output_path="reduction.html",
   )

Equivalent to :func:`make_reduction_report` plus an extra
``add_container`` call.

3.2 ``from_experiment_result``
--------------------------------

Builds a full decoding report from an
:class:`~coco_pipe.decoding.result.ExperimentResult`.

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   report = from_experiment_result(
       result,
       feature_metadata=meta_df,      # for sensor topomap sections
       info=raw.info,                 # for topomap sensor coordinates
       title="Decoding Report",
       output_path="decoding.html",
   )

Identical surface to :func:`make_decoding_report`; choose whichever
import path reads better in your script.

---

4. ``merge_reports``
======================

Combine multiple reports for cross-run / cross-cohort comparison.
Every source section is deep-copied, its title prefixed with the
source report's title, and appended to the merged output in the order
supplied.

.. code-block:: python

   from coco_pipe.report import from_experiment_result, merge_reports

   r1 = from_experiment_result(result_cohort_a, title="Cohort A")
   r2 = from_experiment_result(result_cohort_b, title="Cohort B")

   merged = merge_reports(r1, r2, title="Cross-Cohort Comparison")
   merged.save("comparison.html")

Raises ``ValueError`` if fewer than two reports are supplied.

The merged report inherits the provenance, theme, and asset mode of
the first input. Override by setting them on the merged report before
saving:

.. code-block:: python

   merged = merge_reports(r1, r2)
   merged.theme = "poster"
   merged.asset_mode = "inline"      # rebuild urls accordingly with _resolve_assets

---

5. Argument Conventions
=========================

==================================  ============================================================
Argument                            Purpose
==================================  ============================================================
``title``                           Report title; shown in the header and the browser tab.
``config``                          Dict of free-form parameters. Stored on the
                                    :class:`ReportConfig` for the Run Info drawer.
``theme``                           ``"paper"`` (default) / ``"notebook"`` / ``"poster"``.
                                    Mirrors :mod:`coco_pipe.viz.theme` modes.
``asset_urls``                      ``None`` / ``dict`` / ``"inline"``. See :ref:`report-assets`.
``output_path``                     Convenience: when set, the factory also calls
                                    :meth:`Report.save(output_path)` and returns the report.
``sections``                        ``"default"`` or list of section names. Names are
                                    factory-specific (see the section catalogs).
==================================  ============================================================

When in doubt, prefer the factory functions over the lower-level
``make_*_report`` functions — they accept everything ``make_*_report``
does, plus optional ``container=`` / ``raw_preview=`` for additional
inspection sections.
