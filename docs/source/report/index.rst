.. _report:

=======
Reports
=======

The ``coco_pipe.report`` module assembles single-file HTML reports from
:ref:`decoding <decoding>` results, :ref:`dim-reduction <dim-reduction>`
runs, raw data containers, and any combination thereof. Reports are
**self-contained** (everything in one ``.html``), **interactive**
(Plotly-driven plots, search, dark mode), and **lineage-aware**
(automatic environment + git provenance).

.. admonition:: Design Philosophy

   A report is a tree of :class:`~coco_pipe.report.core.Section` containers
   holding :class:`~coco_pipe.report.elements.Element` primitives. Heavy
   data (Plotly figures, interactive tables) is collected into a single
   gzip+base64 payload embedded in the HTML, then decompressed in the
   browser on demand. The same HTML opens identically online or offline
   when assets are inlined.

.. rubric:: Key Features

- One-shot factories (``from_experiment_result``, ``from_reductions``,
  ``from_container``, ``from_bids``, …) for the common cases.
- 19 reusable HTML primitives in :mod:`~coco_pipe.report.elements`:
  images, Plotly figures, tables, code blocks, tabs, accordions,
  download assets, etc.
- 9 decoding section adders + 11 dim-reduction section adders, attached
  as fluent methods on ``Report``.
- Automatic provenance: git hash, package versions, Python/OS, command,
  timestamp.
- Strict pydantic configs (:class:`ReportConfig`, :class:`ProvenanceConfig`).
- Data-quality checks (missingness, flatline, outliers, constant
  columns) run inline when a ``DataContainer`` is added.
- Three asset modes: CDN (default), self-hosted URLs, or **inline**
  (downloads bundles once, embeds them in the HTML — fully offline).

---

.. rubric:: Quickstart

The fastest path: pass a result to one of the factory functions.

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   report = from_experiment_result(result, output_path="decoding.html")

Build a report incrementally with the fluent API:

.. code-block:: python

   from coco_pipe.report import Report

   report = (
       Report(title="My Analysis")
       .add_container(container)
       .add_decoding_overview(result)
       .add_decoding_performance(result)
       .add_decoding_temporal(result)
   )
   report.save("report.html")

Make the rendered HTML offline-openable (downloads and inlines the JS
bundles on first run):

.. code-block:: python

   report = Report(title="Air-gapped", asset_urls="inline")
   report.add_reduction(pca_reducer, X_emb=embedding)
   report.save("offline.html")

---

.. rubric:: How to Choose an Entry Point

==========================================  ============================================================
You have…                                    Reach for…
==========================================  ============================================================
An ``ExperimentResult``                      :func:`~coco_pipe.report.from_experiment_result` or
                                             :func:`~coco_pipe.report.make_decoding_report`
A list of scored ``DimReduction`` objects    :func:`~coco_pipe.report.from_reductions` or
                                             :func:`~coco_pipe.report.make_reduction_report`
A :class:`~coco_pipe.io.DataContainer`       :func:`~coco_pipe.report.from_container`
A BIDS dataset directory                     :func:`~coco_pipe.report.from_bids`
A tabular file (CSV/parquet/…)               :func:`~coco_pipe.report.from_tabular`
A raw embedding array                        :func:`~coco_pipe.report.from_embeddings`
Multiple reports to combine                  :func:`~coco_pipe.report.merge_reports`
Hand-built sections                          :class:`~coco_pipe.report.Report` +
                                             :mod:`~coco_pipe.report.elements`
==========================================  ============================================================

---

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide
   sections
   elements
   advanced
