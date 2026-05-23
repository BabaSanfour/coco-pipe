.. _report-core:

============================
``Section`` and ``Report``
============================

:class:`~coco_pipe.report.core.Section` and
:class:`~coco_pipe.report.core.Report` are the two container classes
that hold the tree of HTML elements assembled by every other module.

---

1. Quick Reference
====================

.. code-block:: python

   from coco_pipe.report import Report, Section
   from coco_pipe.report.elements import PlotlyElement, TableElement

   report = Report(title="My Analysis")

   sec = Section(title="Performance", description="CV scores per fold")
   sec.add_element(TableElement(scores_df))
   sec.add_element(PlotlyElement(roc_figure))
   report.add_section(sec)

   report.save("performance.html")

Every method on :class:`Report` and :class:`Section` returns ``self``
so they chain:

.. code-block:: python

   Report("Demo").add_container(c).add_decoding_overview(result).save("r.html")

---

2. ``Section``
================

A logical group of elements with a title, optional icon, optional
description, tags (for sidebar filtering), and a status (``"OK"`` /
``"WARN"`` / ``"FAIL"``).

.. code-block:: python

   Section(
       title="Quality Findings",
       icon="⚠",
       tags=["quality", "preprocessing"],
       status="WARN",
       description="2 channels flatlined during run 02.",
       code="reducer.fit(X)",  # rendered behind a "Source" modal
       metadata={"input": "X.npy", "rows": 1024},
   )

.. rubric:: Methods

==================================  ================================================
``add_element(element)``             Append any :class:`Element` or raw HTML string.
``add_columns(elements, cols=?)``    Render children in a CSS grid row.
``add_finding(check_result)``        Attach a :class:`CheckResult`; section status
                                     auto-upgrades to ``WARN`` / ``FAIL``.
``render() -> str``                  Render the section to HTML (called by Report).
==================================  ================================================

.. rubric:: Status semantics

``FAIL`` is sticky. Once a section's status is ``FAIL`` it cannot be
downgraded by a later ``WARN``. The sidebar marks any
``WARN`` / ``FAIL`` section with a colored dot and surfaces an
"Attention Needed" summary at the bottom of the sidebar.

---

3. ``Report``
================

The top-level container. Owns the asset URLs, theme, configuration,
provenance metadata, and the section list.

.. code-block:: python

   Report(
       title="CoCo Analysis Report",
       config={"experiment_name": "demo"},   # dict or ReportConfig
       theme="paper",                        # "paper" | "notebook" | "poster"
       asset_urls=None,                      # None | dict | "inline"
   )

.. rubric:: Construction

The constructor delegates to three helpers (kept lazy and side-effect-free
in their own right):

==============================  ============================================
``_apply_theme(theme)``          Sets matplotlib rcParams and the Plotly
                                ``coco`` template for any figures rendered
                                during the session.
``_resolve_assets(asset_urls)``  Returns the ``(urls, mode)`` tuple that
                                drives ``<script src=…>`` versus inlined
                                ``<script>…</script>``. See
                                :ref:`report-assets`.
``_resolve_config(config)``      Coerces ``config`` into a
                                :class:`ReportConfig`, honoring an explicit
                                ``title`` argument.
==============================  ============================================

.. rubric:: Section adders

================================================  =====================================
Method                                            Use
================================================  =====================================
:meth:`Report.add_section(section)`               Append a pre-built :class:`Section`.
                                                  IDs are auto-uniqued on collision.
:meth:`Report.add_figure(fig, caption=)`          Shortcut: wraps a matplotlib /
                                                  Plotly figure in its own section.
:meth:`Report.add_container(container, ...)`      Inspect a :class:`~coco_pipe.io.DataContainer`:
                                                  dimensions, coordinates, missingness,
                                                  flatline / outlier checks, sample
                                                  histograms.
:meth:`Report.add_raw_preview(data, name=)`       Interactive Plotly preview for raw
                                                  arrays or :class:`DataContainer` X.
:meth:`Report.add_summary_card(metrics)`          Top-of-report key/value stat strip
                                                  (e.g., ``{"Accuracy": 0.83, "Folds": 5}``).
:meth:`Report.add_markdown(text)`                 Append a markdown block (inherited
                                                  from :class:`ContainerElement`).
:meth:`Report.add_element(element)`               Append any :class:`Element` outside a
                                                  section.
================================================  =====================================

Domain-specific adders (``add_decoding_*``, ``add_reduction_*``,
``add_reduction``, ``add_comparison``) are bound at import time from
:mod:`~coco_pipe.report.decoding` and :mod:`~coco_pipe.report.dim_reduction`
— see :ref:`report-section-decoding` and
:ref:`report-section-dim-reduction`.

.. rubric:: Rendering

================================  ============================================
``render() -> str``                Collect payload + render full HTML.
``save(filename)``                 Render and write to disk.
``show(port=None)``                Open the rendered report in a new browser
                                  tab via the stdlib ``http.server``.
``_repr_html_()``                  Allows ``report`` to display inline in
                                  Jupyter notebooks.
================================  ============================================

The rendered HTML is fully self-contained: heavy data (Plotly figures,
interactive tables) is collected into one gzip+base64 payload embedded
in the page (see :ref:`report-concepts`).

---

4. Common Patterns
====================

4.1 One-shot save
-------------------

.. code-block:: python

   from coco_pipe.report import Report
   from coco_pipe.report.elements import PlotlyElement

   Report("Quick").add_element(PlotlyElement(fig)).save("quick.html")

4.2 Build a multi-section report by hand
------------------------------------------

.. code-block:: python

   from coco_pipe.report import Report, Section
   from coco_pipe.report.elements import (
       CalloutElement, CodeBlockElement, PlotlyElement, TableElement,
   )

   report = Report(title="Cross-validation comparison")

   overview = Section(title="Overview", description="3 models, 5-fold CV")
   overview.add_element(CalloutElement(
       "All models trained on the same splits; see Provenance for details.",
       kind="info",
   ))
   overview.add_element(TableElement(scores_summary_df))
   report.add_section(overview)

   plots = Section(title="ROC + Calibration")
   plots.add_columns([
       PlotlyElement(roc_fig, height="380px"),
       PlotlyElement(cal_fig, height="380px"),
   ])
   report.add_section(plots)

   src = Section(title="Source", code=open("script.py").read())
   src.add_element(CodeBlockElement(open("script.py").read(), language="python"))
   report.add_section(src)

   report.save("comparison.html")

4.3 Display inline in a notebook
----------------------------------

.. code-block:: python

   report  # _repr_html_ auto-displays the rendered HTML in Jupyter

4.4 Serve over HTTP
---------------------

.. code-block:: python

   report.show()         # picks a free port, opens browser
   report.show(port=8123)

---

5. State and Identity
=======================

- Each section is given an HTML id derived from its title (slugified).
  Collisions are auto-suffixed (``"repeated"`` → ``"repeated-2"``).
- The collected payload registry uses element-level UUIDs; nothing in
  the HTML hard-codes a section count or position.
- The TOC sidebar is built from the section tree; it updates the
  active link via an ``IntersectionObserver`` and sets
  ``aria-current="location"`` on the active link for screen readers.

---

6. Failure Modes
==================

================================  ========================================================
Action                            Behavior
================================  ========================================================
Missing reducer ``get_summary``   :meth:`Report.add_reduction` raises ``TypeError``.
Empty comparison frame            :meth:`Report.add_comparison` raises ``ValueError``.
Broken ``DataContainer``          :meth:`Report.add_container` emits a ``UserWarning``
                                  and skips the section.
Section produces an error         The factories (``make_decoding_report`` /
                                  ``make_reduction_report``) log the failure and skip
                                  that section; the rest of the report still renders.
================================  ========================================================

The factory functions are deliberately permissive: a single bad
section never blocks the rest of the report.
