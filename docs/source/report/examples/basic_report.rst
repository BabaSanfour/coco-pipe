.. _report-example-basic:

==================================
Example: Basic Hand-Built Report
==================================

The shortest path from data to a self-contained HTML file: no
factories, just :class:`~coco_pipe.report.core.Report`,
:class:`~coco_pipe.report.core.Section`, and a handful of element
primitives.

**Scientific context**: a Friday afternoon QC report for a freshly
loaded tabular dataset — inspect dimensions, peek at one summary
plot, attach the source script, ship the HTML to a colleague.

---

1. Load the Data
==================

.. code-block:: python

   import numpy as np
   import pandas as pd

   df = pd.read_csv("data/scores.csv")
   summary = df.describe().T.round(3)

---

2. Build a Three-Section Report
=================================

.. code-block:: python

   from coco_pipe.report import Report, Section
   from coco_pipe.report.elements import (
       CalloutElement, CodeBlockElement, MetricsTableElement,
       PlotlyElement, StatCardElement,
   )
   import plotly.express as px

   report = Report(
       title="Score Table QC",
       config={"experiment_id": "EXP-007", "loaded_from": "data/scores.csv"},
   )

   # ----- Section 1: top-line stats ---------------------------------------
   report.add_summary_card({
       "Rows": len(df),
       "Columns": df.shape[1],
       "Mean accuracy": round(df["accuracy"].mean(), 3),
       "Best run": df.sort_values("accuracy").iloc[-1]["run_id"],
   })

   # ----- Section 2: full summary table -----------------------------------
   stats = Section(title="Summary Statistics",
                   description="Per-column describe() output")
   stats.add_element(MetricsTableElement(
       summary,
       highlight_cols=["mean", "max"],
       higher_is_better=["mean", "max"],
   ))
   report.add_section(stats)

   # ----- Section 3: distribution -----------------------------------------
   fig = px.histogram(df, x="accuracy", nbins=30,
                      title="Accuracy distribution")
   dist = Section(title="Distribution")
   dist.add_element(PlotlyElement(fig, height="380px"))
   if df["accuracy"].isna().any():
       dist.add_element(CalloutElement(
           f"{df['accuracy'].isna().sum()} rows have missing accuracy.",
           kind="warning", title="Missing values",
       ))
   report.add_section(dist)

   # ----- Section 4: source code -------------------------------------------
   source_path = "scripts/run_qc.py"
   src = Section(title="Source", code=open(source_path).read())
   src.add_element(CodeBlockElement(open(source_path).read(), language="python"))
   report.add_section(src)

   report.save("qc_report.html")

---

3. What the Reader Sees
=========================

When ``qc_report.html`` opens in a browser:

- The header shows the title, generation timestamp, and a "Run Info"
  button that opens the provenance drawer (git hash, Python version,
  package versions, full configuration JSON).
- The sidebar shows the four sections, with a search box and
  status-filter buttons.
- Section 1 renders four KPI cards across the top of the main area.
- Section 2 highlights the best mean / max per row in green.
- Section 3 lazy-loads the Plotly histogram when it scrolls into
  view; the warning callout (if any) appears above it.
- Section 4's header shows a "Source" button that opens a modal with
  the syntax-highlighted Python file and a copy-to-clipboard action.
- Dark-mode toggle, back-to-top button, and per-section copy-link
  buttons are wired up via delegated handlers in
  ``static/report_scripts.html``.

---

4. Sharing
============

The rendered file is a single ``.html`` (~ 80 KB for this report).
The JS bundles load from CDN by default. To make it openable on a
machine without internet, switch to inline-asset mode:

.. code-block:: python

   report = Report(
       title="Score Table QC",
       config={"experiment_id": "EXP-007"},
       asset_urls="inline",       # downloads + inlines the JS bundles
   )

See :ref:`report-example-offline` for the full offline story.

---

5. Notes
==========

- ``report.add_summary_card`` is a shortcut: it builds one
  :class:`StatCardElement` per key/value pair and places them in a
  :class:`ColumnsElement`. Each card auto-cycles through theme colors.
- Section status is ``"OK"`` by default. Attach a
  :class:`~coco_pipe.report.data_quality.CheckResult` via
  ``sec.add_finding(...)`` to upgrade it to ``"WARN"`` / ``"FAIL"`` and
  surface a colored dot in the sidebar.
- The ``code=`` argument on :class:`Section` powers the "Source"
  button in the section header; the element-level
  :class:`CodeBlockElement` renders the same code inline. Use one,
  the other, or both.
- For one-shot saves you can chain the fluent API:
  ``Report("X").add_section(sec).save("x.html")``.
