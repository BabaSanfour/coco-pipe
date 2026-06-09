.. _report-elements:

==================================
Element Catalog
==================================

:mod:`coco_pipe.report.elements` provides 19 reusable HTML primitives.
Every section adder in the report module — and any custom adder you
write — ultimately renders one or more of these. The primitives are
data-first: they accept Python objects (DataFrames, Plotly figures,
strings, byte buffers) and emit standalone HTML.

All elements share a common contract:

- Subclass of :class:`~coco_pipe.report.elements.Element`.
- ``render() -> str`` returns an HTML fragment.
- ``collect_payload(registry)`` (optional) pushes heavy data into the
  global registry under a UUID; the rendered fragment references that
  UUID via ``data-id``.

---

1. Catalog at a Glance
========================

==========================================  ========================================================
Element                                     Use
==========================================  ========================================================
:class:`~coco_pipe.report.elements.HtmlElement`            Raw HTML string passthrough.
:class:`~coco_pipe.report.elements.ImageElement`           Base64-encoded image from bytes, file,
                                                            ``Path``, or a matplotlib :class:`~matplotlib.figure.Figure`.
:class:`~coco_pipe.report.elements.PlotlyElement`          Plotly figure with lazy hydration.
:class:`~coco_pipe.report.elements.TableElement`           Static HTML table from a DataFrame /
                                                            dict / list.
:class:`~coco_pipe.report.elements.InteractiveTableElement`  Searchable, sortable, paged table with
                                                            CSV export.
:class:`~coco_pipe.report.elements.MetricsTableElement`    Table with per-column "best" highlighting.
:class:`~coco_pipe.report.elements.StatCardElement`        Big-number stat with delta and unit.
:class:`~coco_pipe.report.elements.CalloutElement`         Boxed info/warning/error/success message.
:class:`~coco_pipe.report.elements.CodeBlockElement`       Syntax-tagged ``<pre><code>`` with copy.
:class:`~coco_pipe.report.elements.MarkdownElement`        Markdown → HTML (graceful fallback).
:class:`~coco_pipe.report.elements.BadgeElement`           Pill label with color.
:class:`~coco_pipe.report.elements.ProgressBarElement`     Inline progress bar.
:class:`~coco_pipe.report.elements.TimelineElement`        Ordered list of timestamped events.
:class:`~coco_pipe.report.elements.TabsElement`            Tabbed container holding other elements.
:class:`~coco_pipe.report.elements.AccordionElement`       ``<details>``-backed collapsible section.
:class:`~coco_pipe.report.elements.ColumnsElement`         CSS-grid row of side-by-side elements.
:class:`~coco_pipe.report.elements.ContainerElement`       Base class for any container; itself
                                                            adds ``add_element`` / ``add_markdown``.
:class:`~coco_pipe.report.elements.DownloadAssetElement`   ``<a download>`` button backed by inline
                                                            base64 payload (CSV, JSON, bytes).
:class:`~coco_pipe.report.elements.Element`                Abstract base; subclass for custom widgets.
==========================================  ========================================================

---

2. Data Elements
==================

2.1 ``HtmlElement``
---------------------

Bare HTML wrapper — bypass the typed elements when you need a one-off
fragment.

.. code-block:: python

   from coco_pipe.report.elements import HtmlElement

   sec.add_element(HtmlElement("<div class='text-xs'>raw HTML</div>"))

2.2 ``ImageElement``
----------------------

Embeds any image as a base64 data URI. Accepts:

- ``bytes`` — raw image bytes (any format the browser supports).
- ``str`` or :class:`~pathlib.Path` — read from disk.
- :class:`~matplotlib.figure.Figure` — saved at 150 DPI to PNG.

.. code-block:: python

   from coco_pipe.report.elements import ImageElement

   ImageElement(matplotlib_fig, caption="ROC curve", width="600px")
   ImageElement(Path("logo.png"))
   ImageElement(raw_bytes)         # accepts any browser-supported image format

Every image also emits an inline "Download PNG" link so the reader can
extract the asset.

2.3 ``PlotlyElement``
-----------------------

Plotly figure with lazy hydration: the figure JSON is collected into
the global payload registry; the rendered HTML emits a
``<div class="lazy-plot" data-id="…">`` placeholder. The browser
hydrates each placeholder when it scrolls into view.

.. code-block:: python

   import plotly.graph_objects as go
   from coco_pipe.report.elements import PlotlyElement

   fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
   sec.add_element(PlotlyElement(fig, height="400px"))

PlotlyElement understands Plotly's optimization that base64-encodes
large numeric arrays (``{"dtype": "f8", "bdata": "…"}``) and decodes
them so the rendered JSON is roundtrip-stable.

2.4 ``TableElement``
----------------------

Static HTML table. Accepts a DataFrame, a dict (rendered as one-row
key/value), a list of records, or a dict of lists.

.. code-block:: python

   from coco_pipe.report.elements import TableElement

   TableElement(scores_df, title="Per-fold scores")
   TableElement({"Accuracy": 0.84, "ROC AUC": 0.91})
   TableElement([{"x": 1}, {"x": 2}])

Tables include an inline "Download CSV" button.

2.5 ``InteractiveTableElement``
---------------------------------

Same input as :class:`TableElement` but client-side searchable,
sortable, paged, with per-column "selector" dropdowns. Useful for
exploratory tables with hundreds of rows.

.. code-block:: python

   from coco_pipe.report.elements import InteractiveTableElement

   InteractiveTableElement(
       big_df,
       selector_columns=["dataset", "model"],
       default_sort={"column": "score", "direction": "desc"},
       page_size=25,
   )

2.6 ``MetricsTableElement``
-----------------------------

A :class:`TableElement` that highlights the "best" value in each
metric column. Pass ``higher_is_better=`` to control direction per
column (``True`` / ``False`` / per-column list).

.. code-block:: python

   from coco_pipe.report.elements import MetricsTableElement

   MetricsTableElement(
       scores_df,
       highlight_cols=["accuracy", "log_loss"],
       higher_is_better=["accuracy"],          # lower is better for log_loss
   )

---

3. UI Primitives
==================

3.1 ``StatCardElement``
-------------------------

Large-number card with unit, delta, and a color tag. Use
:meth:`Report.add_summary_card` to render a row of these at the top of
a report.

.. code-block:: python

   from coco_pipe.report.elements import StatCardElement

   StatCardElement("Accuracy", 0.95, unit="%", delta="+2.1%", color="green")

3.2 ``CalloutElement``
------------------------

Boxed message with an icon. ``kind`` is one of ``"info"``,
``"warning"``, ``"error"``, ``"success"``.

.. code-block:: python

   from coco_pipe.report.elements import CalloutElement

   CalloutElement("Result includes 3 outliers.", kind="warning",
                  title="Outliers detected")

3.3 ``CodeBlockElement``
--------------------------

Syntax-tagged ``<pre><code>`` block. Adds a "Copy" button when
``copyable=True``.

.. code-block:: python

   from coco_pipe.report.elements import CodeBlockElement

   CodeBlockElement(
       'reducer.fit(X, y=labels)',
       language="python",
       title="Reproducer",
       copyable=True,
   )

3.4 ``MarkdownElement``
-------------------------

Markdown → HTML using the ``markdown`` package when available; falls
back to ``<pre class="whitespace-pre-wrap">`` rendering when not.

.. code-block:: python

   from coco_pipe.report.elements import MarkdownElement

   MarkdownElement("# Notes\n* Item 1\n* Item 2")

3.5 ``BadgeElement`` / ``ProgressBarElement`` / ``TimelineElement``
---------------------------------------------------------------------

.. code-block:: python

   from coco_pipe.report.elements import (
       BadgeElement, ProgressBarElement, TimelineElement,
   )

   BadgeElement("Experimental", color="red")
   ProgressBarElement(value=75, max_value=100, label="Accuracy", color="green")
   TimelineElement([
       {"title": "Start",  "time": "10:00", "description": "Fit reducer"},
       {"title": "End",    "time": "10:42", "description": "Score complete",
        "status": "green"},
   ])

---

4. Layout Primitives
======================

4.1 ``TabsElement``
---------------------

Tabbed container; keys are tab labels, values are any element.

.. code-block:: python

   from coco_pipe.report.elements import TabsElement, PlotlyElement

   TabsElement({
       "ROC":    PlotlyElement(roc_fig),
       "PR":     PlotlyElement(pr_fig),
       "Calibr": PlotlyElement(cal_fig),
   })

4.2 ``AccordionElement``
--------------------------

Collapsible section backed by ``<details>``.

.. code-block:: python

   from coco_pipe.report.elements import AccordionElement, CodeBlockElement

   acc = AccordionElement("Show training script", open=False)
   acc.add_element(CodeBlockElement(open("train.py").read(), language="python"))
   sec.add_element(acc)

4.3 ``ColumnsElement``
------------------------

CSS-grid row. Sub-elements are placed side by side.

.. code-block:: python

   from coco_pipe.report.elements import ColumnsElement, PlotlyElement

   ColumnsElement(
       [PlotlyElement(roc_fig), PlotlyElement(pr_fig)],
       cols=2, gap="gap-4",
   )

:meth:`Section.add_columns` is a shortcut for the same pattern.

4.4 ``ContainerElement``
--------------------------

Base class. :class:`Section` and :class:`Report` inherit from it.
Useful when subclassing for custom containers.

---

5. Download Helpers
=====================

``DownloadAssetElement`` embeds binary or text payloads as base64 data
and produces a ``<a download>`` link. Useful for shipping the source
data alongside the chart.

.. code-block:: python

   from coco_pipe.report.elements import DownloadAssetElement

   DownloadAssetElement(
       data=csv_bytes,
       filename="scores.csv",
       mime_type="text/csv",
       label="Download per-fold scores",
       style="gray",
   )

The payload is captured into the global registry and decoded
on-demand when the user clicks; the HTML stays small.

---

6. Custom Elements
====================

To add a new element type, subclass
:class:`~coco_pipe.report.elements.Element`:

.. code-block:: python

   from coco_pipe.report.elements import Element

   class KPIElement(Element):
       def __init__(self, label, value):
           self.label, self.value = label, value

       def render(self) -> str:
           return (
               f"<div class='kpi'>"
               f"<span class='label'>{self.label}</span>"
               f"<span class='value'>{self.value}</span>"
               f"</div>"
           )

If your element holds heavy data, override ``collect_payload`` and use
``data-id="…"`` in the rendered fragment. See
:class:`PlotlyElement` and :class:`DownloadAssetElement` for canonical
implementations.

See :ref:`report-extensions` for binding custom adders to ``Report``.
