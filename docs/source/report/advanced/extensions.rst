.. _report-extensions:

==================================
Custom Elements and Adders
==================================

The report module is built so that custom widgets and custom section
adders compose with the shipped ones using the same patterns. This
page walks through the three common extension points.

---

1. Custom :class:`Element` Subclasses
=======================================

Every primitive in :mod:`~coco_pipe.report.elements` is a subclass of
the abstract :class:`~coco_pipe.report.elements.Element`. To add your
own, override two methods at most.

1.1 Stateless element
-----------------------

If your element doesn't hold heavy data, ``render()`` is all you need:

.. code-block:: python

   from coco_pipe.report.elements import Element

   class KPICard(Element):
       """Big stat with a sub-label, no payload registration."""

       def __init__(self, label: str, value: str, sub: str = ""):
           self.label, self.value, self.sub = label, value, sub

       def render(self) -> str:
           return (
               "<div class='rounded-lg border p-4 bg-white'>"
               f"  <div class='text-xs uppercase text-gray-500'>{self.label}</div>"
               f"  <div class='text-3xl font-bold mt-1'>{self.value}</div>"
               f"  <div class='text-xs text-gray-400 mt-1'>{self.sub}</div>"
               "</div>"
           )

Use Tailwind utility classes for styling (they're available because of
the inline Tailwind config) or your own CSS via a
:class:`~coco_pipe.report.elements.HtmlElement` ``<style>`` block.

1.2 Lazy-loaded element with payload
--------------------------------------

For heavy data that should not be rendered upfront, push the data
into the global registry and emit a placeholder ``<div data-id="…">``
in ``render()``. The shipped
:class:`~coco_pipe.report.elements.PlotlyElement` is the canonical
example.

.. code-block:: python

   import uuid
   from coco_pipe.report.elements import Element

   class HeavyHistogram(Element):
       def __init__(self, values):
           self.values = list(values)
           self.registry_id = str(uuid.uuid4())

       def collect_payload(self, registry):
           registry[self.registry_id] = {"values": self.values}

       def render(self) -> str:
           return (
               f"<div class='lazy-histogram' data-id='{self.registry_id}' "
               f"style='min-height: 220px;'></div>"
           )

Then a small browser-side hook (added via a template override) reads
``REPORT_DATA[dataId]`` and renders the histogram on intersection.
See :ref:`report-templates` for how to override the JS partial.

---

2. Custom :class:`Section` Subclasses
=======================================

If a Section needs section-specific markup (extra status badges, a
custom header strip), subclass and point at your own template:

.. code-block:: python

   from coco_pipe.report.core import Section
   from coco_pipe.report._engine import render_template

   class GraphSection(Section):
       """Section with a Graphviz preview baked into the header."""

       def __init__(self, title, dot_source, **kwargs):
           super().__init__(title=title, **kwargs)
           self.dot_source = dot_source

       def render(self) -> str:
           return render_template(
               "partials/graph_section.html",   # in your overrides
               title=self.title,
               id=self.id,
               status=self.status,
               dot_source=self.dot_source,
               content=self.render_children(),
           )

Most users won't need this — the default ``section.html`` is general
enough that a custom :class:`Element` slotted into a normal Section is
usually cleaner.

---

3. Custom ``add_*`` Methods on :class:`Report`
================================================

The shipped ``add_decoding_*`` and ``add_reduction_*`` methods are
defined as top-level functions and attached to :class:`Report` at
import time. Custom adders follow the same pattern — no subclassing
needed.

3.1 Define the function
-------------------------

The first argument's name doesn't matter, but typing it as
:class:`Report` makes the binding obvious:

.. code-block:: python

   # my_pkg/report_adders.py
   from coco_pipe.report.core import Report, Section
   from coco_pipe.report.elements import PlotlyElement, TableElement

   def add_my_summary(self: Report, my_result, *, name: str = "My Summary") -> Report:
       """Add a custom summary section to *self*."""
       sec = Section(title=name)
       sec.add_element(TableElement(my_result.to_frame(), title="Top metrics"))
       sec.add_element(PlotlyElement(my_result.plot(), height="380px"))
       self.add_section(sec)
       return self

3.2 Attach to :class:`Report`
-------------------------------

At the bottom of the module:

.. code-block:: python

   Report.add_my_summary = add_my_summary

3.3 Eager-load at package init
--------------------------------

If users import ``Report`` directly from ``coco_pipe.report.core``,
the binding only happens once your module is imported. The shipped
adders solve this by eager-loading in :mod:`coco_pipe.report.__init__`.
For your own package, do the same in your top-level ``__init__.py``:

.. code-block:: python

   # my_pkg/__init__.py
   from . import report_adders as _report_adders     # noqa: F401

   # users can now do:
   from coco_pipe.report import Report
   import my_pkg                                     # triggers binding
   Report().add_my_summary(result)

3.4 Why the monkey-patch pattern?
-----------------------------------

The alternative — putting every adder method directly on
:class:`Report` — forces ``core.py`` to import every domain module
(decoding, dim-reduction, your custom one). That recreates the
circular-import problem the report module was specifically refactored
to avoid. The bound-function pattern keeps :mod:`coco_pipe.report.core`
ignorant of any domain.

---

4. Custom Factory Functions
=============================

For one-shot workflows ("build a report from object X"), follow the
:mod:`coco_pipe.report.api` pattern:

.. code-block:: python

   from pathlib import Path
   from coco_pipe.report import Report

   def from_my_result(
       my_result,
       *,
       title: str = "My Workflow",
       theme: str = "paper",
       asset_urls=None,
       output_path: str | Path | None = None,
   ) -> Report:
       """Build a report from a MyResult object."""
       report = Report(title=title, theme=theme, asset_urls=asset_urls)
       report.add_my_summary(my_result)
       # ... more custom sections ...
       if output_path is not None:
           report.save(output_path)
       return report

Keep the signature aligned with the shipped factories
(``title=``, ``theme=``, ``asset_urls=``, ``output_path=``) so users
don't have to relearn the convention.

---

5. Custom Data-Quality Checks
===============================

Any function returning a :class:`~coco_pipe.report.data_quality.CheckResult`
(or a list of them) plugs into the existing finding system:

.. code-block:: python

   import pandas as pd
   from coco_pipe.report.data_quality import CheckResult

   def check_class_balance(y, *, threshold_warn: float = 0.10) -> CheckResult:
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

   sec.add_finding(check_class_balance(y))

The section's status automatically upgrades to match the worst
finding. See :ref:`report-data-quality` for the full ``CheckResult``
contract.

---

6. End-to-End Custom Workflow
===============================

Putting it all together:

.. code-block:: python

   # my_pkg/report_adders.py
   from coco_pipe.report.core import Report, Section
   from coco_pipe.report.elements import (
       PlotlyElement, TableElement, CalloutElement,
   )
   from coco_pipe.report.data_quality import CheckResult

   def add_my_summary(self: Report, payload, *, name: str = "My Summary") -> Report:
       sec = Section(title=name, description="Per-run summary")
       if payload.has_warnings():
           sec.add_finding(CheckResult(
               "Run integrity", "WARN",
               f"{len(payload.warnings)} runtime warnings", 4,
           ))
       sec.add_element(TableElement(payload.summary_frame()))
       sec.add_element(PlotlyElement(payload.summary_plot()))
       self.add_section(sec)
       return self

   Report.add_my_summary = add_my_summary

.. code-block:: python

   # my_pkg/__init__.py
   from . import report_adders as _report_adders     # noqa: F401

.. code-block:: python

   # User script
   import my_pkg
   from coco_pipe.report import Report

   Report("My Workflow").add_my_summary(payload).save("custom.html")

---

7. Testing Custom Adders
==========================

Mirror the test pattern used by the shipped adders
(``tests/test_report_dimred.py``):

.. code-block:: python

   from coco_pipe.report import Report
   from my_pkg.report_adders import add_my_summary       # function form is testable

   def test_my_summary_function():
       rep = Report()
       add_my_summary(rep, make_fake_payload())
       assert len(rep.children) == 1
       assert "My Summary" in rep.children[-1].render()

   def test_my_summary_bound_to_report():
       """Ensures the import-time binding fired."""
       import my_pkg                                       # noqa: F401
       assert hasattr(Report, "add_my_summary")
