.. _report-advanced:

===============
Advanced Topics
===============

.. _report-templates:

Templates and Layout Customization
==================================

The report HTML is produced by Jinja templates living under
``coco_pipe/report/templates/``. The default layout is opinionated but
intentionally easy to extend or replace — you can override individual
partials, swap the entire base template, or build a fresh layout from
the published Element catalog.

---

1. Template Layout
------------------

After the file-split refactor, the templates tree is::

   coco_pipe/report/templates/
   ├── base.html                  thin orchestrator (~70 lines)
   ├── section.html               renders one Section
   ├── partials/
   │   ├── header.html            top nav bar
   │   ├── sidebar.html           TOC, search, filter, findings summary
   │   ├── report_summary.html    counts strip at top of main
   │   └── info_drawer.html       slide-out Run Info panel
   └── static/
       ├── tailwind_config.html   inline Tailwind config <script>
       ├── print_styles.html      print-only @media CSS
       └── report_scripts.html    main behavior <script> (delegated
                                  handlers, plot lazy-load, search,
                                  CSV export, dark mode)

:func:`coco_pipe.report._engine.render_template` resolves templates
through a single Jinja :class:`~jinja2.FileSystemLoader` rooted at
``templates/``. Includes use forward-slash paths regardless of OS:

.. code-block:: jinja

   {% include 'partials/header.html' %}

---

2. Template Variables
---------------------

Every render of ``base.html`` is called with this context (see
:meth:`Report.render <coco_pipe.report.core.Report.render>`):

.. list-table::
   :header-rows: 1

   * - Variable
     - Type / source
   * - ``title``
     - ``Report.title``
   * - ``content``
     - Pre-rendered HTML from the section tree (``| safe``)
   * - ``timestamp``
     - ``Report.timestamp``
   * - ``toc``
     - List of ``{id, title, icon, status}`` per section
   * - ``metadata``
     - ``Report.config.provenance.model_dump()``
   * - ``config``
     - ``Report.config.model_dump_json(indent=2)``
   * - ``payload``
     - gzip+base64 string of the global data registry
   * - ``report_summary``
     - Dict of ``sections``/``warnings``/``failures``/ ``plot_count``/``table_count``/etc.
   * - ``asset_urls``
     - Dict; in inline mode, holds JS *contents*
   * - ``asset_mode``
     - ``"cdn"`` / ``"custom"`` / ``"inline"``

``section.html`` (called once per :class:`Section`) receives:

.. list-table::
   :header-rows: 1

   * - Variable
     - Type / source
   * - ``title``, ``id``, ``icon``, ``status``, ``description``, ``metadata``, ``tags``, ``code``, ``findings``
     - Section attributes
   * - ``content``
     - Pre-rendered HTML from section children (``| safe``)

---

3. Overriding a Single Partial
------------------------------

The cleanest extension point. The default loader is a
``FileSystemLoader``; swap it for a
``ChoiceLoader([FileSystemLoader(your_dir), FileSystemLoader(default_dir)])``
so your overrides win.

.. code-block:: python

   from pathlib import Path
   import jinja2
   from coco_pipe.report import _engine

   custom_dir = Path("my_report_overrides")        # contains partials/header.html
   default_dir = _engine.TEMPLATE_DIR

   _engine.get_env.cache_clear()                   # drop the cached env
   def _patched_env() -> jinja2.Environment:
       return jinja2.Environment(
           loader=jinja2.ChoiceLoader([
               jinja2.FileSystemLoader(str(custom_dir)),
               jinja2.FileSystemLoader(str(default_dir)),
           ]),
           autoescape=jinja2.select_autoescape(["html", "xml"]),
           trim_blocks=True,
           lstrip_blocks=True,
       )
   _engine.get_env = _patched_env

Now any ``my_report_overrides/partials/header.html`` overrides the
shipped partial; everything else falls through to the defaults. This
keeps your customization out of the package's source tree.

.. tip::

   If you need this often, wrap the snippet above in a small helper
   (``with_template_overrides(path)``) and call it once at module
   import.

---

4. Replacing ``base.html`` Entirely
-----------------------------------

The base template uses Jinja blocks so callers can subclass it
rather than fork it. Currently only one block is exposed:

.. code-block:: jinja

   {% block content %}
   <div class="space-y-6" id="main-content-area">
       {{ content | safe }}
   </div>
   {% endblock %}

A custom ``base.html`` can extend the default and override the block:

.. code-block:: jinja

   {% extends "base.html" %}

   {% block content %}
   <header class="my-custom-banner">Quarterly Review</header>
   {{ super() }}
   <footer class="my-custom-footer">© 2026 Lab Name</footer>
   {% endblock %}

Drop the file at ``my_report_overrides/base.html`` and the loader
override above picks it up.

---

5. Modifying the Inline JS / CSS
--------------------------------

The three ``static/*.html`` files are plain ``<script>`` and
``<style>`` blocks that the base template ``{% include %}``-s. To
patch behavior without forking:

1. Create ``my_report_overrides/static/report_scripts.html`` with your
   modified ``<script>...</script>`` content.
2. Apply the loader override from §3.

Common tweaks:

- Override the dark-mode default (search for ``color-theme``).
- Change the Plotly modeBar buttons (search for ``modeBarButtonsToAdd``).
- Adjust the lazy-plot intersection rootMargin (``200px`` default).

---

6. Custom Section Templates
---------------------------

The default ``section.html`` works for every Element type because all
content goes through ``{{ content | safe }}``. If you need
section-specific markup (different status pill colors, an extra
metadata strip), override ``section.html`` using the same loader
trick.

Alternative: subclass :class:`~coco_pipe.report.core.Section` and
override ``render()`` to point at your own template:

.. code-block:: python

   from coco_pipe.report.core import Section
   from coco_pipe.report._engine import render_template

   class FlaggedSection(Section):
       """A Section that renders with our 'flagged' template."""

       def render(self) -> str:
           return render_template(
               "partials/flagged_section.html",
               title=self.title,
               id=self.id,
               status=self.status,
               flag_color=getattr(self, "flag_color", "amber"),
               content=self.render_children(),
           )

The new ``partials/flagged_section.html`` lives in your overrides
directory (or in the package once merged); Jinja resolves it through
the loader chain.

---

7. Theming and Tailwind
-----------------------

The template uses Tailwind's Play CDN, which JITs CSS from the markup
at load time. To customize colors, edit ``static/tailwind_config.html``
— specifically the ``brand`` and ``coco`` palettes in
``tailwind.config.theme.extend.colors``.

For production deployments where the runtime CSS compiler isn't an
option:

1. Pre-build the CSS bundle using the Tailwind CLI on the rendered
   HTML.
2. Replace the ``tailwindcss`` script with the pre-built CSS via the
   ``asset_urls`` override.
3. Ship the resulting standalone HTML.

This is the most reliable path for embedding reports in environments
that disallow eval-style runtime CSS (some strict CSP setups).

---

8. Print Stylesheet
-------------------

``static/print_styles.html`` contains ``@media print`` rules that:

- Hide the sidebar, info drawer, Back-to-top button, and other
  controls.
- Reset main margins for full-width page output.
- Force light backgrounds (cheaper toner and better OCR).
- Keep sections from breaking mid-figure.

To export a printable PDF, open the report in Chromium-family browsers
and use "Save as PDF" from the print dialog — the print styles take
over automatically.

.. _report-extensions:

Custom Elements and Adders
==========================

The report module is built so that custom widgets and custom section
adders compose with the shipped ones using the same patterns. This
page walks through the three common extension points.

---

1. Custom :class:`Element` Subclasses
-------------------------------------

Every primitive in :mod:`~coco_pipe.report.elements` is a subclass of
the abstract :class:`~coco_pipe.report.elements.Element`. To add your
own, override two methods at most.

1.1 Stateless element
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-------------------------------------

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
----------------------------------------------

The shipped ``add_decoding_*`` and ``add_reduction_*`` methods are
defined as top-level functions and attached to :class:`Report` at
import time. Custom adders follow the same pattern — no subclassing
needed.

3.1 Define the function
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the bottom of the module:

.. code-block:: python

   Report.add_my_summary = add_my_summary

3.3 Eager-load at package init
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The alternative — putting every adder method directly on
:class:`Report` — forces ``core.py`` to import every domain module
(decoding, dim-reduction, your custom one). That recreates the
circular-import problem the report module was specifically refactored
to avoid. The bound-function pattern keeps :mod:`coco_pipe.report.core`
ignorant of any domain.

---

4. Custom Factory Functions
---------------------------

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
-----------------------------

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
-----------------------------

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
------------------------

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
