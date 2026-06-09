.. _report-templates:

==================================
Templates and Layout Customization
==================================

The report HTML is produced by Jinja templates living under
``coco_pipe/report/templates/``. The default layout is opinionated but
intentionally easy to extend or replace — you can override individual
partials, swap the entire base template, or build a fresh layout from
the published Element catalog.

---

1. Template Layout
====================

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
=======================

Every render of ``base.html`` is called with this context (see
:meth:`Report.render <coco_pipe.report.core.Report.render>`):

================================  ==========================================================
Variable                          Type / source
================================  ==========================================================
``title``                         ``Report.title``
``content``                       Pre-rendered HTML from the section tree (``| safe``)
``timestamp``                     ``Report.timestamp``
``toc``                           List of ``{id, title, icon, status}`` per section
``metadata``                      ``Report.config.provenance.model_dump()``
``config``                        ``Report.config.model_dump_json(indent=2)``
``payload``                       gzip+base64 string of the global data registry
``report_summary``                Dict of ``sections``/``warnings``/``failures``/
                                  ``plot_count``/``table_count``/etc.
``asset_urls``                    Dict; in inline mode, holds JS *contents*
``asset_mode``                    ``"cdn"`` / ``"custom"`` / ``"inline"``
================================  ==========================================================

``section.html`` (called once per :class:`Section`) receives:

================================  ==========================================================
Variable                          Type / source
================================  ==========================================================
``title``, ``id``, ``icon``,      Section attributes
``status``, ``description``,
``metadata``, ``tags``, ``code``,
``findings``
``content``                       Pre-rendered HTML from section children (``| safe``)
================================  ==========================================================

---

3. Overriding a Single Partial
================================

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
=====================================

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
==================================

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
=============================

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
=========================

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
=====================

``static/print_styles.html`` contains ``@media print`` rules that:

- Hide the sidebar, info drawer, Back-to-top button, and other
  controls.
- Reset main margins for full-width page output.
- Force light backgrounds (cheaper toner and better OCR).
- Keep sections from breaking mid-figure.

To export a printable PDF, open the report in Chromium-family browsers
and use "Save as PDF" from the print dialog — the print styles take
over automatically.
