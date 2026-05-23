.. _report-concepts:

==================================
Scientific Concepts and Principles
==================================

Reports are the boundary where computation meets the reader. The
``coco_pipe.report`` module is built around a few principles that keep
that boundary trustworthy, reproducible, and offline-friendly.

---

1. Tree of Sections, Tree of Elements
=======================================

A :class:`~coco_pipe.report.core.Report` is a container of
:class:`~coco_pipe.report.core.Section` objects. Each section is itself
a container of :class:`~coco_pipe.report.elements.Element` instances —
images, Plotly figures, tables, code blocks, callouts, accordions, and
so on (see :ref:`report-elements`).

.. code-block:: text

   Report
   ├── Section ("Overview")
   │   ├── TableElement
   │   └── CalloutElement
   ├── Section ("Performance")
   │   ├── PlotlyElement
   │   ├── TableElement
   │   └── ImageElement
   └── Section ("Provenance")
       └── CodeBlockElement

This tree is rendered in two passes:

1. **Collect** — every element pushes payload data into a global
   registry keyed by UUID.
2. **Render** — Jinja templates emit HTML; elements that own payload
   data emit a placeholder ``<div data-id="…" class="lazy-plot"></div>``;
   the browser hydrates them on demand.

---

2. Self-Contained, Lazy-Hydrating HTML
========================================

A rendered report is a single ``.html`` file. Heavy data (Plotly
figures, interactive tables) is not duplicated per element — it goes
through one registry:

.. code-block:: text

   Report.render():
       registry = {}                          # one dict, all elements
       self.collect_payload(registry)         # walk tree, populate

       payload = json.dumps(registry).encode("utf-8")
       payload = gzip.compress(payload)
       payload = base64.b64encode(payload).decode("utf-8")

       <html>...
         <script type="application/json" id="report-payload">{payload}</script>
         <script>
           const REPORT_DATA = JSON.parse(pako.inflate(atob(...)));
           // lazy-plot div hydration on intersection
         </script>
       </html>

This contract lets us:

- compress all data once (typically 60-90 % size reduction),
- render only what scrolls into view (no upfront Plotly cost),
- keep the file self-contained: no sibling assets to ship.

The :meth:`Report.render <coco_pipe.report.core.Report.render>`
docstring spells out the contract; the matching browser-side decoder
lives in ``templates/static/report_scripts.html``.

---

3. Three Asset Modes
======================

Reports load three JavaScript bundles (Plotly, Tailwind Play CDN, pako).
The :class:`Report` constructor takes ``asset_urls=`` controlling how
they're sourced:

==========================  =========================================================
``asset_urls`` argument     Behavior
==========================  =========================================================
``None`` (default)          ``<script src="…CDN URL…">`` — requires network.
``dict``                    Override one or more URLs (e.g. self-hosted copies).
``"inline"``                Download the bundles once (cached at
                            ``~/.cache/coco-pipe/report-assets/``), then **inline
                            them** in ``<script>`` tags. Fully offline.
==========================  =========================================================

See :ref:`report-assets` for the full asset story and the
:func:`~coco_pipe.report._assets.vendor_assets` helper for pre-warming
the cache on an internet-connected machine.

---

4. Provenance Is Automatic, Not Optional
==========================================

Every :class:`Report` captures a
:class:`~coco_pipe.report.config.ProvenanceConfig` at construction
time:

- ``git_hash`` — current commit (or ``"Unknown"`` outside a git repo).
- ``python_version``, ``os_platform`` — runtime metadata.
- ``coco_pipe_version`` — installed package version.
- ``versions`` — versions of every imported scientific package.
- ``timestamp_utc`` — when the report was assembled.
- ``command`` — the command-line invocation that produced it.

Provenance lives under the **Run Info** drawer in the rendered HTML and
is also serialized into the on-page payload. To audit a report after
the fact, callers don't need the original code — the report carries
enough metadata to reproduce the environment.

You can override or extend any field via the ``config`` argument to
:class:`Report`; see :ref:`report-configuration`.

---

5. Section Adders Are Functions, Bound at Import Time
=======================================================

Each ``add_decoding_*`` / ``add_reduction_*`` method on
:class:`Report` is a top-level function in
:mod:`coco_pipe.report.decoding` or
:mod:`coco_pipe.report.dim_reduction`, attached to ``Report`` at
package import:

.. code-block:: python

   # In coco_pipe/report/decoding.py:
   def add_decoding_overview(self: Report, result, *, name="Overview"):
       ...
   Report.add_decoding_overview = add_decoding_overview

This pattern keeps ``core.py`` free of domain-specific imports (no
circular dependency on ``viz`` or ``decoding``) while preserving the
fluent ``report.add_decoding_overview(...)`` API. The standalone
functions remain importable for testing.

See :ref:`report-extensions` for the same pattern applied to your own
custom section adders.

---

6. Data-Quality Findings Are First-Class
==========================================

When a :class:`~coco_pipe.io.DataContainer` is added via
:meth:`Report.add_container <coco_pipe.report.core.Report.add_container>`,
the section is automatically populated with quality findings:

- Missingness (NaN fraction)
- Flatline detection (zero-variance signals)
- Outliers (z-score thresholding)
- Constant columns

Each finding is a :class:`~coco_pipe.report.data_quality.CheckResult`
with a severity and status (``"OK"`` / ``"WARN"`` / ``"FAIL"``). The
section's overall status reflects the worst finding (FAIL > WARN > OK).
The sidebar TOC then surfaces sections needing attention with colored
dots.

See :ref:`report-data-quality`.

---

7. Three Audiences, One API
=============================

The module is structured so each audience uses just enough of it:

================================  ======================================================
Audience                          Surface
================================  ======================================================
**Run-and-save user**             ``from_experiment_result(...)`` or sibling factories.
                                  One call, one ``.html`` file.
**Composer**                      ``Report()`` + fluent ``add_*`` methods. Custom titles,
                                  multiple sections, custom themes.
**Extender**                      Subclass :class:`~coco_pipe.report.elements.Element`,
                                  attach functions to ``Report``, or override Jinja
                                  partials. See :ref:`report-extensions` and
                                  :ref:`report-templates`.
================================  ======================================================

Most users stay at the first level. The lower levels are available
when needed and don't require the upper levels to know about them.
