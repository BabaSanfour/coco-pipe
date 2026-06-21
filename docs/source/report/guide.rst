.. _report-guide:

================
Building Reports
================

.. _report-concepts:

Scientific Concepts and Principles
==================================

Reports are the boundary where computation meets the reader. The
``coco_pipe.report`` module is built around a few principles that keep
that boundary trustworthy, reproducible, and offline-friendly.

---

1. Tree of Sections, Tree of Elements
-------------------------------------

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
--------------------------------------

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
--------------------

Reports load three JavaScript bundles (Plotly, Tailwind Play CDN, pako).
The :class:`Report` constructor takes ``asset_urls=`` controlling how
they're sourced:

.. list-table::
   :header-rows: 1

   * - ``asset_urls`` argument
     - Behavior
   * - ``None`` (default)
     - ``<script src="…CDN URL…">`` — requires network.
   * - ``dict``
     - Override one or more URLs (e.g. self-hosted copies).
   * - ``"inline"``
     - Download the bundles once (cached at ``~/.cache/coco-pipe/report-assets/``), then **inline them** in ``<script>`` tags. Fully offline.

See :ref:`report-assets` for the full asset story and the
:func:`~coco_pipe.report._assets.vendor_assets` helper for pre-warming
the cache on an internet-connected machine.

---

4. Provenance Is Automatic, Not Optional
----------------------------------------

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
-----------------------------------------------------

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
----------------------------------------

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
---------------------------

The module is structured so each audience uses just enough of it:

.. list-table::
   :header-rows: 1

   * - Audience
     - Surface
   * - **Run-and-save user**
     - ``from_experiment_result(...)`` or sibling factories. One call, one ``.html`` file.
   * - **Composer**
     - ``Report()`` + fluent ``add_*`` methods. Custom titles, multiple sections, custom themes.
   * - **Extender**
     - Subclass :class:`~coco_pipe.report.elements.Element`, attach functions to ``Report``, or override Jinja partials. See :ref:`report-extensions` and :ref:`report-templates`.

Most users stay at the first level. The lower levels are available
when needed and don't require the upper levels to know about them.

.. _report-core:

``Section`` and ``Report``
==========================

:class:`~coco_pipe.report.core.Section` and
:class:`~coco_pipe.report.core.Report` are the two container classes
that hold the tree of HTML elements assembled by every other module.

---

1. Quick Reference
------------------

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
--------------

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

.. list-table::

   * - ``add_element(element)``
     - Append any :class:`Element` or raw HTML string.
   * - ``add_columns(elements, cols=?)``
     - Render children in a CSS grid row.
   * - ``add_finding(check_result)``
     - Attach a :class:`CheckResult`; section status auto-upgrades to ``WARN`` / ``FAIL``.
   * - ``render() -> str``
     - Render the section to HTML (called by Report).

.. rubric:: Status semantics

``FAIL`` is sticky. Once a section's status is ``FAIL`` it cannot be
downgraded by a later ``WARN``. The sidebar marks any
``WARN`` / ``FAIL`` section with a colored dot and surfaces an
"Attention Needed" summary at the bottom of the sidebar.

---

3. ``Report``
-------------

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

.. list-table::

   * - ``_apply_theme(theme)``
     - Sets matplotlib rcParams and the Plotly ``coco`` template for any figures rendered during the session.
   * - ``_resolve_assets(asset_urls)``
     - Returns the ``(urls, mode)`` tuple that drives ``<script src=…>`` versus inlined ``<script>…</script>``. See :ref:`report-assets`.
   * - ``_resolve_config(config)``
     - Coerces ``config`` into a :class:`ReportConfig`, honoring an explicit ``title`` argument.

.. rubric:: Section adders

.. list-table::
   :header-rows: 1

   * - Method
     - Use
   * - :meth:`Report.add_section(section)`
     - Append a pre-built :class:`Section`. IDs are auto-uniqued on collision.
   * - :meth:`Report.add_figure(fig, caption=)`
     - Shortcut: wraps a matplotlib / Plotly figure in its own section.
   * - :meth:`Report.add_container(container, ...)`
     - Inspect a :class:`~coco_pipe.io.DataContainer`: dimensions, coordinates, missingness, flatline / outlier checks, sample histograms.
   * - :meth:`Report.add_raw_preview(data, name=)`
     - Interactive Plotly preview for raw arrays or :class:`DataContainer` X.
   * - :meth:`Report.add_summary_card(metrics)`
     - Top-of-report key/value stat strip (e.g., ``{"Accuracy": 0.83, "Folds": 5}``).
   * - :meth:`Report.add_markdown(text)`
     - Append a markdown block (inherited from :class:`ContainerElement`).
   * - :meth:`Report.add_element(element)`
     - Append any :class:`Element` outside a section.

Domain-specific adders (``add_decoding_*``, ``add_reduction_*``,
``add_reduction``, ``add_comparison``) are bound at import time from
:mod:`~coco_pipe.report.decoding` and :mod:`~coco_pipe.report.dim_reduction`
— see :ref:`report-section-decoding` and
:ref:`report-section-dim-reduction`.

.. rubric:: Rendering

.. list-table::

   * - ``render() -> str``
     - Collect payload + render full HTML.
   * - ``save(filename)``
     - Render and write to disk.
   * - ``show(port=None)``
     - Open the rendered report in a new browser tab via the stdlib ``http.server``.
   * - ``_repr_html_()``
     - Allows ``report`` to display inline in Jupyter notebooks.

The rendered HTML is fully self-contained: heavy data (Plotly figures,
interactive tables) is collected into one gzip+base64 payload embedded
in the page (see :ref:`report-concepts`).

---

4. Common Patterns
------------------

4.1 One-shot save
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from coco_pipe.report import Report
   from coco_pipe.report.elements import PlotlyElement

   Report("Quick").add_element(PlotlyElement(fig)).save("quick.html")

4.2 Build a multi-section report by hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   report  # _repr_html_ auto-displays the rendered HTML in Jupyter

4.4 Serve over HTTP
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   report.show()         # picks a free port, opens browser
   report.show(port=8123)

---

5. State and Identity
---------------------

- Each section is given an HTML id derived from its title (slugified).
  Collisions are auto-suffixed (``"repeated"`` → ``"repeated-2"``).
- The collected payload registry uses element-level UUIDs; nothing in
  the HTML hard-codes a section count or position.
- The TOC sidebar is built from the section tree; it updates the
  active link via an ``IntersectionObserver`` and sets
  ``aria-current="location"`` on the active link for screen readers.

---

6. Failure Modes
----------------

.. list-table::
   :header-rows: 1

   * - Action
     - Behavior
   * - Missing reducer ``get_summary``
     - :meth:`Report.add_reduction` raises ``TypeError``.
   * - Empty comparison frame
     - :meth:`Report.add_comparison` raises ``ValueError``.
   * - Broken ``DataContainer``
     - :meth:`Report.add_container` emits a ``UserWarning`` and skips the section.
   * - Section produces an error
     - The factories (``make_decoding_report`` / ``make_reduction_report``) log the failure and skip that section; the rest of the report still renders.

The factory functions are deliberately permissive: a single bad
section never blocks the rest of the report.

.. _report-api:

Factories and Assembly
======================

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
----------------------

.. list-table::
   :header-rows: 1

   * - Factory
     - Input                                            Output sections
   * - :func:`~coco_pipe.report.from_container`
     - :class:`~coco_pipe.io.DataContainer`             Data overview + optional raw preview
   * - :func:`~coco_pipe.report.from_bids`
     - BIDS root directory                              Data overview from loaded container
   * - :func:`~coco_pipe.report.from_tabular`
     - Path to CSV / parquet / etc.                     Data overview from loaded container
   * - :func:`~coco_pipe.report.from_embeddings`
     - ``X_emb`` array                                 Standalone embedding section
   * - :func:`~coco_pipe.report.from_reductions`
     - list of scored :class:`DimReduction`             One section per reducer + optional comparison
   * - :func:`~coco_pipe.report.from_experiment_result`
     - :class:`ExperimentResult`                  Full decoding report
   * - :func:`~coco_pipe.report.merge_reports`
     - ``*Report`` arguments                            Sections of every input, prefixed

Every factory accepts the standard report kwargs: ``title``,
``config``, ``theme``, ``asset_urls``, ``output_path``.

---

2. Container Factories
----------------------

2.1 ``from_container``
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

Wraps :func:`coco_pipe.io.load_data` in BIDS mode + ``from_container``.

.. code-block:: python

   from coco_pipe.report import from_bids

   report = from_bids(
       root="/data/bids_root",
       task="resting",
       output_path="bids_overview.html",
   )

2.3 ``from_tabular``
~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

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
-------------------

3.1 ``from_reductions``
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
--------------------

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
-----------------------

.. list-table::
   :header-rows: 1

   * - Argument
     - Purpose
   * - ``title``
     - Report title; shown in the header and the browser tab.
   * - ``config``
     - Dict of free-form parameters. Stored on the :class:`ReportConfig` for the Run Info drawer.
   * - ``theme``
     - ``"paper"`` (default) / ``"notebook"`` / ``"poster"``. Mirrors :mod:`coco_pipe.viz.theme` modes.
   * - ``asset_urls``
     - ``None`` / ``dict`` / ``"inline"``. See :ref:`report-assets`.
   * - ``output_path``
     - Convenience: when set, the factory also calls :meth:`Report.save(output_path)` and returns the report.
   * - ``sections``
     - ``"default"`` or list of section names. Names are factory-specific (see the section catalogs).

When in doubt, prefer the factory functions over the lower-level
``make_*_report`` functions — they accept everything ``make_*_report``
does, plus optional ``container=`` / ``raw_preview=`` for additional
inspection sections.

.. _report-configuration:

Configuration and Provenance
============================

:mod:`coco_pipe.report.config` defines two pydantic models that govern
how a report identifies itself and the environment that produced it:

- :class:`~coco_pipe.report.config.ReportConfig` — title, author,
  description, run parameters.
- :class:`~coco_pipe.report.config.ProvenanceConfig` — git hash,
  Python / OS, package versions, command line, timestamp.

Both are strict (pydantic ``extra="allow"`` for forward-compatibility),
serializable to JSON, and rendered into the report's "Run Info"
drawer.

---

1. ``ReportConfig``
-------------------

.. code-block:: python

   from coco_pipe.report.config import ReportConfig

   ReportConfig(
       title="EEG Decoding — Cohort A",
       author="Hamza Abdelhedi",
       description="3-class motor imagery, 12 subjects, LOSO CV.",
       run_params={
           "experiment_id": "exp_007",
           "max_iter": 500,
           "scaler": "StandardScaler",
       },
       # provenance defaults to ProvenanceConfig.from_env()
   )

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``title``
     - Report title. Defaults to ``"CoCo Analysis Report (YYYY-MM-DD)"``.
   * - ``author``
     - Optional author name.
   * - ``description``
     - Optional one-line summary.
   * - ``provenance``
     - :class:`ProvenanceConfig`; defaults to ``ProvenanceConfig.from_env()`` which captures git hash + package versions + python/OS at instantiation.
   * - ``run_params``
     - Free-form dict of analysis parameters, exposed in the Run Info drawer.

Pydantic ``extra="allow"`` is set, so unknown fields are preserved on
the model (accessible via attribute access).

---

2. Passing Config to a Report
-----------------------------

Three styles, ordered by ceremony:

.. code-block:: python

   # 1. Title only (most common)
   Report(title="Quick", config={"experiment": "demo"})

   # 2. Dict — pydantic-coerced to ReportConfig
   Report(config={
       "title": "Detailed",
       "description": "Quarterly QC",
       "run_params": {"experiment": "Q1", "subjects": 24},
   })

   # 3. Typed ReportConfig
   cfg = ReportConfig(title="Typed", description="QC")
   Report(config=cfg)

The :meth:`Report._resolve_config` helper handles the coercion. If
the dict is malformed (raises pydantic ``ValidationError``), the
report falls back to a minimal ``ReportConfig(title=title,
run_params=config)`` — the report still renders rather than failing
hard.

When the config dict carries its own ``"title"``, that value wins
over the ``title=`` constructor argument.

---

3. ``ProvenanceConfig``
-----------------------

Captures runtime metadata automatically:

.. list-table::
   :header-rows: 1

   * - Field
     - Source
   * - ``source``
     - User-tagged data source (e.g., ``"BIDS"``, ``"Tabular"``). Default ``"Unknown"``.
   * - ``git_hash``
     - Output of ``git rev-parse --short HEAD`` (``"Unknown"`` outside a git repo).
   * - ``timestamp_utc``
     - UTC ISO-ish timestamp at config-instantiation.
   * - ``command``
     - Original ``sys.argv`` (truncated for safety).
   * - ``python_version``
     - ``platform.python_version()``.
   * - ``os_platform``
     - ``platform.platform()``.
   * - ``coco_pipe_version``
     - ``importlib.metadata.version("coco-pipe")``.
   * - ``versions``
     - ``{package_name: version}`` for every imported scientific package detected at capture time.

3.1 Auto-capture
~~~~~~~~~~~~~~~~

.. code-block:: python

   from coco_pipe.report.config import ProvenanceConfig

   prov = ProvenanceConfig.from_env(source="BIDS")
   prov.git_hash, prov.python_version, prov.coco_pipe_version

This is what :meth:`Report.__init__` calls when no provenance is
passed. The captured snapshot is then frozen onto the report — even
if the working tree changes after rendering.

3.2 Manual override
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prov = ProvenanceConfig(
       source="cluster-shared",
       git_hash="a1b2c3d",
       command="python train.py --cohort A",
   )
   cfg = ReportConfig(title="Override", provenance=prov)
   report = Report(config=cfg)

Useful for batch runs where the auto-captured ``command`` would be
the orchestrator's command (e.g., ``snakemake``) rather than the
actual analysis invocation.

---

4. Where Config Shows Up in the Report
--------------------------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Location in rendered HTML
   * - ``title``
     - ``<title>`` and the header brand panel.
   * - ``provenance.timestamp_utc``
     - Header subheading and Run Info > Environment.
   * - ``provenance.git_hash``
     - Header summary "Git" column and Run Info.
   * - ``provenance.python_version``
     - Run Info > Environment.
   * - ``provenance.os_platform``
     - Run Info > Environment.
   * - ``provenance.coco_pipe_version``
     - Run Info > Environment and the footer.
   * - ``run_params``
     - Run Info > Configuration (rendered as syntax- highlighted JSON).
   * - ``provenance.command``
     - Run Info > Execution Command + footer.

The Run Info drawer slides in from the right when the user clicks the
"Run Info" button in the header.

---

5. Serializing and Inspecting Config
------------------------------------

Both models are standard pydantic — ``model_dump()``,
``model_dump_json()``, ``model_validate()`` all work.

.. code-block:: python

   from coco_pipe.report.config import ReportConfig

   cfg = ReportConfig(title="serializable", run_params={"k": 1})
   payload = cfg.model_dump_json(indent=2)

   restored = ReportConfig.model_validate_json(payload)

The rendered report also embeds the full config as syntax-highlighted
JSON in the Run Info drawer, so the report itself is its own
configuration audit trail.

---

6. Custom Fields
----------------

Because ``extra="allow"`` is set on both models, any extra fields on a
config dict are kept:

.. code-block:: python

   rep = Report(config={"title": "Demo", "extra_field": 42})
   rep.config.extra_field      # -> 42

Use this sparingly — extras don't get a dedicated section in the
template and just live on the pydantic instance. For analysis
parameters, prefer the typed ``run_params`` field, which gets a real
home in the Run Info drawer.
