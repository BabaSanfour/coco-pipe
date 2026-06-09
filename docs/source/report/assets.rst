.. _report-assets:

============================
JavaScript Asset Modes
============================

Rendered reports rely on three JavaScript bundles:

- **Plotly** — interactive plot rendering.
- **Tailwind Play CDN** — runtime CSS compiler driven by the markup.
- **pako** — gzip decompression of the embedded data payload.

How those bundles are served is controlled by the ``asset_urls``
constructor argument on :class:`~coco_pipe.report.core.Report` and
every factory in :mod:`coco_pipe.report.api`.

---

1. Three Modes
================

==========================  =========================================================
``asset_urls`` argument     Behavior
==========================  =========================================================
``None`` (default)          ``<script src="…cdn URL…">`` tags. Requires network.
``dict``                    CDN defaults merged with the user's overrides.
``"inline"``                The bundles are downloaded once (with caching) and
                            embedded directly in ``<script>...</script>`` tags.
                            Resulting HTML opens fully offline.
==========================  =========================================================

The current mode is exposed on the report as ``Report.asset_mode``
(``"cdn"`` / ``"custom"`` / ``"inline"``) and surfaced in the Run
Info drawer.

---

2. CDN Mode (Default)
=======================

.. code-block:: python

   from coco_pipe.report import Report

   report = Report(title="Online")
   report.save("online.html")

The rendered HTML loads:

- ``https://cdn.plot.ly/plotly-2.27.0.min.js``
- ``https://cdn.tailwindcss.com``
- ``https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js``

Pros: nothing to host, small HTML.
Cons: requires network when the report is opened.

---

3. Self-Hosted URLs
=====================

Pass a dict to point at your own bundle URLs:

.. code-block:: python

   report = Report(
       title="Self-hosted",
       asset_urls={
           "plotly": "/static/plotly-2.27.0.min.js",
           "tailwind": "/static/tailwind.min.js",
           "pako": "/static/pako-2.1.0.min.js",
       },
   )

Only override the URLs you want to change; unspecified slots fall back
to the CDN defaults. The mode is set to ``"custom"``.

Useful when your team intranet hosts vendored JS, or when you need a
specific Plotly version different from the default.

---

4. Inline Mode (Fully Offline)
================================

.. code-block:: python

   report = Report(title="Air-gapped", asset_urls="inline")
   report.save("standalone.html")

On the first run, :func:`~coco_pipe.report._assets.get_vendored_contents`
downloads the three bundles into
``~/.cache/coco-pipe/report-assets/`` and reads them as strings. The
template then inlines them in ``<script>...</script>`` tags instead of
``<script src=...>``. Subsequent runs skip the download and just read
from cache.

The resulting HTML is fully self-contained — opens identically on a
laptop with no internet, an air-gapped review machine, or a printed-
PDF reference copy.

4.1 Cache location
--------------------

The cache directory defaults to
``~/.cache/coco-pipe/report-assets/``. Override it via environment
variable when generating reports on a shared cluster:

.. code-block:: bash

   export COCO_PIPE_REPORT_ASSET_CACHE=/scratch/coco-pipe-assets

4.2 Pre-warming the cache
---------------------------

For automated pipelines running on a machine without internet, run
the download step once on a machine that does:

.. code-block:: python

   from coco_pipe.report._assets import vendor_assets

   cache_dir = vendor_assets()                        # downloads if missing
   cache_dir = vendor_assets(force=True)              # re-download

Then ship the cache directory alongside the code base, set the env
var to its path on the target machine, and ``asset_urls="inline"``
will use the local copies without ever touching the network.

4.3 Cost
----------

Inline mode adds roughly:

==================  =========================
Bundle              Approximate size
==================  =========================
Plotly              ~3 MB minified
Tailwind Play CDN   ~70 KB
pako                ~50 KB
==================  =========================

So a typical inline report grows by ~3-4 MB compared to a CDN report.
Worth it when offline access matters; skip when it doesn't.

---

5. Failure Modes
==================

==================================  =====================================================
Scenario                            Behavior
==================================  =====================================================
Network unavailable in CDN mode     Report renders fine; opens blank or broken in the
                                    browser. Switch to ``"inline"`` to fix.
Network unavailable in inline mode  First call raises :class:`OSError` (urllib). Pre-warm
                                    the cache via ``vendor_assets()`` on an online machine.
Unknown ``asset_urls`` value        :meth:`Report._resolve_assets` raises ``TypeError``
                                    at construction.
Cache directory not writable        First call raises ``OSError``. Override location with
                                    ``COCO_PIPE_REPORT_ASSET_CACHE``.
==================================  =====================================================

---

6. Asset Mode at a Glance — When to Use Which
===============================================

================================  ========================================================
Use case                          Recommended mode
================================  ========================================================
Interactive notebook exploration  ``None`` (CDN) — smallest HTML, network always available.
Web-served reports                ``dict`` pointing at your CDN paths — control over
                                  the asset bundle versions.
Email / Slack-shared HTML         ``"inline"`` — recipients can open the file with no
                                  network access.
Long-term archive / publication   ``"inline"`` — frozen, self-contained record.
Air-gapped / cluster runs         ``"inline"`` + pre-warmed cache + env override.
================================  ========================================================
