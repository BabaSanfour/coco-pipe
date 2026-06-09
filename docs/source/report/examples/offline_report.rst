.. _report-example-offline:

==========================================
Example: Fully Offline / Air-Gapped Report
==========================================

By default a report references three JS bundles from CDN. For
air-gapped review machines, archival snapshots, or recipients on a
spotty connection, the inline-asset mode produces a single HTML file
that opens without any network access.

---

1. The One-Liner
==================

.. code-block:: python

   from coco_pipe.report import Report

   report = Report(title="Standalone Review", asset_urls="inline")
   report.add_decoding_overview(result)
   report.add_decoding_performance(result)
   report.save("standalone.html")

On the first run, the three bundles (Plotly ~3 MB, Tailwind ~70 KB,
pako ~50 KB) are downloaded into
``~/.cache/coco-pipe/report-assets/`` and inlined into the rendered
HTML. Subsequent runs reuse the cache without hitting the network.

The resulting ``standalone.html`` opens identically with WiFi off,
on an air-gapped laptop, or on an offline review machine.

---

2. Pre-Warming the Cache (For Air-Gapped Servers)
===================================================

When the analysis is generated on a machine that has no internet:

1. On any internet-connected machine, pre-warm the cache:

   .. code-block:: python

      from coco_pipe.report._assets import vendor_assets

      cache_dir = vendor_assets()                  # downloads if missing
      cache_dir = vendor_assets(force=True)        # re-download to refresh
      print(cache_dir)
      # /Users/me/.cache/coco-pipe/report-assets

2. Copy the cache directory to the offline machine.

3. Set the env var so the offline machine reads from that path:

   .. code-block:: bash

      export COCO_PIPE_REPORT_ASSET_CACHE=/cluster/shared/coco-pipe-assets

4. Use ``asset_urls="inline"`` as normal. No network call is made.

---

3. Combining with Factories
=============================

Every factory accepts the same ``asset_urls`` argument:

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   report = from_experiment_result(
       result,
       feature_metadata=feature_meta,
       info=raw.info,
       title="Cohort A — Frozen Snapshot",
       asset_urls="inline",
       output_path="reports/cohort_a_offline.html",
   )

Useful when shipping reports to clinical reviewers, conference
organizers, or anywhere the recipient cannot run JS from a CDN.

---

4. Hybrid: Self-Hosted URLs (No Download Needed)
==================================================

If your team already serves vendored copies of the JS files (e.g.
behind a corporate intranet), skip the inline mode and just point at
your own URLs:

.. code-block:: python

   report = Report(
       title="Intranet-Served",
       asset_urls={
           "plotly": "/static/js/plotly-2.27.0.min.js",
           "tailwind": "/static/js/tailwind.min.js",
           "pako": "/static/js/pako-2.1.0.min.js",
       },
   )

The HTML stays small (no inlined JS) but only loads from your
intranet rather than CDN.

---

5. File-Size Comparison
=========================

Approximate sizes for a typical decoding report (one experiment, ~10
sections, ~6 Plotly figures):

==========================  =================  ============================
Asset mode                  HTML size          Opens offline?
==========================  =================  ============================
``None`` (CDN)              ~ 200 KB           No
``dict`` (self-hosted)      ~ 200 KB           Only if URLs reachable
``"inline"``                ~ 3.2 MB           Yes
==========================  =================  ============================

The ~3 MB delta is dominated by the Plotly bundle (the others are
each well under 100 KB). Worth it for offline use; skip for casual
local-only sharing.

---

6. Verifying the Inline Output
================================

A quick sanity check:

.. code-block:: bash

   $ grep -c '<script src=' standalone.html
   0
   $ grep -c 'pako.inflate' standalone.html
   1                                # inlined; was loaded from CDN before

If you see ``<script src=`` entries pointing at ``cdn.plot.ly`` /
``cdnjs`` / ``cdn.tailwindcss.com``, the inline mode didn't activate.
Common causes:

================================  =====================================================
Problem                           Fix
================================  =====================================================
``OSError`` during render          Network blocked on first download. Pre-warm the cache
                                  on an online machine and ship the cache dir.
``TypeError`` at construction      ``asset_urls`` value isn't ``None`` / ``dict`` /
                                  ``"inline"``. The error message lists the supported
                                  forms.
Cache directory not writable       Override with ``COCO_PIPE_REPORT_ASSET_CACHE`` (point
                                  at a writable path).
================================  =====================================================

---

7. Archival Pattern
=====================

For long-term archives — e.g. one report per publication, kept for
the lifetime of a paper — combine inline assets with a snapshot of
the full provenance:

.. code-block:: python

   import datetime as dt
   import json

   from coco_pipe.report import from_experiment_result

   archive_path = f"archive/{dt.date.today().isoformat()}_decoding.html"

   report = from_experiment_result(
       result,
       title="Paper Figure 3 — Frozen Reference",
       asset_urls="inline",
       config={
           "doi": "10.1234/abcd.5678",
           "input_data_sha256": data_sha,
           "code_commit": git_hash,
       },
       output_path=archive_path,
   )

   # Optional: also export the provenance separately for indexing
   archive_meta = report.config.provenance.model_dump()
   with open(archive_path.replace(".html", ".meta.json"), "w") as f:
       json.dump(archive_meta, f, indent=2)

The HTML is the archival artifact; the ``.meta.json`` sidecar makes
it searchable in a data catalog without parsing HTML.
