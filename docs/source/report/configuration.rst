.. _report-configuration:

==========================================
Configuration and Provenance
==========================================

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
=====================

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

============================  ===================================================
Field                         Description
============================  ===================================================
``title``                     Report title. Defaults to ``"CoCo Analysis Report
                              (YYYY-MM-DD)"``.
``author``                    Optional author name.
``description``               Optional one-line summary.
``provenance``                :class:`ProvenanceConfig`; defaults to
                              ``ProvenanceConfig.from_env()`` which captures git
                              hash + package versions + python/OS at instantiation.
``run_params``                Free-form dict of analysis parameters, exposed in
                              the Run Info drawer.
============================  ===================================================

Pydantic ``extra="allow"`` is set, so unknown fields are preserved on
the model (accessible via attribute access).

---

2. Passing Config to a Report
===============================

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
=========================

Captures runtime metadata automatically:

============================  ===================================================
Field                         Source
============================  ===================================================
``source``                    User-tagged data source (e.g., ``"BIDS"``,
                              ``"Tabular"``). Default ``"Unknown"``.
``git_hash``                  Output of ``git rev-parse --short HEAD``
                              (``"Unknown"`` outside a git repo).
``timestamp_utc``             UTC ISO-ish timestamp at config-instantiation.
``command``                   Original ``sys.argv`` (truncated for safety).
``python_version``            ``platform.python_version()``.
``os_platform``               ``platform.platform()``.
``coco_pipe_version``         ``importlib.metadata.version("coco-pipe")``.
``versions``                  ``{package_name: version}`` for every imported
                              scientific package detected at capture time.
============================  ===================================================

3.1 Auto-capture
------------------

.. code-block:: python

   from coco_pipe.report.config import ProvenanceConfig

   prov = ProvenanceConfig.from_env(source="BIDS")
   prov.git_hash, prov.python_version, prov.coco_pipe_version

This is what :meth:`Report.__init__` calls when no provenance is
passed. The captured snapshot is then frozen onto the report — even
if the working tree changes after rendering.

3.2 Manual override
---------------------

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
========================================

================================  =====================================================
Field                             Location in rendered HTML
================================  =====================================================
``title``                         ``<title>`` and the header brand panel.
``provenance.timestamp_utc``      Header subheading and Run Info > Environment.
``provenance.git_hash``           Header summary "Git" column and Run Info.
``provenance.python_version``     Run Info > Environment.
``provenance.os_platform``        Run Info > Environment.
``provenance.coco_pipe_version``  Run Info > Environment and the footer.
``run_params``                    Run Info > Configuration (rendered as syntax-
                                  highlighted JSON).
``provenance.command``            Run Info > Execution Command + footer.
================================  =====================================================

The Run Info drawer slides in from the right when the user clicks the
"Run Info" button in the header.

---

5. Serializing and Inspecting Config
======================================

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
==================

Because ``extra="allow"`` is set on both models, any extra fields on a
config dict are kept:

.. code-block:: python

   rep = Report(config={"title": "Demo", "extra_field": 42})
   rep.config.extra_field      # -> 42

Use this sparingly — extras don't get a dedicated section in the
template and just live on the pydantic instance. For analysis
parameters, prefer the typed ``run_params`` field, which gets a real
home in the Run Info drawer.
