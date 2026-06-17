.. _report-data-quality:

==================================
Data-Quality Checks and Findings
==================================

:mod:`coco_pipe.io.quality` contains the dataclass-based
quality checks that run automatically when a
:class:`~coco_pipe.io.DataContainer` is passed to
:meth:`Report.add_container <coco_pipe.report.core.Report.add_container>`,
plus the :class:`~coco_pipe.io.quality.CheckResult` type that
:meth:`Section.add_finding <coco_pipe.report.core.Section.add_finding>`
consumes.

---

1. ``CheckResult``
====================

Every check returns a :class:`CheckResult` (or a list of them):

.. code-block:: python

   @dataclass
   class CheckResult:
       check_name: str           # e.g. "Missingness"
       status: Literal["OK", "WARN", "FAIL"]
       message: str              # human-readable summary
       severity: int             # 0 (info) to 10 (critical)
       metric_name: str | None = None
       metric_value: float | None = None

       @property
       def is_issue(self) -> bool:
           return self.status in {"WARN", "FAIL"}

When a section accumulates findings, its overall status is the worst
of any single finding's status. ``FAIL`` is sticky: once a section
reaches it, a later ``WARN`` does not downgrade.

---

2. Built-in Checks
====================

All checks accept either a ``pd.DataFrame`` or a NumPy array and
operate on numeric columns only.

==================================  ============================================================
Check                                What it does
==================================  ============================================================
:func:`~coco_pipe.io.quality.check_missingness`     Fraction of NaN values.
                                    ``WARN`` ≥ ``threshold_warn`` (default 0.01),
                                    ``FAIL`` ≥ ``threshold_fail`` (default 0.20).
:func:`~coco_pipe.io.quality.check_constant_columns`  Per-column near-zero variance.
                                    Returns one ``CheckResult`` per offending column;
                                    ``FAIL`` for any constant numeric column.
:func:`~coco_pipe.io.quality.check_outliers_zscore`  Z-score outlier detection.
                                    ``WARN`` when outlier fraction exceeds threshold.
:func:`~coco_pipe.io.quality.check_flatline`        Detect zero-variance signal
                                    arrays (e.g., flatlined EEG channels).
==================================  ============================================================

.. code-block:: python

   from coco_pipe.io.quality import (
       check_missingness, check_constant_columns,
       check_flatline, check_outliers_zscore,
   )

   miss = check_missingness(df)
   constants = check_constant_columns(df)
   outliers = check_outliers_zscore(df, threshold=3.0)
   flat = check_flatline(signal_array, threshold=1e-10)

   for result in [miss, *constants, outliers, flat]:
       print(result.status, result.check_name, result.message)

---

3. Automatic Integration via ``add_container``
================================================

The most common path: pass a :class:`DataContainer` to a report and
let the checks fire automatically.

.. code-block:: python

   from coco_pipe.report import Report
   from coco_pipe.io import load_data

   container = load_data("scores.csv", mode="tabular", target_col="label")

   report = Report(title="Input QC")
   report.add_container(container)
   report.save("qc.html")

The added section will contain:

- a metadata table (dims, coords, dtype),
- a ``CalloutElement`` per data-quality finding,
- a histogram or scatter preview of the values,
- automatic section-status upgrades to ``WARN`` / ``FAIL`` when
  findings warrant it.

If the container itself can't be inspected (e.g., wrong shape), the
section is skipped with a ``UserWarning`` rather than raising.

---

4. Adding Custom Findings to Any Section
==========================================

A finding doesn't have to come from a built-in check — you can attach
your own at any point.

.. code-block:: python

   from coco_pipe.report import Section
   from coco_pipe.io.quality import CheckResult

   sec = Section(title="Manual QC")
   sec.add_finding(CheckResult(
       check_name="Subject coverage",
       status="WARN",
       message="Only 14 of 20 subjects have all 3 conditions.",
       severity=5,
       metric_name="subjects_with_full_coverage",
       metric_value=14,
   ))

The section is rendered with a colored finding banner per attached
result, the section header shows the status pill, and the sidebar TOC
flags the section with a colored dot.

---

5. Custom Checks
==================

A custom check is any function returning a :class:`CheckResult` (or a
list of them):

.. code-block:: python

   def check_class_balance(y, *, threshold_warn=0.1):
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

Then plug it into your own section adder or call it directly:

.. code-block:: python

   sec.add_finding(check_class_balance(y))

---

6. Severity Conventions
=========================

The numeric ``severity`` field is informational (not used to compute
the displayed status). Useful ranges:

====================  ====================================================
``severity``          Meaning
====================  ====================================================
0                     Informational; no action.
1-3                   Minor; worth noting in the report.
4-6                   Warning; investigate before publishing.
7-9                   Major; the analysis may be invalid.
10                    Critical; downstream results should not be trusted.
====================  ====================================================

Pair high severities with ``status="FAIL"`` for the visual loudness
to match.
