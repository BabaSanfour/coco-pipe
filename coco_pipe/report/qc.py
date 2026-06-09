"""QC section builder for HTML reports.

Produces the standardised "Data Quality (QC)" :class:`~coco_pipe.report.core.Section`
from a :class:`~coco_pipe.io.quality.QCResult`.  Consumed by the decoding and
dimensionality-reduction report builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from coco_pipe.io.quality import QCResult

from .elements import ImageElement, TableElement

if TYPE_CHECKING:
    from .core import Section


def build_qc_section(qc_result: QCResult) -> "Section":
    """Build the standardised QC drop-log report section.

    Parameters
    ----------
    qc_result:
        Populated result from :func:`~coco_pipe.io.quality.run_qc`.

    Returns
    -------
    Section
        A fully populated report section ready to be added to a
        :class:`~coco_pipe.report.core.Report`.
    """
    from .core import Section

    section = Section("Data Quality (QC)")
    summary = pd.DataFrame(
        [
            {
                "Metric": key.replace("_", " ").title(),
                "Value": value,
            }
            for key, value in qc_result.summary().items()
        ]
    )
    section.add_element(TableElement(summary, title="QC Funnel Summary"))

    if qc_result.epochs_dropped:
        epoch_rows = pd.DataFrame(
            [
                {
                    "Observation ID": record.obs_id,
                    "Original Index": record.obs_index,
                    "Outlier Fraction": record.outlier_fraction,
                    "Maximum MAD Z": record.mad_z_max,
                }
                for record in qc_result.epochs_dropped
            ]
        )
        section.add_element(
            TableElement(
                epoch_rows,
                title=f"Dropped Epochs ({qc_result.n_epochs_dropped})",
            )
        )

    if qc_result.subjects_dropped:
        subject_rows = pd.DataFrame(
            [
                {
                    "Subject ID": record.subject_id,
                    "Mean Per-Epoch Outlier Fraction": record.outlier_fraction,
                    "Mean N Outlier Features": record.n_outlier_features,
                }
                for record in qc_result.subjects_dropped
            ]
        )
        section.add_element(
            TableElement(
                subject_rows,
                title=f"Dropped Subjects ({qc_result.n_subjects_dropped})",
            )
        )

    burden = qc_result.subject_outlier_burden
    if burden is not None and not burden.empty:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 3))
        axis.hist(
            burden["outlier_fraction"],
            bins=min(20, max(1, len(burden))),
            color="steelblue",
            alpha=0.8,
        )
        if qc_result.subject_outlier_fraction_threshold is not None:
            axis.axvline(
                qc_result.subject_outlier_fraction_threshold,
                color="red",
                linestyle="--",
                label="Drop threshold",
            )
            axis.legend()
        axis.set_xlabel("Mean Per-Epoch Outlier Fraction")
        axis.set_ylabel("N Subjects")
        axis.set_title("Subject Outlier Burden Distribution")
        figure.tight_layout()
        section.add_element(
            ImageElement(
                figure,
                caption="Mean per-epoch outlier burden for each subject.",
            )
        )
        plt.close(figure)

    family_qc = qc_result.family_qc
    if family_qc is not None and not family_qc.empty:
        section.add_element(
            TableElement(family_qc, title="Family-Level Quality Summary")
        )

    missingness = qc_result.feature_missingness
    if missingness is not None:
        section.add_element(TableElement(missingness, title="Feature Missingness"))
    return section
