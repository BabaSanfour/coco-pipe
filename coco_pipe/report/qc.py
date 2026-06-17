"""QC section builder for HTML reports.

Produces the standardised "Data Quality (QC)" :class:`~coco_pipe.report.core.Section`
from a :class:`~coco_pipe.io.quality.QCResult`.  Consumed by the decoding and
dimensionality-reduction report builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from coco_pipe.io.quality import QCResult

from ._utils import _ensure_static_matplotlib_backend
from .elements import ImageElement, InteractiveTableElement, TableElement

if TYPE_CHECKING:
    from .core import Section


def _add_qc_table(
    section: "Section",
    frame: pd.DataFrame,
    *,
    title: str,
    compact: bool,
    page_size: int,
) -> None:
    """Add a static or paginated QC table according to report density."""
    if compact and title != "QC Funnel Summary" and len(frame) > page_size:
        selectors = [
            str(column)
            for column in frame.columns
            if str(column).startswith("Group (")
            or str(column).lower()
            in {"family", "subfamily", "measure", "channel", "reason"}
        ]
        section.add_element(
            InteractiveTableElement(
                frame,
                title=title,
                selector_columns=selectors,
                page_size=page_size,
            )
        )
        return
    section.add_element(TableElement(frame, title=title))


def build_qc_section(
    qc_result: QCResult,
    *,
    compact: bool = False,
    page_size: int = 10,
) -> "Section":
    """Build the standardised QC drop-log report section.

    Parameters
    ----------
    qc_result:
        Populated result from :func:`~coco_pipe.io.quality.run_qc`.
    compact:
        Paginate long detail tables when ``True``.
    page_size:
        Number of rows shown per page in compact tables.

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
    _add_qc_table(
        section,
        summary,
        title="QC Funnel Summary",
        compact=compact,
        page_size=page_size,
    )

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
        _add_qc_table(
            section,
            epoch_rows,
            title=f"Dropped Epochs ({qc_result.n_epochs_dropped})",
            compact=compact,
            page_size=page_size,
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
        _add_qc_table(
            section,
            subject_rows,
            title=f"Dropped Subjects ({qc_result.n_subjects_dropped})",
            compact=compact,
            page_size=page_size,
        )

    if qc_result.per_family_dropped:
        group_by = (qc_result.thresholds or {}).get("group_by", "family")
        n_in = qc_result.n_obs_in
        group_column = f"Group ({group_by})"
        retention_rows = []
        drop_rows = []
        for group, records in qc_result.per_family_dropped.items():
            dropped_ids = {
                getattr(record, "subject_id", getattr(record, "obs_id", ""))
                for record in records
            }
            retention_rows.append(
                {
                    group_column: group,
                    "N In": n_in,
                    "N Dropped": len(dropped_ids),
                    "N Kept": (n_in - len(dropped_ids)) if n_in else None,
                }
            )
            for record in records:
                drop_rows.append(
                    {
                        group_column: group,
                        "Dropped ID": getattr(
                            record,
                            "subject_id",
                            getattr(record, "obs_id", ""),
                        ),
                        "Outlier Fraction": record.outlier_fraction,
                    }
                )
        _add_qc_table(
            section,
            pd.DataFrame(retention_rows),
            title=f"Per-Group Retention (group_by={group_by})",
            compact=compact,
            page_size=page_size,
        )
        if drop_rows:
            _add_qc_table(
                section,
                pd.DataFrame(drop_rows),
                title=f"Conditional Drops by {group_by}",
                compact=compact,
                page_size=page_size,
            )

    burden = qc_result.subject_outlier_burden
    if burden is not None and not burden.empty:
        _ensure_static_matplotlib_backend()
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
        _add_qc_table(
            section,
            family_qc,
            title="Family-Level Quality Summary",
            compact=compact,
            page_size=page_size,
        )

    missingness = qc_result.feature_missingness
    if missingness is not None and not missingness.empty:
        missingness = missingness.copy()
        value_columns = [
            column
            for column in missingness.columns
            if str(column).lower() == "missing"
            or str(column).lower().startswith("missing_")
        ]
        if value_columns:
            values = missingness[value_columns].apply(pd.to_numeric, errors="coerce")
            missingness = missingness.loc[values.fillna(0).gt(0).any(axis=1)].copy()
        if not missingness.empty:
            _add_qc_table(
                section,
                missingness,
                title="Feature Missingness",
                compact=compact,
                page_size=page_size,
            )
    dropped_columns = qc_result.feature_columns_dropped
    if dropped_columns is not None and not dropped_columns.empty:
        _add_qc_table(
            section,
            dropped_columns,
            title="Pruned Descriptor Columns",
            compact=compact,
            page_size=page_size,
        )
    return section
