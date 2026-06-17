import pandas as pd

from coco_pipe.io.quality import EpochDropRecord, QCResult, SubjectDropRecord
from coco_pipe.report.core import Section
from coco_pipe.report.elements import (
    ImageElement,
    InteractiveTableElement,
    TableElement,
)
from coco_pipe.report.qc import build_qc_section


def test_build_qc_section_empty():
    qc_res = QCResult()
    section = build_qc_section(qc_res)
    assert isinstance(section, Section)
    assert section.title == "Data Quality (QC)"
    # At least the summary table
    assert len(section.children) == 1
    assert isinstance(section.children[0], TableElement)


def test_build_qc_section_full():
    qc_res = QCResult(
        n_obs_in=100,
        n_obs_out=90,
        epochs_dropped=[
            EpochDropRecord(
                obs_id="obs1", obs_index=1, outlier_fraction=0.5, mad_z_max=10.0
            ),
            EpochDropRecord(
                obs_id="obs2", obs_index=2, outlier_fraction=0.6, mad_z_max=12.0
            ),
        ],
        subjects_dropped=[
            SubjectDropRecord(
                subject_id="sub1", outlier_fraction=0.8, n_outlier_features=5
            )
        ],
        subject_outlier_burden=pd.DataFrame({"outlier_fraction": [0.1, 0.2, 0.9]}),
        subject_outlier_fraction_threshold=0.5,
        family_qc=pd.DataFrame({"family": ["fam1"], "status": ["pass"]}),
        feature_missingness=pd.DataFrame({"feature": ["feat1"], "missing": [0.0]}),
        per_family_dropped={
            "band": [
                SubjectDropRecord(
                    subject_id="sub1", outlier_fraction=0.8, n_outlier_features=5
                )
            ]
        },
        feature_columns_dropped=pd.DataFrame(
            {"column": ["band_bad_ch-Fz"], "drop_reason": ["all_nan"]}
        ),
    )
    section = build_qc_section(qc_res)
    assert len(section.children) == 8
    types = [type(e) for e in section.children]
    assert types == [
        TableElement,  # Summary
        TableElement,  # Epochs Dropped
        TableElement,  # Subjects Dropped
        TableElement,  # Per-group retention
        TableElement,  # Family-scoped drops
        ImageElement,  # Outlier Burden Image
        TableElement,  # Family QC
        TableElement,  # Pruned columns
    ]

    # Verify titles
    titles = [getattr(e, "title", None) for e in section.children]
    assert titles[0] == "QC Funnel Summary"
    assert titles[1] == "Dropped Epochs (2)"
    assert titles[2] == "Dropped Subjects (1)"
    assert titles[3] == "Per-Group Retention (group_by=family)"
    assert titles[4] == "Conditional Drops by family"
    assert titles[6] == "Family-Level Quality Summary"
    assert titles[7] == "Pruned Descriptor Columns"


def test_build_qc_section_compacts_long_tables_and_filters_zero_missingness():
    qc_res = QCResult(
        n_obs_in=20,
        n_obs_out=8,
        epochs_dropped=[
            EpochDropRecord(
                obs_id=f"obs{index}",
                obs_index=index,
                outlier_fraction=0.5,
                mad_z_max=10.0,
            )
            for index in range(12)
        ],
        feature_missingness=pd.DataFrame(
            {
                "feature": ["complete", "incomplete"],
                "missing": [0.0, 0.25],
            }
        ),
    )

    section = build_qc_section(qc_res, compact=True, page_size=10)

    assert isinstance(section.children[1], InteractiveTableElement)
    missingness = next(
        element
        for element in section.children
        if getattr(element, "title", None) == "Feature Missingness"
    )
    assert TableElement._to_frame(missingness.data)["feature"].tolist() == [
        "incomplete"
    ]
