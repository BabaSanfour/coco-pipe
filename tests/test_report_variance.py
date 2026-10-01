import pandas as pd

from coco_pipe.report import (
    build_alignment_coverage_section,
    build_alignment_tradeoff_section,
    build_subject_alignment_diagnostics_section,
    select_subject_alignment_diagnostics,
    validate_subject_alignment_diagnostics,
)
from coco_pipe.report.elements import (
    CalloutElement,
    ColumnsElement,
    InteractiveTableElement,
    PlotlyElement,
)
from coco_pipe.report.variance import _add_raw_delta


def test_subject_alignment_diagnostics_section_includes_probe_and_table():
    diagnostics = pd.DataFrame(
        [
            {
                "transform": "none",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "metric": "subject_probe_linear_balanced_accuracy",
                "value": 0.91,
                "n_subjects": 4,
                "n_observations": 48,
                "design": "nested_subject_within_label",
            },
            {
                "transform": "none",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "metric": "subject_probe_chance",
                "value": 0.25,
                "n_subjects": 4,
                "n_observations": 48,
                "design": "nested_subject_within_label",
            },
            {
                "transform": "leace",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "metric": "between_subject_eta2",
                "value": 0.12,
                "n_subjects": 4,
                "n_observations": 48,
                "design": "nested_subject_within_label",
            },
        ]
    )

    section = build_subject_alignment_diagnostics_section(diagnostics)

    assert section is not None
    assert section.title == "Subject Alignment Diagnostics"
    assert isinstance(section.children[0], CalloutElement)
    assert isinstance(section.children[1], ColumnsElement)
    assert any(isinstance(child, PlotlyElement) for child in section.children)
    assert any(isinstance(child, InteractiveTableElement) for child in section.children)


def test_subject_alignment_diagnostics_section_ignores_empty_frames():
    assert build_subject_alignment_diagnostics_section(pd.DataFrame()) is None


def _cross_model_diagnostics():
    values = {
        "none": {
            "subject_probe_linear_balanced_accuracy": 0.8,
            "subject_probe_chance": 0.25,
            "between_subject_excess_over_null": 2.0,
            "marginal_label_excess_over_null": 1.5,
            "total_sample_variance": 100.0,
            "variance_participation_ratio": 8.0,
        },
        "leace": {
            "subject_probe_linear_balanced_accuracy": 0.5,
            "subject_probe_chance": 0.25,
            "between_subject_excess_over_null": 0.8,
            "marginal_label_excess_over_null": 1.2,
            "total_sample_variance": 70.0,
            "variance_participation_ratio": 6.0,
        },
    }
    return pd.DataFrame(
        [
            {
                "model": model,
                "transform": transform,
                "scope": "pooled",
                "target": "diagnosis",
                "cohort_name": "cohort",
                "population": "clinical_task_subset",
                "selection_fingerprint": "same-selection",
                "eval_name": "diagnosis",
                "target_col": "diagnosis",
                "design": "nested_subject_within_label",
                "metric": metric,
                "value": value,
                "n_subjects": 8,
                "n_observations": 80,
                "n_features": 10,
                "n_constant_features": 0,
            }
            for model in ("model-a", "model-b")
            for transform, metrics in values.items()
            for metric, value in metrics.items()
        ]
    )


def test_alignment_coverage_and_tradeoff_sections_are_paired_by_model():
    diagnostics = _cross_model_diagnostics()
    performance = pd.DataFrame(
        [
            {
                "model": model,
                "decoder": "ridge",
                "transform": transform,
                "scope": "pooled",
                "target": "diagnosis",
                "performance": score,
            }
            for model, raw, transformed in (
                ("model-a", 0.7, 0.69),
                ("model-b", 0.8, 0.74),
            )
            for transform, score in (("none", raw), ("leace", transformed))
        ]
    )

    coverage = build_alignment_coverage_section(diagnostics)
    tradeoff = build_alignment_tradeoff_section(performance, diagnostics)

    assert coverage is not None
    assert tradeoff is not None
    coverage_table = next(
        child
        for child in coverage.children
        if isinstance(child, InteractiveTableElement)
    )
    assert coverage_table.data["same_observation_selection"].all()
    ranking = next(
        child
        for child in tradeoff.children
        if isinstance(child, InteractiveTableElement)
    ).data
    assert set(ranking["model"]) == {"model-a", "model-b"}
    assert ranking["pareto_optimal"].all()
    assert set(ranking["label_variance_retention"].round(2)) == {0.8}


def test_subject_alignment_section_plots_representation_variance_retention():
    section = build_subject_alignment_diagnostics_section(_cross_model_diagnostics())

    assert section is not None
    assert "Representation variance retained after alignment" in section.render()


def test_raw_delta_requires_the_same_population_and_selection_fingerprint():
    diagnostics = pd.DataFrame(
        [
            {
                "transform": "none",
                "population": "clinical_task_subset",
                "selection_fingerprint": "selection-a",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "metric": "between_subject_excess_over_null",
                "value": 0.6,
            },
            {
                "transform": "leace",
                "population": "clinical_task_subset",
                "selection_fingerprint": "selection-b",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "metric": "between_subject_excess_over_null",
                "value": 0.2,
            },
        ]
    )

    paired = _add_raw_delta(diagnostics)

    transformed = paired[paired["transform"] == "leace"].iloc[0]
    assert pd.isna(transformed["raw_value"])
    assert pd.isna(transformed["delta_vs_raw"])


def _selection_frame():
    return pd.DataFrame(
        [
            {
                "transform": "none",
                "cohort_name": cohort,
                "population": population,
                "selection_fingerprint": f"{cohort}-{population}",
                "scope": "pooled",
                "eval_name": "diagnosis",
                "target_col": "diagnosis",
                "metric": "between_subject_excess_over_null",
                "value": 0.4,
            }
            for cohort, population in (
                ("wanted", "clinical_task_subset"),
                ("other", "clinical_task_subset"),
                ("wanted", "transform_training_population"),
            )
        ]
    )


def test_select_subject_alignment_diagnostics_is_exact():
    selected = select_subject_alignment_diagnostics(
        _selection_frame(),
        cohort_name="wanted",
        population="clinical_task_subset",
    )

    assert len(selected) == 1
    assert selected.loc[0, "selection_fingerprint"] == "wanted-clinical_task_subset"


def test_validate_subject_alignment_diagnostics_requires_generic_schema():
    incomplete = _selection_frame().drop(columns="selection_fingerprint")

    try:
        validate_subject_alignment_diagnostics(incomplete)
    except ValueError as exc:
        assert "selection_fingerprint" in str(exc)
    else:
        raise AssertionError("Expected missing-schema validation to fail.")


def test_select_subject_alignment_diagnostics_rejects_absent_assessment():
    try:
        select_subject_alignment_diagnostics(
            _selection_frame(),
            cohort_name="missing",
            population="clinical_task_subset",
        )
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("Expected absent assessment selection to fail.")
