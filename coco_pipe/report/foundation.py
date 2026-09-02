"""Reports for foundation-model extraction and decoding workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from .core import Report, Section
from .decoding_sweep import (
    build_capability_matrix_section,
    build_foundation_comparison_sections,
    hp_tuning_section,
    make_decoding_sweep_report,
)
from .elements import TableElement

_FOUNDATION_STRATEGY_NOTE = (
    "Linear probes are treated as the primary foundation comparison; full "
    "fine-tuning and LoRA are secondary."
)
_FOUNDATION_LEADERBOARDS = (
    {
        "title": "Linear Probe Leaderboard",
        "table_title": "Primary linear-probe leaderboard",
        "filters": {"train_mode": "linear_probe"},
        "comparison_axis": "model_key",
        "group_by": ("condition", "target"),
    },
    {
        "title": "LoRA Leaderboard",
        "table_title": "Low-Rank Adaptation leaderboard",
        "filters": {"train_mode": "lora"},
        "comparison_axis": "model_key",
        "group_by": ("condition", "target"),
    },
    {
        "title": "Full Fine-Tuning Leaderboard",
        "table_title": "Full fine-tuning leaderboard",
        "filters": {"train_mode": "full"},
        "comparison_axis": "model_key",
        "group_by": ("condition", "target"),
    },
)


def make_foundation_embedding_report(
    records: Iterable[Mapping[str, Any]],
    title: str = "Foundation Embedding Extraction",
    config: dict[str, Any] | None = None,
    output_path: str | None = None,
    asset_urls: dict[str, str] | str | None = "inline",
) -> Report:
    """Build a dataset-level extraction coverage and provenance report."""
    frame = pd.DataFrame([dict(record) for record in records])
    report = Report(title=title, config=config or {}, asset_urls=asset_urls)
    overview = Section("Extraction Overview")
    if frame.empty:
        overview.add_markdown("No embedding extraction records were produced.")
    else:
        status = frame.get("status", pd.Series(["unknown"] * len(frame)))
        summary = pd.DataFrame(
            {
                "Metric": ["Artifacts", "Successful", "Failed", "Skipped"],
                "Value": [
                    len(frame),
                    int((status == "success").sum()),
                    int((status == "failed").sum()),
                    int((status == "skipped").sum()),
                ],
            }
        )
        overview.add_element(TableElement(summary, title="Run Coverage"))
        display_columns = [
            column
            for column in (
                "subject",
                "session",
                "task",
                "run",
                "condition",
                "model_key",
                "model_checkpoint",
                "window_count",
                "embedding_shape",
                "status",
                "reason",
                "artifact_path",
            )
            if column in frame.columns
        ]
        overview.add_element(
            TableElement(frame[display_columns], title="Embedding Inventory")
        )
    report.add_section(overview)

    if not frame.empty:
        adaptation_frame = frame.copy()
        if "channel_adaptation" in adaptation_frame:
            nested = adaptation_frame["channel_adaptation"].map(
                lambda value: value if isinstance(value, Mapping) else {}
            )
            for column in (
                "interpolated_channels",
                "zero_filled_channels",
                "dropped_channels",
                "interpolation_method",
                "interpolation_matrix_shape",
            ):
                adaptation_frame[column] = nested.map(
                    lambda value, key=column: value.get(key)
                )
        adaptation_columns = [
            column
            for column in (
                "model_key",
                "input_sfreq",
                "pretrained_sfreq",
                "requires_resampling",
                "original_channels",
                "model_channels",
                "interpolated_channels",
                "zero_filled_channels",
                "dropped_channels",
            )
            if column in adaptation_frame.columns
        ]
        if adaptation_columns:
            adaptation = Section("Signal Adaptation")
            adaptation.add_markdown(
                "Channel and sampling-rate changes are shown explicitly because "
                "montage mismatch changes the scientific interpretation of embeddings."
            )
            adaptation.add_element(
                TableElement(
                    adaptation_frame[adaptation_columns],
                    title="Adaptation Inventory",
                )
            )
            report.add_section(adaptation)
    if output_path is not None:
        report.save(output_path)
    return report


def make_foundation_decoding_report(
    records: Iterable[Mapping[str, Any]],
    *,
    capability_records: Iterable[Mapping[str, Any]] = (),
    title: str = "Foundation Model Decoding",
    dataset_name: str = "dataset",
    strategy_note: str | None = None,
    group_by: tuple[str, ...] = ("condition", "target"),
    per_result_sections: str = "compact",
    include_hp_tuning: bool = True,
    frame: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    asset_urls: dict[str, str] | str | None = "inline",
) -> Report:
    """Build the foundation decoding sweep report.

    Renders the shared sweep skeleton for a foundation sweep: scientific
    overview, a preflight capability matrix, per-train-mode leaderboards, the
    model/train-mode comparison figures + per-result diagnostics, and a failures
    section. ``records`` are the per-model foundation decoding rows;
    ``capability_records`` are the preflight model/train-mode decisions. Pass a
    pre-built *frame* (from :func:`prepare_sweep_frame`) to avoid re-preparing it
    when the caller already has one.
    """
    capability_section = build_capability_matrix_section(capability_records)
    comparison_body = partial(
        build_foundation_comparison_sections,
        group_by=group_by,
        per_result_sections=per_result_sections,
    )

    def _body(report: Report, body_frame: pd.DataFrame) -> None:
        comparison_body(report, body_frame)
        if include_hp_tuning:
            tuning = hp_tuning_section(body_frame)
            if tuning is not None:
                report.add_section(tuning)

    return make_decoding_sweep_report(
        records,
        frame=frame,
        title=title,
        kind="foundation",
        dataset_name=dataset_name,
        strategy_note=strategy_note or _FOUNDATION_STRATEGY_NOTE,
        empty_message="No foundation decoding units were produced.",
        leaderboards=_FOUNDATION_LEADERBOARDS,
        pre_sections=[capability_section] if capability_section is not None else [],
        body=_body,
        scope_from="condition",
        default_scope=None,
        config=config,
        asset_urls=asset_urls,
        output_path=output_path,
    )
