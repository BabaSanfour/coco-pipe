"""Reports for foundation-model extraction and decoding workflows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from .core import Report, Section
from .decoding import make_decoding_report
from .elements import TableElement


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
    result: Any,
    *,
    capability_records: Iterable[Mapping[str, Any]] = (),
    title: str = "Foundation Model Decoding",
    output_path: str | None = None,
    asset_urls: dict[str, str] | str | None = "inline",
    **kwargs: Any,
) -> Report:
    """Build the standard decoding report plus a model/mode capability matrix."""
    report = make_decoding_report(
        result,
        title=title,
        asset_urls=asset_urls,
        **kwargs,
    )
    capabilities = pd.DataFrame([dict(record) for record in capability_records])
    if not capabilities.empty:
        section = Section("Foundation Capability Matrix")
        section.add_markdown(
            "Unsupported combinations are reported explicitly and are never "
            "silently downgraded to another training mode."
        )
        section.add_element(TableElement(capabilities, title="Preflight Decisions"))
        report.add_section(section)
    if output_path is not None:
        report.save(output_path)
    return report
