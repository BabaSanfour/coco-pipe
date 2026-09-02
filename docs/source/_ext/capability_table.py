"""Custom Sphinx directives for coco-pipe model capability tables.

This extension provides two directives:

.. code-block:: rst

   .. capability-table::
      :task: classification

   .. capability-table::
      :task: regression
      :show-search-space:

   .. foundation-table::
"""

from __future__ import annotations

from typing import Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _yes_no(value: bool) -> str:
    """Return a compact yes/no label."""
    return "yes" if value else "no"


def _task_label(tasks: tuple[str, ...]) -> str:
    """Return a compact task label."""
    labels = {
        "classification": "clf",
        "regression": "reg",
    }
    return " + ".join(labels.get(task, task) for task in tasks)


def _importance_label(values: tuple[str, ...]) -> str:
    """Return a compact feature-importance label."""
    labels = {
        "coefficients": "coef",
        "feature_importances": "feature importances",
        "permutation": "permutation",
        "unavailable": "no",
    }
    return " / ".join(labels.get(value, value) for value in values)


def _family_label(family: str) -> str:
    """Return a display label for an estimator family."""
    return family.replace("_", " ").title()


def _format_unknown(value: Any, *, suffix: str = "") -> str:
    """Format optional values for display."""
    if value in {None, "", "unknown"}:
        return "unknown"
    return f"{value}{suffix}"


def _literal(text: str) -> nodes.literal:
    """Create an inline literal node."""
    return nodes.literal(text, text)


def _text_cell(value: str) -> nodes.entry:
    """Create a table cell containing plain text."""
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    paragraph += nodes.Text(value)
    entry += paragraph
    return entry


def _literal_cell(value: str) -> nodes.entry:
    """Create a table cell containing inline literal text."""
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    paragraph += _literal(value)
    entry += paragraph
    return entry


def _make_entry(value: str, *, literal: bool = False) -> nodes.entry:
    """Create a table entry."""
    return _literal_cell(value) if literal else _text_cell(value)


def _make_row(values: list[tuple[str, bool]]) -> nodes.row:
    """Create a table row.

    Parameters
    ----------
    values
        Pairs of ``(cell_text, is_literal)``.
    """
    row = nodes.row()
    for text, is_literal in values:
        row += _make_entry(text, literal=is_literal)
    return row


def _make_table(
    headers: list[str],
    rows: list[list[tuple[str, bool]]],
    *,
    css_class: str,
) -> nodes.table:
    """Create a docutils table."""
    table = nodes.table(classes=[css_class])
    tgroup = nodes.tgroup(cols=len(headers))
    table += tgroup

    for _ in headers:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    thead += _make_row([(header, False) for header in headers])
    tgroup += thead

    tbody = nodes.tbody()
    for values in rows:
        tbody += _make_row(values)
    tgroup += tbody

    return table


def _warning_node(message: str) -> nodes.warning:
    """Create a Sphinx/docutils warning node."""
    warning = nodes.warning()
    warning += nodes.paragraph(text=message)
    return warning


def _load_estimator_specs() -> dict[str, Any]:
    """Load estimator specifications from coco-pipe."""
    try:
        from coco_pipe.decoding._specs import ESTIMATOR_SPECS
    except Exception as exc:  # pragma: no cover - Sphinx build safeguard
        msg = f"capability-table: could not load ESTIMATOR_SPECS: {exc}"
        logger.warning(msg)
        raise RuntimeError(msg) from exc

    return ESTIMATOR_SPECS


class CapabilityTableDirective(Directive):
    """Generate a table of registered decoding estimators."""

    has_content = False
    optional_arguments = 0
    option_spec: ClassVar[dict] = {
        "task": directives.unchanged,
        "show-search-space": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        """Run the directive."""
        try:
            estimator_specs = _load_estimator_specs()
        except RuntimeError as exc:
            return [_warning_node(str(exc))]

        task_filter = self.options.get("task", "all").strip().lower()
        show_search_space = "show-search-space" in self.options

        specs = list(estimator_specs.values())
        if task_filter != "all":
            specs = [spec for spec in specs if task_filter in spec.task]

        if not specs:
            return [
                nodes.paragraph(text=f"No estimators found for task='{task_filter}'.")
            ]

        headers = [
            "Estimator",
            "Family",
            "Task",
            "Proba",
            "Score fn",
            "Calibrate",
            "Feature sel.",
            "Importances",
            "Temporal",
            "Dep.",
        ]

        if show_search_space:
            headers.append("Search space keys")

        rows = []
        for spec in sorted(specs, key=lambda item: (item.family, item.name)):
            values = [
                (spec.name, True),
                (_family_label(spec.family), False),
                (_task_label(spec.task), False),
                (_yes_no(spec.supports_proba), False),
                (_yes_no(spec.supports_decision_function), False),
                (_yes_no(spec.supports_calibration), False),
                (_yes_no("disabled" not in spec.feature_selection), False),
                (_importance_label(spec.importance), False),
                (spec.temporal if spec.temporal != "none" else "no", False),
                (
                    spec.dependency_extra
                    if spec.dependency_extra != "core"
                    else "core",
                    False,
                ),
            ]

            if show_search_space:
                keys = ", ".join(spec.default_search_space) or "none"
                values.append((keys, True))

            rows.append(values)

        return [
            _make_table(
                headers,
                rows,
                css_class="capability-table",
            )
        ]


class FoundationTableDirective(Directive):
    """Generate a table of registered foundation models."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: ClassVar[dict] = {}

    def run(self) -> list[nodes.Node]:
        """Run the directive."""
        try:
            estimator_specs = _load_estimator_specs()
        except RuntimeError as exc:
            return [_warning_node(str(exc))]

        specs = [
            spec for spec in estimator_specs.values() if spec.family == "foundation"
        ]

        if not specs:
            return [nodes.paragraph(text="No foundation models found.")]

        headers = [
            "Model",
            "Hub repo",
            "Emb. dim",
            "sfreq",
            "Channels",
            "Interpolation",
            "Train modes",
            "Backend",
        ]

        rows = []
        for spec in sorted(specs, key=lambda item: item.name):
            channels = getattr(spec, "pretrained_n_chans", None)
            train_modes = getattr(spec, "supported_train_modes", [])

            values = [
                (getattr(spec, "display_name", spec.name) or spec.name, False),
                (getattr(spec, "hub_repo", "unknown"), True),
                (_format_unknown(getattr(spec, "embedding_dim", None)), False),
                (
                    _format_unknown(
                        getattr(spec, "pretrained_sfreq", None),
                        suffix=" Hz",
                    ),
                    False,
                ),
                (str(channels) if channels else "varies", False),
                (
                    _yes_no(
                        getattr(
                            spec,
                            "supports_channel_interpolation",
                            False,
                        )
                    ),
                    False,
                ),
                (", ".join(train_modes) or "none", True),
                (getattr(spec, "preferred_backend", "unknown"), True),
            ]

            rows.append(values)

        return [
            _make_table(
                headers,
                rows,
                css_class="foundation-table",
            )
        ]


def setup(app):
    """Register the custom Sphinx directives."""
    app.add_directive("capability-table", CapabilityTableDirective)
    app.add_directive("foundation-table", FoundationTableDirective)

    return {
        "version": "1.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
