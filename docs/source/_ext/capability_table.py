"""
capability_table
================

A custom Sphinx directive that reads ``ESTIMATOR_SPECS`` from
``coco_pipe.decoding._specs`` at build time and emits a formatted RST
``list-table`` showing each estimator's capabilities.

Usage in any .rst file::

    .. capability-table::
        :task: classification

    .. capability-table::
        :task: regression

    .. capability-table::
        :task: all

Options
-------
task : str, default="all"
    Filter by task: ``classification``, ``regression``, or ``all``.
show-search-space : flag
    If present, append a column with the default search space keys.
"""

from __future__ import annotations

import os
import sys
from typing import List

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Ensure the package root is on sys.path so we can import coco_pipe
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _yes_no(value: bool) -> str:
    return "✅" if value else "❌"


def _importance_label(tup: tuple) -> str:
    mapping = {
        "coefficients": "coef\\_",
        "feature_importances": "feat\\_imp",
        "permutation": "permutation",
        "unavailable": "❌",
    }
    return " / ".join(mapping.get(v, v) for v in tup)


def _task_label(tup: tuple) -> str:
    abbrev = {"classification": "clf", "regression": "reg"}
    return " + ".join(abbrev.get(t, t) for t in tup)


def _family_label(family: str) -> str:
    return family.capitalize()


class CapabilityTableDirective(Directive):
    """
    Sphinx directive ``.. capability-table::``

    Emits an RST list-table of all registered estimators and their key
    capabilities, optionally filtered by task.
    """

    has_content = False
    optional_arguments = 0
    option_spec = {
        "task": directives.unchanged,
        "show-search-space": directives.flag,
    }

    # Column definitions: (header, width, extractor)
    _COLUMNS = [
        ("Estimator", 28, lambda s: f"``{s.name}``"),
        ("Family", 12, lambda s: _family_label(s.family)),
        ("Task", 9, lambda s: _task_label(s.task)),
        ("Proba", 6, lambda s: _yes_no(s.supports_proba)),
        ("Score fn", 8, lambda s: _yes_no(s.supports_decision_function)),
        ("Calibrate", 9, lambda s: _yes_no(s.supports_calibration)),
        ("Feature sel", 11, lambda s: _yes_no("disabled" not in s.feature_selection)),
        ("Importances", 13, lambda s: _importance_label(s.importance)),
        ("Temporal", 11, lambda s: s.temporal if s.temporal != "none" else "❌"),
        (
            "Dep",
            7,
            lambda s: s.dependency_extra if s.dependency_extra != "core" else "—",
        ),
    ]

    def run(self) -> List[nodes.Node]:
        try:
            # Import directly from submodule to avoid triggering coco_pipe.__init__
            # which may pull in optional heavy dependencies (pydantic, etc.)
            import importlib.util
            import pathlib

            spec_file = (
                pathlib.Path(_REPO_ROOT) / "coco_pipe" / "decoding" / "_specs.py"
            )
            spec_mod = importlib.util.spec_from_file_location("_specs", spec_file)
            mod = importlib.util.module_from_spec(spec_mod)
            # _specs.py depends on _constants.py — load that first
            const_file = (
                pathlib.Path(_REPO_ROOT) / "coco_pipe" / "decoding" / "_constants.py"
            )
            const_spec = importlib.util.spec_from_file_location(
                "_constants", const_file
            )
            const_mod = importlib.util.module_from_spec(const_spec)
            const_spec.loader.exec_module(const_mod)
            import sys as _sys

            _sys.modules["coco_pipe.decoding._constants"] = const_mod
            spec_mod.loader.exec_module(mod)
            ESTIMATOR_SPECS = mod.ESTIMATOR_SPECS
        except Exception as exc:
            error_msg = f"capability-table: could not load ESTIMATOR_SPECS: {exc}"
            logger.warning(error_msg)
            return [nodes.warning("", nodes.paragraph(text=error_msg))]

        task_filter = self.options.get("task", "all").strip().lower()
        show_search = "show-search-space" in self.options

        specs = list(ESTIMATOR_SPECS.values())
        if task_filter != "all":
            specs = [s for s in specs if task_filter in s.task]

        if not specs:
            return [
                nodes.paragraph(text=f"No estimators found for task='{task_filter}'.")
            ]

        columns = list(self._COLUMNS)
        if show_search:
            columns.append(
                (
                    "Search space keys",
                    20,
                    lambda s: ", ".join(f"``{k}``" for k in s.default_search_space)
                    or "—",
                )
            )

        # Build the RST list-table text
        col_widths = " ".join(str(c[1]) for c in columns)
        header_cells = "\n".join(f"   * - {c[0]}" for c in columns)

        rows_rst = []
        for spec in sorted(specs, key=lambda s: (s.family, s.name)):
            cells = []
            for i, (_, _, extractor) in enumerate(columns):
                prefix = "     - " if i > 0 else "   * - "
                cells.append(f"{prefix}{extractor(spec)}")
            rows_rst.append("\n".join(cells))

        table_rst = (
            f".. list-table::\n"
            f"   :header-rows: 1\n"
            f"   :widths: {col_widths}\n"
            f"\n"
            f"{header_cells}\n" + "\n".join(rows_rst)
        )

        # Parse the generated RST into docutils nodes
        from docutils.statemachine import ViewList
        from sphinx.util.docutils import switch_source_input

        result = ViewList()
        source = self.get_source_info()[0]
        for lineno, line in enumerate(table_rst.splitlines()):
            result.append(line, source, lineno)

        node = nodes.section()
        node.document = self.state.document
        with switch_source_input(self.state, result):
            self.state.nested_parse(result, 0, node)

        return node.children


def setup(app):
    app.add_directive("capability-table", CapabilityTableDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
