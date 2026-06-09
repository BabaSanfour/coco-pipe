"""Shared helpers for report section builders."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from .elements import CodeBlockElement, TableElement

if TYPE_CHECKING:
    pass


def _coerce_kind(value: Any, kind: type) -> Any:
    """Return ``value`` if it is an instance of ``kind``, else ``kind()``."""
    return value if isinstance(value, kind) else kind()


def _table_from_mapping(mapping: Mapping[str, Any], *, title: str) -> TableElement:
    """Create a two-column key/value TableElement from a flat mapping."""
    records = [{"Key": str(key), "Value": value} for key, value in mapping.items()]
    # explicitly define columns so an empty mapping yields a
    # valid empty dataframe structure
    df = pd.DataFrame(records, columns=["Key", "Value"])
    return TableElement(df, title=title)


def _config_element(
    config: Mapping[str, Any], *, title: str
) -> TableElement | CodeBlockElement:
    """Return a TableElement for flat configs, CodeBlockElement for nested ones."""
    # Check if any value is a container type (excluding simple strings)
    has_nested = any(
        isinstance(v, (Mapping, Sequence, set)) and not isinstance(v, (str, bytes))
        for v in config.values()
    )

    if has_nested:
        return CodeBlockElement(
            json.dumps(config, indent=2, default=str),
            language="json",
            title=title,
        )
    return _table_from_mapping(config, title=title)


def _resolve_sections(
    sections: list[str] | Literal["default"],
    *,
    default: Sequence[str],
    valid: Iterable[str],
    context: str = "report",
) -> list[str]:
    """Resolve and validate a section selection against the allowed names.

    Parameters
    ----------
    sections
        Either ``"default"`` (use ``default``) or an explicit list of section keys.
    default
        Section keys to use when ``sections == "default"``.
    valid
        Iterable of every allowed section key.
    context
        Human-readable label inserted into the error message (e.g. ``"decoding"``).

    Returns
    -------
    list[str]
        The resolved list of section keys, in caller-specified order.

    Raises
    ------
    ValueError
        If any selected key is not present in ``valid``.
    """
    selected = list(default) if sections == "default" else list(sections)
    valid_set = set(valid)
    unknown = sorted(set(selected) - valid_set)
    if unknown:
        valid_list = ", ".join(sorted(valid_set))
        raise ValueError(
            f"Unknown {context} report section(s): {', '.join(unknown)}. "
            f"Valid sections are: {valid_list}."
        )
    return selected
