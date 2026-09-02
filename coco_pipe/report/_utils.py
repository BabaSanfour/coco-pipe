"""Shared helpers for report section builders."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from .elements import (
    CodeBlockElement,
    DownloadAssetElement,
    ImageElement,
    PlotlyElement,
    TableElement,
    TabsElement,
)

if TYPE_CHECKING:
    from .core import Section

logger = logging.getLogger(__name__)


def _figure_element(
    plot_result: Any,
    caption: str | None = None,
    *,
    width: str = "100%",
) -> ImageElement:
    """Eagerly render a Matplotlib figure (or ``(fig, ax)`` tuple) to a PNG element.

    ``ImageElement`` can encode a live figure on its own, but it does so lazily at
    render time. Encoding here and closing the figure immediately keeps peak memory
    bounded across large sweeps, where lazy encoding would retain every open figure.
    """
    import matplotlib.pyplot as plt

    figure = plot_result[0] if isinstance(plot_result, tuple) else plot_result
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    finally:
        plt.close(figure)
    return ImageElement(buffer.getvalue(), caption=caption, width=width)


def _csv_download(
    frame: pd.DataFrame,
    filename: str,
    label: str,
) -> DownloadAssetElement:
    """Return a gray CSV download button for *frame*."""
    return DownloadAssetElement(
        frame.to_csv(index=False),
        filename,
        "text/csv",
        label=label,
        style="gray",
    )


def _plotly_element(figure: Any) -> PlotlyElement:
    """Wrap a Plotly figure (or ``(fig, ...)`` tuple) in a ``PlotlyElement``."""
    return PlotlyElement(figure[0] if isinstance(figure, tuple) else figure)


def _plot_or_none(
    plotter: Callable[..., Any],
    *args: Any,
    caption: str,
    as_plotly: bool = False,
    **kwargs: Any,
) -> ImageElement | PlotlyElement | None:
    """Render a plot to an element, returning ``None`` if it cannot be drawn.

    With ``as_plotly=True`` the plotter is expected to return a Plotly figure,
    wrapped in a :class:`PlotlyElement`; otherwise a Matplotlib figure is encoded
    to a PNG :class:`ImageElement`.
    """
    try:
        result = plotter(*args, **kwargs)
    except (ImportError, TypeError, ValueError) as exc:
        logger.debug("%s skipped: %s", caption, exc)
        return None
    return _plotly_element(result) if as_plotly else _figure_element(result, caption)


def _add_tabs_or_single(
    container: Section | Any,
    elements: Mapping[str, ImageElement | None],
) -> None:
    """Add one image directly, or several as a tab group, skipping ``None``."""
    available = {label: image for label, image in elements.items() if image is not None}
    if len(available) == 1:
        container.add_element(next(iter(available.values())))
    elif available:
        container.add_element(TabsElement(available))


def _ensure_static_matplotlib_backend() -> None:
    """Use a headless backend before report figures are created."""
    import matplotlib

    if str(matplotlib.get_backend()).lower() != "agg":
        matplotlib.use("Agg", force=True)


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
