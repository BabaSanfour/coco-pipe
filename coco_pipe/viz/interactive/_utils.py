"""Plotly-specific helpers shared across interactive modules."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ..theme import _COLORBLIND_COLORS, DIVERGING, SEQUENTIAL, ColorKind

COCO_TEMPLATE = "coco"


def _register_coco_template() -> None:
    """Register a 'coco' Plotly template that mirrors the static matplotlib theme."""
    base = copy.deepcopy(pio.templates["plotly_white"])
    base.layout.update(
        font=dict(family="Arial, DejaVu Sans, Liberation Sans, sans-serif", size=12),
        colorway=list(_COLORBLIND_COLORS),
        colorscale=dict(
            sequential=SEQUENTIAL,
            diverging=DIVERGING,
            sequentialminus=SEQUENTIAL,
        ),
    )
    base.layout.xaxis.update(gridcolor="rgba(0,0,0,0.12)")
    base.layout.yaxis.update(gridcolor="rgba(0,0,0,0.12)")
    pio.templates[COCO_TEMPLATE] = base


_register_coco_template()


def _apply_layout(
    fig: go.Figure,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int | None = None,
    template: str = COCO_TEMPLATE,
    barmode: str | None = None,
    legend_horizontal: bool = False,
) -> go.Figure:
    """Apply common layout settings to a Plotly figure."""
    kwargs: dict[str, Any] = {
        "template": template,
        "margin": dict(l=50, r=40, b=50, t=55),
    }
    if title is not None:
        kwargs["title"] = title
    if xaxis_title is not None:
        kwargs["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        kwargs["yaxis_title"] = yaxis_title
    if height is not None:
        kwargs["height"] = height
    if barmode is not None:
        kwargs["barmode"] = barmode
    if legend_horizontal:
        kwargs["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    fig.update_layout(**kwargs)
    return fig


def _discrete_colorscale(
    categories: Sequence[Any], palette: Optional[str | Sequence[str]] = None
) -> tuple[list[str], list[list[Any]]]:
    """Build a discrete Plotly colorscale from category values."""
    if palette is None:
        colors = list(_COLORBLIND_COLORS)
    elif isinstance(palette, str):
        colors = list(getattr(px.colors.qualitative, palette, _COLORBLIND_COLORS))
    else:
        colors = list(palette)
    if not colors:
        raise ValueError("palette must contain at least one color.")

    n_categories = max(1, len(categories))
    actual_colors = [colors[i % len(colors)] for i in range(n_categories)]
    scale: list[list[Any]] = []
    step = 1.0 / n_categories
    for i, color in enumerate(actual_colors):
        scale.append([i * step, color])
        scale.append([(i + 1) * step, color])
    return actual_colors, scale


def _marker_restyle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a marker payload into a Plotly restyle payload."""
    colorbar = payload.get("colorbar", {})
    return {
        "marker.color": [payload["color"]],
        "marker.colorscale": [payload["colorscale"]],
        "marker.colorbar.title": colorbar.get("title"),
        "marker.colorbar.tickmode": colorbar.get("tickmode", "auto"),
        "marker.colorbar.tickvals": [colorbar["tickvals"]]
        if "tickvals" in colorbar
        else None,
        "marker.colorbar.ticktext": [colorbar["ticktext"]]
        if "ticktext" in colorbar
        else None,
        "marker.cmin": [payload.get("cmin")],
        "marker.cmax": [payload.get("cmax")],
    }


def _marker_payload(
    df: pd.DataFrame,
    column: str,
    cmap: str,
    palette: Optional[str | Sequence[str]],
    color_kind: ColorKind,
    restyle: bool,
):
    """Build a Plotly marker color payload for categorical or continuous columns."""
    values = df[column]
    if color_kind not in {"categorical", "continuous"}:
        raise ValueError("`color_kind` must be 'categorical' or 'continuous'.")
    if color_kind == "categorical":
        if hasattr(values, "cat"):
            categories = [str(category) for category in values.cat.categories.tolist()]
            lookup_values = values.astype(str)
        else:
            categories = sorted(
                pd.Series(values).dropna().astype(str).unique().tolist()
            )
            lookup_values = pd.Series(values).astype(str)
        cat_map = {cat: i for i, cat in enumerate(categories)}
        mapped = [cat_map.get(v, np.nan) for v in lookup_values]
        _, colorscale = _discrete_colorscale(categories, palette=palette)
        payload = {
            "color": mapped,
            "colorscale": colorscale,
            "colorbar": {
                "title": column,
                "tickmode": "array",
                "tickvals": list(range(len(categories))),
                "ticktext": categories,
            },
            "cmin": 0,
            "cmax": max(1, len(categories) - 1),
        }
        if restyle:
            return _marker_restyle_payload(payload)
        return payload

    continuous_values = pd.to_numeric(values, errors="raise")
    payload = {
        "color": continuous_values,
        "colorscale": cmap,
        "colorbar": {"title": column},
        "cmin": None,
        "cmax": None,
    }
    if restyle:
        return _marker_restyle_payload(payload)
    return payload
