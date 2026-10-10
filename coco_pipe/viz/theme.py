from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

ThemeMode = Literal["paper", "notebook", "poster"]
ColorKind = Literal["categorical", "continuous"]

DIVERGING: str = "RdBu_r"
SEQUENTIAL: str = "viridis"
QUALITATIVE: str = "tab10"
QUALITATIVE_COLORBLIND: str = "colorblind"
_COLORBLIND_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
)

_PAPER_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_NOTEBOOK_RC: dict = {
    **_PAPER_RC,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
}
_POSTER_RC: dict = {
    **_PAPER_RC,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
}

_MODE_RC = {"paper": _PAPER_RC, "notebook": _NOTEBOOK_RC, "poster": _POSTER_RC}


def _rc_for(mode: ThemeMode, colorblind: bool) -> dict:
    if mode not in _MODE_RC:
        raise ValueError(f"Unknown coco theme mode: {mode!r}")
    rc = dict(_MODE_RC[mode])
    if colorblind:
        import cycler

        rc["axes.prop_cycle"] = cycler.cycler(color=_COLORBLIND_COLORS)
    return rc


@contextmanager
def coco_theme(mode: ThemeMode = "paper", colorblind: bool = False):
    """Scope rcParams to a with-block. Safe in notebooks and tests."""
    import matplotlib as mpl

    old = mpl.rcParams.copy()
    mpl.rcParams.update(_rc_for(mode, colorblind))
    try:
        yield
    finally:
        mpl.rcParams.update(old)


def set_coco_theme(mode: ThemeMode = "paper", colorblind: bool = False) -> None:
    """Set rcParams globally. Use coco_theme() context manager in notebooks/tests."""
    import matplotlib as mpl

    mpl.rcParams.update(_rc_for(mode, colorblind))


def figure_size(
    columns: int = 1,
    aspect_ratio: float = 1.0,
    width_pt: float | None = None,
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches."""
    if columns not in {1, 2}:
        raise ValueError("columns must be 1 or 2")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    width_in = (
        (width_pt / 72.27) if width_pt is not None else (3.5 if columns == 1 else 7.0)
    )
    return width_in, width_in * aspect_ratio


def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 300,
    facecolor: str = "white",
    bbox_inches: str = "tight",
) -> None:
    """Save a Matplotlib figure with coco_pipe defaults.

    Parameters
    ----------
    fig
        Matplotlib figure object to save.
    path
        Output path accepted by ``Figure.savefig``.
    dpi
        Output resolution in dots per inch.
    facecolor
        Figure background color.
    bbox_inches
        Bounding-box mode passed to ``Figure.savefig``.
    """
    fig.savefig(path, dpi=dpi, facecolor=facecolor, bbox_inches=bbox_inches)


def _palette_for(signed: bool) -> str:
    """Return DIVERGING if signed else SEQUENTIAL."""
    return DIVERGING if signed else SEQUENTIAL
