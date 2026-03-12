"""
Shared package utilities.

This module holds small helpers that are not specific to one subpackage.
"""

from typing import Any, Optional

__all__ = ["import_optional_dependency"]


def import_optional_dependency(
    loader: Any,
    feature: str,
    dependency: str,
    install_hint: Optional[str] = None,
) -> Any:
    """
    Lazily import an optional dependency with clearer failure modes.

    Parameters
    ----------
    loader : callable
        Zero-argument callable returning the imported dependency.
    feature : str
        Feature or component name using the dependency.
    dependency : str
        Human-readable dependency name.
    install_hint : str, optional
        Installation hint shown on ImportError.

    Returns
    -------
    Any
        Imported dependency returned by ``loader``.

    Raises
    ------
    ImportError
        If the dependency is not installed.
    RuntimeError
        If the dependency is installed but fails during initialization.
    """
    try:
        return loader()
    except ImportError as exc:
        msg = f"{dependency} is required for {feature}."
        if install_hint:
            msg += f" Install it with '{install_hint}'."
        raise ImportError(msg) from exc
    except Exception as exc:
        raise RuntimeError(
            f"{dependency} failed to initialize for {feature}: {exc}"
        ) from exc
