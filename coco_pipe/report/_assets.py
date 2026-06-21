"""
Asset vendoring for offline-friendly reports.

By default a rendered report references three CDN-hosted JS bundles
(Plotly, Tailwind Play CDN, pako). For air-gapped environments or
long-term archival, callers can pass ``asset_urls="inline"`` to
:class:`coco_pipe.report.core.Report` — :func:`get_vendored_contents`
then downloads (once, with on-disk caching) and returns the JS source
strings to inline into the rendered HTML.

The cache lives at ``~/.cache/coco-pipe/report-assets/`` by default;
override with the ``COCO_PIPE_REPORT_ASSET_CACHE`` environment
variable. Downloads use ``urllib.request`` (stdlib) with a 30 s
timeout.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

VENDORED_URLS: dict[str, str] = {
    "plotly": "https://cdn.plot.ly/plotly-2.27.0.min.js",
    "tailwind": "https://cdn.tailwindcss.com",
    "pako": "https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js",
}

INLINE_SENTINEL = "inline"


def _cache_dir() -> Path:
    """Return the on-disk cache directory for vendored asset bundles."""
    override = os.environ.get("COCO_PIPE_REPORT_ASSET_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "coco-pipe" / "report-assets"


_USER_AGENT = "coco-pipe/asset-vendor (+https://github.com/BabaSanfour/coco-pipe)"


def _download_to(path: Path, url: str, timeout: float = 30.0) -> None:
    """Fetch ``url`` into ``path`` atomically.

    Sends a generic User-Agent header because some CDNs (notably
    ``cdn.tailwindcss.com``) return HTTP 403 to requests with the default
    ``Python-urllib/...`` UA.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def get_vendored_contents() -> dict[str, str]:
    """
    Return ``{name: js_source}`` for every vendored asset bundle.

    Downloads any bundle that's not already cached on disk. Subsequent
    calls read from the cache without hitting the network. Raises
    :class:`OSError` (network or filesystem) on failure so the caller
    can fall back to CDN URLs if needed.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    contents: dict[str, str] = {}
    for name, url in VENDORED_URLS.items():
        path = cache / f"{name}.js"
        if not path.exists():
            logger.info("Vendoring %s from %s", name, url)
            _download_to(path, url)
        contents[name] = path.read_text(encoding="utf-8")
    return contents


def vendor_assets(force: bool = False) -> Path:
    """
    Eagerly download every vendored asset and return the cache directory.

    Useful for setting up offline machines: call once with network
    access, then ship the cache dir alongside the code base.

    Parameters
    ----------
    force
        If True, re-download even if the cached file exists.

    Returns
    -------
    Path
        The cache directory containing the downloaded bundles.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in VENDORED_URLS.items():
        path = cache / f"{name}.js"
        if force or not path.exists():
            logger.info("Vendoring %s from %s", name, url)
            _download_to(path, url)
    return cache


__all__ = [
    "INLINE_SENTINEL",
    "VENDORED_URLS",
    "get_vendored_contents",
    "vendor_assets",
]
