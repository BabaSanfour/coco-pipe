"""
Jinja template rendering for HTML reports.
"""

import functools
from typing import Any

import jinja2

from . import _constants

MODULE_DIR = _constants.MODULE_DIR
TEMPLATE_DIR = _constants.TEMPLATE_DIR


@functools.lru_cache(maxsize=1)
def get_env() -> jinja2.Environment:
    """Get or create the global Jinja2 environment."""
    if not TEMPLATE_DIR.is_dir():
        raise RuntimeError(f"Template directory not found: {TEMPLATE_DIR}")

    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(template_name: str, **context: Any) -> str:
    """
    Render a specific template with the provided context.

    Parameters
    ----------
    template_name : str
        Name of the template file in `coco_pipe/report/templates/`.
    **context : dict
        Variables to pass to the template.

    Returns
    -------
    str
        Rendered HTML string.
    """
    env = get_env()
    template = env.get_template(template_name)
    return template.render(**context)
