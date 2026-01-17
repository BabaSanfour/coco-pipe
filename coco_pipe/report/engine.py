"""
Template Rendering Engine
=========================

Manages the Jinja2 environment and template loading for the report module.
"""

import jinja2
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Define template directory relative to this file
MODULE_DIR = Path(__file__).parent
TEMPLATE_DIR = MODULE_DIR / "templates"

def _create_env() -> jinja2.Environment:
    """Create and configure the Jinja2 environment."""
    loader = jinja2.FileSystemLoader(str(TEMPLATE_DIR))
    env = jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True
    )
    return env

_ENV: Optional[jinja2.Environment] = None

def get_env() -> jinja2.Environment:
    """Get or create the global Jinja2 environment."""
    global _ENV
    if _ENV is None:
        _ENV = _create_env()
    return _ENV

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
