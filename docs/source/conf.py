"""Sphinx configuration for coco-pipe documentation."""

from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_SOURCE_DIR = Path(__file__).resolve().parent
DOCS_DIR = DOCS_SOURCE_DIR.parent
REPO_ROOT = DOCS_DIR.parent
PACKAGE_DIR = REPO_ROOT / "coco_pipe"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOCS_SOURCE_DIR / "_ext"))


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "coco-pipe"
author = "coco-pipe developers"

_today = date.today()
copyright = (
    f"2025-{_today.year}, coco-pipe developers. Last updated {_today.isoformat()}"
)

version = "0.0.1"
release = version


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    # Core Sphinx
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    # API generation
    "autoapi.extension",
    # Markdown
    "myst_parser",
    # Design components (grids, cards, tabs)
    "sphinx_design",
    # Examples
    "sphinx_gallery.gen_gallery",
    # UX
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    # Local extensions
    "capability_table",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "_ideas",
]

nitpicky = False
keep_warnings = True


# ---------------------------------------------------------------------------
# Autodoc / Autosummary
# ---------------------------------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": False,
}


# ---------------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(PACKAGE_DIR)]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_ignore = [
    "*/tests/*",
    "*/test_*.py",
    "*/cbramod_src/*",
]


# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}


# ---------------------------------------------------------------------------
# MyST Markdown
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3


# ---------------------------------------------------------------------------
# Copy button
# ---------------------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False


# ---------------------------------------------------------------------------
# Sphinx Gallery
# ---------------------------------------------------------------------------

sphinx_gallery_conf = {
    "doc_module": "coco_pipe",
    "reference_url": {
        "coco_pipe": None,
    },
    "examples_dirs": str(REPO_ROOT / "examples"),
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "generated",
    "filename_pattern": r".*",
    "ignore_pattern": r"__init__\.py",
    "run_stale_examples": False,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "coco-pipe"
html_short_title = "coco-pipe"
html_show_sphinx = False
html_show_copyright = True

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "announcement": (
        "coco-pipe is in active pre-release development — APIs may change before 1.0."
    ),
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_persistent": [],
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BabaSanfour/coco-pipe",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/coco-pipe/",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "github_user": "BabaSanfour",
    "github_repo": "coco-pipe",
    "github_version": "main",
    "doc_path": "docs/source",
}


# Optional: enable once files exist
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"


# ---------------------------------------------------------------------------
# Repository docs pulled into the build (single source of truth at repo root)
# ---------------------------------------------------------------------------

# Files copied from the repository root into the docs source tree at build time
# so they can be referenced from toctrees, then removed afterwards.
_REPO_DOC_COPIES = {
    REPO_ROOT / "CONTRIBUTING.md": DOCS_SOURCE_DIR / "contributing.md",
}


def copy_repo_docs() -> None:
    """Copy repository-root docs into the documentation source directory."""
    for source, destination in _REPO_DOC_COPIES.items():
        if source.exists():
            shutil.copyfile(source, destination)


def cleanup_repo_docs(app, exception) -> None:
    """Remove the copied repository docs after the build."""
    for destination in _REPO_DOC_COPIES.values():
        if destination.exists():
            destination.unlink()


def autoapi_skip_member(app, what, name, obj, skip, options):
    """
    Hook for Sphinx AutoAPI to determine if a specific member should be excluded from the docs.

    AutoAPI works by statically parsing the Python AST without importing code. Because of this,
    it finds classes both where they are defined (e.g., `coco_pipe/io/structures.py`) AND
    where they are re-exported (e.g., `coco_pipe/io/__init__.py`).
    We use this hook to prune out redundant internal paths and unwanted Pydantic boilerplate.
    """
    short_name = name.split(".")[-1]

    # 1. Hide standard Pydantic V1/V2 internal methods that clutter the API documentation.
    # Pydantic injects dozens of helper methods into BaseModel subclasses, which we don't want users to see.
    if short_name.startswith("model_") or short_name in {
        "dict",
        "json",
        "parse_obj",
        "parse_raw",
        "construct",
        "schema",
        "schema_json",
        "update_forward_refs",
    }:
        return True

    # 2. Prevent Duplicate Target Generation.
    # Because these classes are defined in internal files (e.g. `structures.py`) but exported
    # publicly in `__init__.py`, AutoAPI will try to generate TWO separate documentation pages for them.
    # This causes Sphinx to crash with "more than one target found" when we try to cross-reference them.
    # By forcing AutoAPI to skip their internal source path, we ensure they are ONLY documented
    # at their clean, public API path (e.g., `coco_pipe.io.DataContainer`).
    duplicates_to_hide = {
        "coco_pipe.io.structures.DataContainer",
        "coco_pipe.dim_reduction.reducers.base.BaseReducer",
        "coco_pipe.dim_reduction.core.DimReduction",
        "coco_pipe.decoding.configs.ExperimentConfig",
        "coco_pipe.decoding.result.ExperimentResult",
        "coco_pipe.report.core.Report",
        "coco_pipe.report.core.Section",
    }
    if name in duplicates_to_hide:
        return True

    return skip


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Hook for Sphinx Autodoc to determine if a member should be excluded.

    Unlike AutoAPI (which is static), Autodoc works by dynamically importing modules into memory.
    While AutoAPI builds the main API reference, Autodoc is triggered by the `autosummary` tables
    in our index files.
    We must duplicate the Pydantic filter here because Autodoc parses the imported class objects
    and will expose the `model_*` methods in the summary tables otherwise.
    """
    # Hide standard Pydantic internal methods that clutter the API
    if name.startswith("model_") or name in {
        "dict",
        "json",
        "parse_obj",
        "parse_raw",
        "construct",
        "schema",
        "schema_json",
        "update_forward_refs",
    }:
        return True
    return skip


def setup(app):
    """Register Sphinx build hooks."""
    copy_repo_docs()
    app.connect("build-finished", cleanup_repo_docs)
    app.connect("autoapi-skip-member", autoapi_skip_member)
    app.connect("autodoc-skip-member", autodoc_skip_member)


napoleon_use_ivar = True
