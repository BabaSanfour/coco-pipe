# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

# sys.path.insert(0, os.path.abspath('.'))

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "coco-pipe")))

import shutil

def copy_readme():
    """Copy README.md from the root directory to docs/source/."""
    source = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../README.md"))
    destination = os.path.abspath(os.path.join(os.path.dirname(__file__), "README.md"))
    
    if os.path.exists(source):
        shutil.copyfile(source, destination)
        print(f"Copied {source} -> {destination}")

def cleanup_readme(app, exception):
    """Delete README.md in docs/source/ after the build."""
    destination = os.path.abspath(os.path.join(os.path.dirname(__file__), "README.md"))
    
    if os.path.exists(destination):
        os.remove(destination)
        print(f"Deleted {destination} after build.")

# Register the cleanup function to run at the end
def setup(app):
    app.connect("build-finished", cleanup_readme)

copy_readme()

# -- Project information -----------------------------------------------------

project = 'coco-pipe'
copyright = '2025, HA'
author = "coco-pipe developers"
_today = date.today()
copyright = f"2025-{_today.year}, cocopipe developers. Last updated {_today.isoformat()}"

# The short X.Y version
version = '0.0.1'
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
#    "sphinx.ext.intersphinx",
    #"numpydoc",
    "sphinx_gallery.gen_gallery",
    #"gh_substitutions",  # custom extension, see ./sphinxext/gh_substitutions.py
    "sphinx_copybutton",
    'sphinxcontrib.mermaid',
    'sphinx.ext.napoleon',
    "myst_parser",
]

# Allow Markdown files to be used as documentation pages
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


master_doc = "index"
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

sphinx_gallery_conf = {
    "doc_module": "coco-pipe",
    "reference_url": {
        "coco-pipe": None,
    },
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "backreferences_dir": "generated",
    'run_stale_examples': True, #Force (or not) re running examples,
    #"default_thumb_file": "_static/default_thumbnail.png",  # Set a global default thumbnail

}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store",'_ideas']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static'] #already done in the setup(app) section
html_extra_path = ['_copyover']


###################################################################################################
# Seems like this is not needed anymore ###########################################################
# Replace gallery.css for changing the highlight of the output cells in sphinx gallery
# See:
# https://github.com/sphinx-gallery/sphinx-gallery/issues/399
# https://github.com/sphinx-doc/sphinx/issues/2090
# https://github.com/sphinx-doc/sphinx/issues/7747
# def setup(app):
#    app.connect('builder-inited', lambda app: app.config.html_static_path.append('_static'))
#    app.add_css_file('gallery.css')
###################################################################################################

# Auto API
extensions += ['autoapi.extension']

autoapi_type = 'python'
autoapi_dirs = ["../../coco-pipe"]

extensions += ['sphinx.ext.viewcode'] #see https://github.com/readthedocs/sphinx-autoapi/issues/422