# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath("../.."))

from aml.__version__ import __version__, __doc__, __author__, __title__, __copyright__

version = __version__
doc = __doc__
author = __author__
project = __title__
copyright = __copyright__
release = '.'.join(version.split('.')[:2])

# -- General configuration

extensions = [
    'readthedocs_ext.readthedocs',
    'sphinx.ext.duration',
    "sphinx_rtd_theme",
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# --, options for HTML output

html_theme = 'sphinx_rtd_theme'

# --, options for EPUB output
epub_show_urls = 'footnote'

master_doc = 'index'


napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
autodoc_typehints = 'both'


import re

def remove_default_value(app, what, name, obj, options, signature, return_annotation):
    if signature:
        search = re.findall(r"(\w*)=", signature)
        if search:
            signature = "({})".format(", ".join([s for s in search]))

    return (signature, return_annotation)

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    #app.connect("autodoc-process-signature", remove_default_value)
    app.connect("autodoc-skip-member", skip)

