"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use resolve() to make it absolute, like shown here.
#
from pathlib import Path

import imap_processing

sys.path.insert(0, Path("../../imap_processing").resolve())

# -- Project information -----------------------------------------------------

project = "imap_processing"
copyright = "2024, Regents of the University of Colorado"
author = "IMAP Science Operations Center"

# The full version, including alpha/beta/rc tags
release = imap_processing.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.githubpages",  # Helpful for publishing to gh-pages
    "sphinx.ext.napoleon",
    "sphinxcontrib.openapi",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_logo = "_static/imap-mark-hor-multicolor-dark.png"

html_theme_options = {
    "navigation_with_keys": True,
    "github_url": "https://github.com/IMAP-Science-Operations-Center/imap_processing",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Autosummary
autosummary_generate = True
# Do not generate separate pages for class methods
numpydoc_show_class_members = False

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "space_packet_parser": (
        "https://space-packet-parser.readthedocs.io/en/stable/",
        None,
    ),
}

# Reference targets not found
nitpicky = True

# Some inherited method targets aren't found through intersphinx
# TODO: comeback and clean this up later
nitpick_ignore_regex = [
    (r"py:.*", r".*APID\..*"),
    (r"py:.*", r".*Scitype.*"),
    (r"py:.*", r".*Mode.*"),
    (r"py:.*", r".*SWAPIMODE.*"),
    (r"py:.*", r".*TriggerDescription.*"),
    (r"py:.*", r".*IntEnum.*"),
    (r"py:.*", r".*space_packet_parser.*"),
    (r"py:.*", r".*CoDICECompression.*"),
    (r"py:.*", r".*.lo.l0.utils.*"),
    (r"py:.*", r".*.lo.l0.data_classes.*"),
    (r"py:.*", r".*.hi.l1b.hi_l1b.CoincidenceBitmap.*"),
    (r"py:.*", r".*.hi.l1b.hi_l1b.TriggerId.*"),
    (r"py:.*", r".*.hit.l0.utils.*"),
    (r"py:.*", r".*.hit.l0.data_classes.*"),
    (r"py:.*", r".*.hit.l1a.*"),
    (r"py:.*", r".*InitVar*"),
    (r"py:.*", r".*.glows.utils.constants.TimeTuple.*"),
    (r"py:.*", r".*glows.utils.constants.DirectEvent.*"),
    (r"py:.*", r".*numpy.int.*"),
    (r"py:.*", r".*np.ndarray.*"),
    (r"py:.*", r".*numpy._typing._array_like._ScalarType_co.*"),
    (r"py:.*", r".*idex.l1a.TRIGGER_DESCRIPTION.*"),
    (r"py:.*", r".*.spice.geometry.SpiceBody.*"),
    (r"py:.*", r".*.spice.geometry.SpiceFrame.*"),
    (r"py:class", r"numpy._typing.*"),
    (r"py:class", r"^numpy\.(u?int(?:8|16|32|64))$"),
]

# Ignore the inherited members from the <instrument>APID IntEnum class
numpydoc_show_inherited_class_members = {
    "imap_processing.hit.l0.hit_l1a_decom.HitAPID": False,
    "imap_processing.codice.utils.CODICEAPID": False,
}

# Suppress specific warnings
suppress_warnings = [
    "autosectionlabel.*"
]  # Duplicate label use (e.g. imap_processing.codice.codice_l0)
