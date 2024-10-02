"""Sphinx configuration."""

project = "sonicdb"
author = "Daniel Kadyrov"
copyright = "2024, Daniel Kadyrov"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
