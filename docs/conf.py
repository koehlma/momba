import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import momba  # noqa


project = "Momba"
copyright = "2019, Dependable Systems and Software Group, Saarland University"
author = "Maximilian Köhl"

try:
    release = (
        subprocess.check_output(["git", "describe", "--dirty"]).decode().strip()[1:]
    )
except subprocess.CalledProcessError:
    release = "unknown"

version = release


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

html_theme_options = {"display_version": True}

templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"

html_static_path = []  # type: ignore
