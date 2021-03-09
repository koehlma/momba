import typing as t

import os
import sys
import subprocess
import re
import pathlib
import urllib.request
import zipfile
import subprocess

from pygments.lexer import RegexLexer
from pygments import token


MODEST_URL = (
    "https://www.modestchecker.net/"
    "Downloads/Modest-Toolset-v3.1.75-gcc6169502-linux-x64.zip"
)
MODEST_PATH = pathlib.Path(__file__).parent / "build" / "modest"
MODEST_ZIP = MODEST_PATH / "Modest-Toolset.zip"
MODEST_BIN = MODEST_PATH / "Modest"
MODEST_EXE = MODEST_BIN / "modest"

MODEST_PATH.mkdir(parents=True, exist_ok=True)

if not MODEST_ZIP.exists():
    response = urllib.request.urlopen(MODEST_URL)
    assert response.status == 200
    MODEST_ZIP.write_bytes(response.read())

if not MODEST_EXE.exists():
    with zipfile.ZipFile(MODEST_ZIP) as modest_zip:
        modest_zip.extractall(MODEST_PATH)
    subprocess.check_call(["chmod", "+x", MODEST_EXE])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "racetrack")
)

os.environ["PATH"] = ":".join(
    (
        str(MODEST_BIN.resolve()),
        os.environ.get("PATH", ""),
    )
)


os.environ["PYTHONPATH"] = ":".join(
    (
        os.path.join(os.path.dirname(__file__), "..", "examples", "racetrack"),
        os.path.join(os.path.dirname(__file__), "..", "examples", "guide"),
        os.environ.get("PYTHONPATH", ""),
    )
)


import racetrack  # noqa

import momba  # noqa

from momba import moml  # noqa


class MomlLexer(RegexLexer):  # type:ignore
    name = "Moml"
    aliases = ["moml"]
    filenames = ["*.moml"]

    tokens = {
        "root": [
            (r"\s+", token.Whitespace),
            (
                r"edge|transient|instance|variable|probability|network|location|model_type|from|"
                r"rate|automaton|guard|initial|invariant|constant|composition|synchronize|input|"
                r"action|enable|restrict|to|metadata",
                token.Keyword,
            ),
            (r"bool|int|real|clock|continuous", token.Keyword),
            (r"\"[^\"]*\"", token.String),
            (r"\\d+\\.\\d+", token.Number),
            (r"\\d+", token.Number),
            (r":|\[|\]|\(|\)|,", token.Punctuation),
            (
                r":=|→|->|≤|<=|≥|>=|<|>|∧|and|∨|or|⊕|xor|⇒|==>|⇐|<==|⇔|<=>|¬|not|=|≠|!=",
                token.Operator,
            ),
            (r"\+|-|\*|/|//|%", token.Operator),
            (r"\w+", token.Name),
        ]
    }


class BNFLexer(RegexLexer):  # type:ignore
    name = "BNF"
    aliases = ["bnf"]

    tokens = {
        "root": [
            (r"\s+", token.Whitespace),
            (r"<[^>]*>", token.Keyword),
            (r"‘[^’]*’", token.String),
            (r"\[|\]|\(|\)", token.Punctuation),
            (r"\||::=|\*|\+", token.Operator),
            (r"/([^/]|\\/)*/", token.Text),
            (r"…[^…]*…", token.Comment),
            (r"[\w\-]+", token.Name),
        ]
    }


project = "Momba"
copyright = "2020-2021, Dependable Systems and Software Group, Saarland University"
author = "Maximilian Köhl"

try:
    release = (
        subprocess.check_output(["git", "describe", "--dirty"]).decode().strip()[1:]
    )
except subprocess.CalledProcessError:
    release = "master"

version = release


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "jupyter_sphinx",
]

# disable type hints
autodoc_typehints = "none"

html_show_sphinx = False

templates_path = ["_templates"]

html_theme = "furo"

html_title = "Momba"

html_static_path = ["_static"]
html_css_files = ["css/jupyter.css", "css/fix-source-link.css"]

html_theme_options: t.Dict[str, t.Any] = {
    "light_logo": "images/logo_with_text.svg",
    "dark_logo": "images/logo_with_text_white.svg",
    "sidebar_hide_name": True,
}


type_re = re.compile(":type:.*$")


def process_docstring(app, what, name, obj, options, lines):  # type: ignore
    for index, line in enumerate(lines):
        match = type_re.search(line)
        if match:
            # strip away the type information
            lines[index] = line[: match.span()[0]]


def setup(app):  # type:ignore
    from sphinx.highlighting import lexers

    app.connect("autodoc-process-docstring", process_docstring)

    lexers["moml"] = MomlLexer()
    lexers["bnf"] = BNFLexer()
