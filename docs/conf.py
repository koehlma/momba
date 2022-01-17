import typing as t

import os
import sys
import subprocess
import re
import subprocess

from pygments.lexer import RegexLexer
from pygments import token


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "racetrack")
)

os.environ["PYTHONPATH"] = ":".join(
    (
        os.path.join(os.path.dirname(__file__), "..", "examples", "racetrack"),
        os.path.join(os.path.dirname(__file__), "..", "examples", "airtrack"),
        os.path.join(os.path.dirname(__file__), "..", "examples", "guide"),
        os.environ.get("PYTHONPATH", ""),
    )
)


import racetrack  # noqa

import momba  # noqa

from momba import moml  # noqa
from momba import gym  # noqa


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
copyright = "2020-2022, Dependable Systems and Software Group, Saarland University"
# author = "Maximilian Köhl"

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

html_theme = "sphinx_book_theme"

html_title = "Momba"
html_logo = "_static/images/logo_with_text.svg"

html_static_path = ["_static"]
html_css_files = ["css/fix-jupyter-cells.css"]

EXTRA_FOOTER = """
<a href="https://www.uni-saarland.de/impressum.html">Imprint</a>
| Hosted by <a href="https://github.com">GitHub</a>
– <a href="https://docs.github.com/en/github/site-policy/github-privacy-statement#github-pages">Privacy Policy</a>.
"""

html_theme_options: t.Dict[str, t.Any] = {
    "repository_url": "https://github.com/koehlma/momba",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "logo_only": True,
    "extra_footer": EXTRA_FOOTER,
    "extra_navbar" : "",
    "use_repository_button": True,
    "use_issues_button": True,
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
