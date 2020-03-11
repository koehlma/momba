import os
import sys
import subprocess

from pygments.lexer import RegexLexer
from pygments import token

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import momba  # noqa

from momba import moml  # noqa


class MomlLexer(RegexLexer):  # type:ignore
    name = "Moml"
    aliases = ["moml"]
    filenames = ["*.moml"]

    tokens = {
        "root": [
            (r"\s+", token.Whitespace),
            (r"edge|transient|instance|variable|probability|network|location|model_type|from|"
             r"rate|automaton|guard|initial|invariant|constant|composition|synchronize|input|"
             r"action|enable|restrict|to|metadata", token.Keyword),
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
copyright = "2020, Dependable Systems and Software Group, Saarland University"
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
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

html_theme_options = {"display_version": True}

templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"

html_static_path = []  # type: ignore


def setup(app):  # type:ignore
    from sphinx.highlighting import lexers

    lexers["moml"] = MomlLexer()
    lexers["bnf"] = BNFLexer()
