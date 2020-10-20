# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum
import re


KEYWORDS = {
    "model_type",
    "transient",
    "assign",
    "variable",
    "constant",
    "action",
    "comment",
    "parameter",
    "automaton",
    "network",
    "initial",
    "location",
    "invariant",
    "edge",
    "from",
    "to",
    "guard",
    "rate",
    "probability",
    "restrict",
    "instance",
    "input",
    "enable",
    "composition",
    "synchronize",
    "metadata",
    "true",
    "false",
}

PRIMITIVE_TYPES = {"bool", "int", "real", "clock", "continuous"}


class TokenType(enum.Enum):
    regex: str
    pseudo: bool

    COMMENT = r"#.*|\(\*(.|\s)*?\*\)"

    READ = r"\?>"
    WRITE = r"<!"

    ASSIGN = r":="

    SLASH_SLASH = r"//"

    NAMED_REAL = r"real\[\w+\]"

    TAU = r"τ"

    ARROW = r"→|->"

    COMP_LE = r"≤|<="
    COMP_GE = r"≥|>="
    COMP_LT = r"<"
    COMP_GT = r">"

    LOGIC_AND = r"∧|and"
    LOGIC_OR = r"∨|or"
    LOGIC_XOR = r"⊕|xor"
    LOGIC_IMPLIES = r"⇒|==>"
    LOGIC_EQUIV = r"⇔|<=>"
    LOGIC_NOT = r"¬|not"

    COMP_EQ = r"==|="
    COMP_NEQ = r"≠|!="

    POWER = r"\*\*"

    PLUS = r"\+"
    STAR = r"\*"
    PERCENTAGE = r"%"
    SLASH = r"/"
    QUESTIONMARK = r"\?"
    PIPE = r"\|"

    COMMA = r","
    COLLON = r":"

    LEFT_BRACE = r"\{"
    RIGHT_BRACE = r"\}"

    FILTER_LEFT = r"\(\{"
    FILTER_RIGHT = r"\}\)"

    LEFT_PAR = r"\("
    RIGHT_PAR = r"\)"

    LEFT_BRACKET = r"\["
    RIGHT_BRACKET = r"\]"

    STRING = r"\"(?P<string>[^\"]*)\""

    REAL = r"\d+\.\d+"
    INTEGER = r"\d+"

    MINUS = r"-"

    MACRO = r"\$(?P<macro_name>\w+)"

    IDENTIFIER = r"\w+"

    WHITESPACE = r"\s+"

    ERROR = r"."

    KEYWORD = r"KEYWORD", True

    INDENT = r"INDENT", True
    DEDENT = r"DEDENT", True

    END_OF_FILE = r"END_OF_FILE", True

    def __init__(self, regex: str, pseudo: bool = False) -> None:
        self.regex = regex
        self.pseudo = pseudo


@d.dataclass(frozen=True)
class Token:
    token_type: TokenType

    text: str
    match: t.Match[str]

    start_row: int
    start_column: int

    end_row: int
    end_column: int


_lexer_regex = re.compile(
    r"|".join(
        f"(?P<{token_type.name}>{token_type.value})"
        for token_type in TokenType
        if not token_type.pseudo
    )
)


_OPEN_PARENTHESIS = {TokenType.LEFT_BRACE, TokenType.LEFT_BRACKET, TokenType.LEFT_PAR}
_CLOSE_PARENTHESIS = {
    TokenType.RIGHT_BRACE,
    TokenType.RIGHT_BRACKET,
    TokenType.RIGHT_PAR,
}


class LexerError(Exception):
    pass


def lex(code: str, *, row: int = 0, column: int = 0) -> t.Iterator[Token]:
    indents: t.List[int] = [0]
    parenthesis_level = 0
    end_row = row
    end_column = column
    for match in _lexer_regex.finditer(code):
        assert isinstance(match.lastgroup, str)
        token_type = TokenType[match.lastgroup]
        text = match[0]
        if token_type is TokenType.IDENTIFIER and text in KEYWORDS:
            token_type = TokenType.KEYWORD
        if token_type is TokenType.ERROR:
            raise LexerError(f"unexpected character {text!r} at {row}:{column}")
        if "\n" in text:
            end_row = row + text.count("\n")
            end_column = len(text.rpartition("\n")[2])
            if token_type is TokenType.WHITESPACE and parenthesis_level == 0:
                if end_column > indents[-1]:
                    indents.append(end_column)
                    yield Token(
                        TokenType.INDENT,
                        "",
                        match,
                        end_row,
                        end_column,
                        end_row,
                        end_column,
                    )
                else:
                    while end_column < indents[-1]:
                        indents.pop()
                        yield Token(
                            TokenType.DEDENT,
                            "",
                            match,
                            end_row,
                            end_column,
                            end_row,
                            end_column,
                        )
                    if indents[-1] != end_column:
                        raise LexerError("inconsistent indentation")
        else:
            end_column = column + len(text)
        if token_type in _OPEN_PARENTHESIS:
            parenthesis_level += 1
        elif token_type in _CLOSE_PARENTHESIS:
            parenthesis_level -= 1
        yield Token(token_type, text, match, row, column, end_row, end_column)
        row = end_row
        column = end_column
    for _ in indents[1:]:
        yield Token(
            TokenType.DEDENT,
            "",
            match,
            end_row,
            end_column,
            end_row,
            end_column,
        )
    yield Token(
        TokenType.END_OF_FILE,
        "",
        match,
        end_row,
        end_column,
        end_row,
        end_column,
    )


if __name__ == "__main__":
    print("|".join(sorted(KEYWORDS)))
