# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>
#
# This code is partially copied from a private project.

from __future__ import annotations

import typing as t

import dataclasses
import enum
import re


class TokenType(enum.Enum):
    AND = r"and|∧"
    OR = r"or|∨"
    NOT = r"not|¬"

    DOUBLE_SLASH = r"//"
    SLASH = r"/"

    PLUS = r"\+"
    MINUS = r"-"
    STAR = r"\*"
    PRECENT = r"%"

    LEFT_PAR = r"\("
    RIGHT_PAR = r"\)"

    INTEGER = r"\d+"

    VARIABLE = r"\$\w+"
    IDENTIFIER = r"\w+"

    NEWLINE = r"\n"
    WHITESPACE = r"(\s(?<!\n))+"

    ERROR = r"."

    END_OF_FILE = r""


@dataclasses.dataclass(frozen=True)
class Token:
    token_type: TokenType
    text: str
    row: int
    start_column: int
    end_column: int


_lexer_regex = re.compile(
    r"|".join(f"(?P<{token_type.name}>{token_type.value})" for token_type in TokenType)
)


class LexerError(Exception):
    pass


def lex(code: str, *, row: int = 0, column: int = 0) -> t.Iterator[Token]:
    for match in _lexer_regex.finditer(code):
        assert isinstance(match.lastgroup, str)
        token_type = TokenType[match.lastgroup]
        text = match[0]
        if token_type is TokenType.ERROR:
            raise LexerError(f"unexpected character {text!r} at {row}:{column}")
        end_column = column + len(text)
        yield Token(token_type, text, row, column, end_column)
        if token_type is TokenType.NEWLINE:
            row += 1
            column = 0
        else:
            column = end_column
    yield Token(TokenType.END_OF_FILE, "", row, column, column)
