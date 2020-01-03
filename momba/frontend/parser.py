# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>
#
# This code is partially copied from a private project.

from __future__ import annotations

import typing as t

from . import lexer
from ..model import expressions


Variables = t.Mapping[str, expressions.Expression]


_IGNORE = {lexer.TokenType.WHITESPACE, lexer.TokenType.NEWLINE}


class Parser:
    code: str
    iterator: t.Iterator[lexer.Token]

    variables: Variables

    _current_token: t.Optional[lexer.Token]
    _next_token: t.Optional[lexer.Token]

    def __init__(self, code: str, *, variables: t.Optional[Variables]) -> None:
        self.code = code
        self.iterator = lexer.lex(code)
        self.variables = variables or {}
        self._current_token = self._forward()
        self._next_token = self._forward()

    def _forward(self) -> t.Optional[lexer.Token]:
        try:
            token = next(self.iterator)
            while token.token_type in _IGNORE:
                token = next(self.iterator)
            return token
        except StopIteration:
            return None
