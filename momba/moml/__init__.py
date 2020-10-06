# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

from .. import model

from ..model import expressions

from . import lexer, parser


def parse_type(source: str) -> model.Type:
    return parser.parse_type(parser.TokenStream(source))


def parse_expression(source: str) -> model.Expression:
    return parser.parse_expression(parser.TokenStream(source))


def parse(source: str, *, ctx: t.Optional[model.Context] = None) -> model.Context:
    return parser.parse_moml(parser.TokenStream(source), ctx=ctx)


def inline_expression(
    source: str, **macros: expressions.MaybeExpression
) -> model.Expression:
    return parser.parse_expression(
        parser.TokenStream(source),
        environment=parser.Environment(
            {name: expressions.convert(value) for name, value in macros.items()}
        ),
    )


__all__ = ["lexer", "parser", "parse_type", "parse_expression", "parse"]
