# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

from .. import model

from ..model import expressions

from . import parser


def parse_type(source: str) -> model.Type:
    return parser.parse_type(parser.TokenStream(source))


def parse_expression(source: str) -> model.Expression:
    return parser.parse_expression(parser.TokenStream(source))


def parse(source: str, *, ctx: t.Optional[model.Context] = None) -> model.Context:
    return parser.parse_moml(parser.TokenStream(source), ctx=ctx)


def expr(source: str, **macros: expressions.ValueOrExpression) -> model.Expression:
    """Parses an expression in MOML syntax with macros."""
    return parser.parse_expression(
        parser.TokenStream(source.strip()),
        environment=parser.Environment(
            {name: expressions.ensure_expr(value) for name, value in macros.items()}
        ),
    )


def prop(source: str, **macros: expressions.ValueOrExpression) -> model.Expression:
    """Parses a property in MOML syntax with macros."""
    return parser.parse_property(
        parser.TokenStream(source.strip()),
        environment=parser.Environment(
            {name: expressions.ensure_expr(value) for name, value in macros.items()}
        ),
    )
