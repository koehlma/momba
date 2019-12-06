# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.model import expressions, types
from momba.model.expressions import const, ite, var

import pytest


def test_basic_inferences():
    ctx = expressions.TypeContext()
    ctx.declare('x', types.BOOL)

    expr = var('x') & var('y')

    with pytest.raises(expressions.UndeclaredVariableError):
        expr.infer_type(ctx)

    ctx.declare('y', types.BOOL)

    assert expr.infer_type(ctx) == types.BOOL

    ctx.declare('z', types.INT)

    with pytest.raises(expressions.InvalidTypeError):
        (var('x') & var('z')).infer_type(ctx)

    with pytest.raises(expressions.InvalidTypeError):
        ite(var('z'), var('x'), var('y')).infer_type(ctx)

    with pytest.raises(expressions.InvalidTypeError):
        ite(var('x'), var('z'), var('y')).infer_type(ctx)

    assert ite(var('x'), var('z'), const(3)).infer_type(ctx) == types.INT
