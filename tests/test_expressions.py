# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.model import context, errors, types
from momba.model.expressions import const, ite, var

import pytest


def test_basic_inferences():
    ctx = context.Context()

    scope = ctx.new_scope()
    scope.declare_variable('x', types.BOOL)

    expr = var('x') & var('y')

    with pytest.raises(errors.UnboundIdentifierError):
        scope.get_type(expr)

    scope.declare_variable('y', types.BOOL)

    assert scope.get_type(expr) == types.BOOL

    scope.declare_variable('z', types.INT)

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(var('x') & var('z'))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(var('z'), var('x'), var('y')))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(var('x'), var('z'), var('y')))

    assert scope.get_type(ite(var('x'), var('z'), const(3))) == types.INT
