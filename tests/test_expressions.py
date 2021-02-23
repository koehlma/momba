# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

from momba import model
from momba.model import context, errors, types
from momba.model.expressions import ensure_expr, ite, name, logic_and

import pytest


def test_basic_inferences() -> None:
    ctx = context.Context(model.ModelType.SHA)

    scope = ctx.global_scope.create_child_scope()
    scope.declare_variable("x", types.BOOL)

    expr = logic_and(name("x"), name("y"))

    with pytest.raises(errors.NotFoundError):
        scope.get_type(expr)

    scope.declare_variable("y", types.BOOL)

    assert scope.get_type(expr) == types.BOOL

    scope.declare_variable("z", types.INT)

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(logic_and(name("x"), name("z")))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(name("z"), name("x"), name("y")))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(name("x"), name("z"), name("y")))

    assert scope.get_type(ite(name("x"), name("z"), ensure_expr(3))) == types.INT
