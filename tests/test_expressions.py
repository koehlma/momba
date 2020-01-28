# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.model import context, errors, types
from momba.model.expressions import const, ite, identifier

import pytest


def test_basic_inferences() -> None:
    ctx = context.Context()

    scope = ctx.global_scope.create_child_scope()
    scope.declare_variable("x", types.BOOL)

    expr = identifier("x").land(identifier("y"))

    with pytest.raises(errors.UnboundIdentifierError):
        scope.get_type(expr)

    scope.declare_variable("y", types.BOOL)

    assert scope.get_type(expr) == types.BOOL

    scope.declare_variable("z", types.INT)

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(identifier("x").land(identifier("z")))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(identifier("z"), identifier("x"), identifier("y")))

    with pytest.raises(errors.InvalidTypeError):
        scope.get_type(ite(identifier("x"), identifier("z"), identifier("y")))

    assert scope.get_type(ite(identifier("x"), identifier("z"), const(3))) == types.INT
