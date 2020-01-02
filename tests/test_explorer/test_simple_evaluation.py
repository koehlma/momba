# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import math

from momba import model
from momba.explore.simple import evaluation

import pytest


def test_basic_evaluation():
    network = model.Network()

    automaton = network.create_automaton()
    automaton.scope.declare_variable("x", model.types.INT)

    x = model.expressions.var("x")

    valuation = evaluation.Valuation()
    valuation.store(x.identifier, evaluation.Integer(5))

    eval_ctx = evaluation.EvaluationContext(valuation=valuation, scope=automaton.scope)

    integer_addition = model.expressions.add(x, model.expressions.const(3))

    result = evaluation.evaluate(integer_addition, eval_ctx)
    assert isinstance(result, evaluation.Integer) and result.number == 8

    real_addition = model.expressions.add(x, model.expressions.const("π"))

    with pytest.warns(evaluation.ImprecisionWarning):
        result = evaluation.evaluate(real_addition, eval_ctx)
        assert isinstance(result, evaluation.Real) and math.isclose(
            result.number, 8.141592653589793
        )
