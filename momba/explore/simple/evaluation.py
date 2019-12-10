# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import functools
import typing
import warnings

from ... import model
from ...model import context, expressions, operators, types
from ...model.expressions import BinaryExpression


class Value(abc.ABC):
    @property
    @abc.abstractmethod
    def typ(self) -> model.Type:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class Bool(Value):
    boolean: bool
    typ: model.Type = dataclasses.field(default=types.BOOL, repr=False)


class Numeric(Value, abc.ABC):
    number: typing.Union[int, float]


@dataclasses.dataclass(frozen=True)
class Integer(Numeric):
    number: int
    typ: model.Type = dataclasses.field(default=types.INT, repr=False)


@dataclasses.dataclass(frozen=True)
class Real(Numeric):
    number: float
    typ: model.Type = dataclasses.field(default=types.REAL, repr=False)


class ImprecisionWarning(UserWarning):
    pass


class Valuation:
    _values: typing.Dict[model.Identifier, Value]

    def __init__(self) -> None:
        self._values = {}

    def load(self, identifier: model.Identifier) -> Value:
        return self._values[identifier]

    def store(self, identifier: model.Identifier, value: Value) -> None:
        self._values[identifier] = value


@dataclasses.dataclass(frozen=True)
class EvaluationContext:
    valuation: Valuation
    scope: context.Scope


@functools.singledispatch
def evaluate(expr: model.Expression, ctx: EvaluationContext) -> Value:
    raise NotImplementedError(
        f'no evaluation function has been implemented for expression {expr}'
    )


@evaluate.register
def _eval_constant(expr: expressions.Constant, ctx: EvaluationContext) -> Value:
    return _eval_model_value(expr.value, ctx)


@functools.singledispatch
def _eval_model_value(value: model.Value, ctx: EvaluationContext) -> Value:
    raise NotImplementedError(
        f'no evaluation function has been implemented for model value {value}'
    )


@_eval_model_value.register
def _model_value_bool(value: model.values.BooleanValue, ctx: EvaluationContext) -> Value:
    return Bool(value.boolean)


@_eval_model_value.register
def _model_value_integer(value: model.values.IntegerValue, ctx: EvaluationContext) -> Value:
    return Integer(value.integer)


@_eval_model_value.register
def _model_value_real(value: model.values.RealValue, ctx: EvaluationContext) -> Value:
    if not isinstance(value.real, float):
        warnings.warn(
            f'imprecise conversion of real, reals are approximated by IEEE 754 doubles',
            category=ImprecisionWarning
        )
    return Real(value.as_float)


@evaluate.register
def _eval_identifier(expr: expressions.Identifier, ctx: EvaluationContext) -> Value:
    return ctx.valuation.load(expr.identifier)


_BinaryOperator = typing.Callable[[BinaryExpression, Value, Value, EvaluationContext], Value]


@evaluate.register
def _eval_binary(expr: BinaryExpression, ctx: EvaluationContext) -> Value:
    left = evaluate(expr.left, ctx)
    right = evaluate(expr.right, ctx)
    return _BINARY_OPERATOR_MAP[expr.operator](expr, left, right, ctx)


def _add(expr: BinaryExpression, left: Value, right: Value, ctx: EvaluationContext) -> Value:
    # type checking guarantees that `left` and `right` are either `Real` or `Integer`
    assert isinstance(left, (Integer, Real))
    assert isinstance(right, (Integer, Real))
    if isinstance(left, Real) or isinstance(right, Real):
        warnings.warn(
            f'imprecise operation on reals, reals are approximated by IEEE 754 doubles',
            category=ImprecisionWarning
        )
    result = left.number + right.number
    if ctx.scope.get_type(expr) == types.INT:
        assert isinstance(result, int)
        return Integer(result)
    else:
        return Real(float(result))


_BINARY_OPERATOR_MAP: typing.Mapping[operators.BinaryOperator, _BinaryOperator] = {
    operators.ArithmeticOperator.ADD: _add
}
