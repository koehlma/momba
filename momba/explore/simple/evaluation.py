# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import dataclasses
import functools
import math
import warnings

from ... import model
from ...model import context, expressions, operators, types
from ...model.expressions import BinaryExpression
from ...utils import checks

from .. import errors


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
    number: t.Union[int, float]


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
    parent: t.Optional[Valuation]

    _values: t.Dict[model.Identifier, Value]

    def __init__(self, parent: t.Optional[Valuation] = None) -> None:
        self.parent = parent
        self._values = {}

    def load(self, identifier: model.Identifier) -> Value:
        try:
            return self._values[identifier]
        except KeyError:
            if self.parent:
                return self.parent.load(identifier)
            else:
                raise errors.UnboundIdentifierError(
                    f"identifier {identifier} is not bound to a value"
                )

    def store(self, identifier: model.Identifier, value: Value) -> None:
        self._values[identifier] = value


@dataclasses.dataclass(frozen=True)
class EvaluationContext:
    """ A context to evaluate an expression in. """

    valuation: Valuation
    scope: context.Scope

    def load(self, identifier: context.Identifier) -> Value:
        return self.valuation.load(identifier)


@functools.singledispatch
def evaluate(expr: model.Expression, ctx: EvaluationContext) -> Value:
    raise NotImplementedError(
        f"no evaluation function has been implemented for expression {expr}"
    )


@evaluate.register
def _eval_constant(expr: expressions.Constant, ctx: EvaluationContext) -> Value:
    return _eval_model_value(expr.value, ctx)


@functools.singledispatch
def _eval_model_value(value: model.Value, ctx: EvaluationContext) -> Value:
    raise NotImplementedError(
        f"no evaluation function has been implemented for model value {value}"
    )


@_eval_model_value.register
def _model_value_bool(
    value: model.values.BooleanValue, ctx: EvaluationContext
) -> Value:
    return Bool(value.boolean)


@_eval_model_value.register
def _model_value_integer(
    value: model.values.IntegerValue, ctx: EvaluationContext
) -> Value:
    return Integer(value.integer)


@_eval_model_value.register
def _model_value_real(value: model.values.RealValue, ctx: EvaluationContext) -> Value:
    if not isinstance(value.real, float):
        warnings.warn(
            f"imprecise conversion of real, reals are approximated by IEEE 754 doubles",
            category=ImprecisionWarning,
        )
    return Real(value.as_float)


@evaluate.register
def _eval_identifier(expr: expressions.Identifier, ctx: EvaluationContext) -> Value:
    return ctx.load(expr.identifier)


@evaluate.register
def _eval_conditional(expr: expressions.Conditional, ctx: EvaluationContext) -> Value:
    condition = evaluate(expr.condition, ctx)
    assert isinstance(
        condition, Bool
    ), f"type checking should guarantee that the value is a bool"
    if condition.boolean:
        return evaluate(expr.consequence, ctx)
    else:
        return evaluate(expr.alternative, ctx)


@evaluate.register(expressions.Sample)
@evaluate.register(expressions.Selection)
@evaluate.register(expressions.Derivative)
def _eval_unsupported(expr: expressions.Expression, ctx: EvaluationContext) -> Value:
    raise errors.UnsupportedExpressionError(
        f"expression {expr} cannot be evaluated with the simple state-space explorer"
    )


@evaluate.register
def _eval_not(expr: expressions.Not, ctx: EvaluationContext) -> Value:
    operand = evaluate(expr.operand, ctx)
    assert isinstance(
        operand, Bool
    ), f"type checking should guarantee that the value is a boolean"
    return Bool(not operand.boolean)


@evaluate.register
def _eval_round(expr: expressions.Round, ctx: EvaluationContext) -> Value:
    operand = evaluate(expr.operand, ctx)
    assert isinstance(
        operand, Numeric
    ), f"type checking should guarantee that the value is a boolean"
    if expr.operator is operators.Round.FLOOR:
        return Integer(math.floor(operand.number))
    else:
        assert expr.operator is operators.Round.CEIL
        return Integer(math.ceil(operand.number))


_BinaryOperator = t.Callable[[BinaryExpression, Value, Value, EvaluationContext], Value]


@evaluate.register
def _eval_binary(expr: BinaryExpression, ctx: EvaluationContext) -> Value:
    left = evaluate(expr.left, ctx)
    right = evaluate(expr.right, ctx)
    return _BINARY_OPERATOR_MAP[expr.operator](expr, left, right, ctx)


def _eval_equality(
    expr: BinaryExpression, left: Value, right: Value, ctx: EvaluationContext
) -> Value:
    assert expr.operator in {
        operators.EqualityOperator.EQ,
        operators.EqualityOperator.NEQ,
    }, f"this function is meant to evaluate equalities and nothing else"
    result: bool
    if isinstance(left, Numeric) and isinstance(right, Numeric):
        result = left.number == right.number
    else:
        result = left == right
    if expr.operator is operators.EqualityOperator.EQ:
        return Bool(result)
    else:
        return Bool(not result)


_BINARY_OPERATOR_MAP: t.Dict[operators.BinaryOperator, _BinaryOperator] = {
    operators.EqualityOperator.EQ: _eval_equality,
    operators.EqualityOperator.NEQ: _eval_equality,
}


_Number = t.Union[float, int]
_ArithmeticFunction = t.Callable[[_Number, _Number], _Number]
_BooleanFunction = t.Callable[[bool, bool], bool]
_ComparisonFunction = t.Callable[[_Number, _Number], bool]


def _register_arithmetic_function(
    operator: operators.ArithmeticOperator, function: _ArithmeticFunction
) -> None:
    def implementation(
        expr: BinaryExpression, left: Value, right: Value, ctx: EvaluationContext
    ) -> Value:
        assert isinstance(left, (Integer, Real)) and isinstance(
            right, (Integer, Real)
        ), f"type checking should guarantee that the values are either integers or reals"
        result = function(left.number, right.number)
        if ctx.scope.get_type(expr) == types.INT:
            assert isinstance(
                result, int
            ), f"type checking should guarantee that the result is an integer"
            return Integer(result)
        else:
            return Real(float(result))

    implementation.__name__ = f"_eval_arithmetic_{operator.name.lower()}"
    _BINARY_OPERATOR_MAP[operator] = implementation


def _register_boolean_function(
    operator: operators.BooleanOperator, function: _BooleanFunction
) -> None:
    def implementation(
        expr: BinaryExpression, left: Value, right: Value, ctx: EvaluationContext
    ) -> Value:
        assert isinstance(left, Bool) and isinstance(
            right, Bool
        ), f"type checking should guarantee that the values are boolean"
        return Bool(function(left.boolean, right.boolean))

    implementation.__name__ = f"_eval_boolean_{operator.name.lower()}"
    _BINARY_OPERATOR_MAP[operator] = implementation


def _register_comparison_function(
    operator: operators.ComparisonOperator, function: _ComparisonFunction
) -> None:
    def implementation(
        expr: BinaryExpression, left: Value, right: Value, ctx: EvaluationContext
    ) -> Value:
        assert isinstance(left, (Integer, Real)) and isinstance(
            right, (Integer, Real)
        ), f"type checking should guarantee that the values are either integers or reals"
        return Bool(function(left.number, right.number))

    implementation.__name__ = f"_eval_comparison_{operator.name.lower()}"
    _BINARY_OPERATOR_MAP[operator] = implementation


_register_arithmetic_function(
    operators.ArithmeticOperator.ADD, lambda left, right: left + right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.SUB, lambda left, right: left - right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.MUL, lambda left, right: left * right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.MOD, lambda left, right: left % abs(right)
)
_register_arithmetic_function(
    operators.ArithmeticOperator.MIN, lambda left, right: min(left, right)
)
_register_arithmetic_function(
    operators.ArithmeticOperator.MAX, lambda left, right: max(left, right)
)
_register_arithmetic_function(
    operators.ArithmeticOperator.FLOOR_DIV, lambda left, right: left // right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.REAL_DIV, lambda left, right: left / right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.POW, lambda left, right: left ** right
)
_register_arithmetic_function(
    operators.ArithmeticOperator.LOG, lambda left, right: math.log(left, right)
)

_register_boolean_function(
    operators.BooleanOperator.AND, lambda left, right: left and right
)
_register_boolean_function(
    operators.BooleanOperator.OR, lambda left, right: left or right
)
_register_boolean_function(
    operators.BooleanOperator.XOR,
    lambda left, right: (not left and right) or (left and not right),
)
_register_boolean_function(
    operators.BooleanOperator.IMPLY, lambda left, right: not left or right
)
_register_boolean_function(
    operators.BooleanOperator.EQUIV, lambda left, right: left is right
)

_register_comparison_function(
    operators.ComparisonOperator.LT, lambda left, right: left < right
)
_register_comparison_function(
    operators.ComparisonOperator.LE, lambda left, right: left <= right
)
_register_comparison_function(
    operators.ComparisonOperator.GE, lambda left, right: left >= right
)
_register_comparison_function(
    operators.ComparisonOperator.GT, lambda left, right: left > right
)


# ensure that all operators have been defined
assert all(
    operator in _BINARY_OPERATOR_MAP for operator in operators.ArithmeticOperator
)
assert all(operator in _BINARY_OPERATOR_MAP for operator in operators.BooleanOperator)
assert all(
    operator in _BINARY_OPERATOR_MAP for operator in operators.ComparisonOperator
)


checks.check_singledispatch(
    evaluate,
    expressions.Expression,
    ignore={expressions.BinaryExpression, expressions.UnaryExpression},
)
