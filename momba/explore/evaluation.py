# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import functools
import fractions

from ..model import context, expressions, operators, types


Number = t.Union[int, fractions.Fraction]


class Value:
    @property
    def as_int(self) -> int:
        assert isinstance(self, Integer), f"expected an integer value but found {self}"
        return self.integer

    @property
    def as_number(self) -> Number:
        assert isinstance(self, Numeric), f"expected numeric value but found {self}"
        return self.number

    @property
    def as_fraction(self) -> fractions.Fraction:
        assert isinstance(self, Numeric), f"expected a numeric value but found {self}"
        return fractions.Fraction(self.number)

    @property
    def as_bool(self) -> bool:
        assert isinstance(self, Boolean), f"expected a boolean value but found {self}"
        return self.boolean


class Numeric(Value):
    number: t.Union[int, fractions.Fraction]

    def __str__(self) -> str:
        return str(self.number)


@d.dataclass(frozen=True)
class Integer(Numeric):
    number: int

    @property
    def integer(self) -> int:
        return self.number


@d.dataclass(frozen=True)
class Real(Numeric):
    number: fractions.Fraction

    @property
    def real(self) -> fractions.Fraction:
        return self.number


@d.dataclass(frozen=True)
class Boolean(Value):
    boolean: bool

    def __str__(self) -> str:
        return "true" if self.boolean else "false"


TRUE = Boolean(True)
FALSE = Boolean(False)


# just a placeholder, arrays are currently not supported by Momba
@d.dataclass(frozen=True)
class Array(Value):
    elements: t.Tuple[Value, ...]


class UnboundNameError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"invalid access to unbound name {name!r}")
        self.name = name


@d.dataclass(eq=False)
class Environment:
    binding: t.Mapping[str, Value]
    scope: context.Scope
    parent: t.Optional[Environment] = None

    def lookup(self, name: str) -> Value:
        try:
            return self.binding[name]
        except KeyError:
            if self.parent is None:
                raise UnboundNameError(name)
            return self.parent.lookup(name)


@functools.singledispatch
def _evaluate(expression: expressions.Expression, environment: Environment) -> Value:
    raise NotImplementedError(
        f"evaluation function not implemented for {type(expression)}"
    )


@_evaluate.register
def _evaluate_integer_constant(
    expression: expressions.IntegerConstant, environment: Environment
) -> Value:
    return Integer(expression.integer)


@_evaluate.register
def _evaluate_real_constant(
    expression: expressions.RealConstant, environment: Environment
) -> Value:
    return Real(expression.as_fraction)


@_evaluate.register
def _evaluate_boolean_constant(
    expression: expressions.BooleanConstant, environment: Environment
) -> Value:
    return Boolean(expression.boolean)


@_evaluate.register
def _evaluate_identifier(
    expression: expressions.Name, environment: Environment
) -> Value:
    return environment.lookup(expression.identifier)


def _evaluate_binary(
    elementary_function: t.Callable[[Value, Value], Value],
    expression: expressions.BinaryExpression,
    environment: Environment,
) -> Value:
    return elementary_function(
        _evaluate(expression.left, environment),
        _evaluate(expression.right, environment),
    )


@_evaluate.register
def _evaluate_boolean(
    expression: expressions.Boolean, environment: Environment
) -> Value:
    def elementary_function(left: Value, right: Value) -> Value:
        return Boolean(expression.operator.native_function(left.as_bool, right.as_bool))

    return _evaluate_binary(elementary_function, expression, environment)


def _convert_numeric(number: operators.Number, typ: types.Type) -> Value:
    if typ == types.INT:
        assert isinstance(number, int), "computation should have returned an integer"
        return Integer(number)
    else:
        assert typ == types.REAL
        return Real(fractions.Fraction(number))


@_evaluate.register
def _evaluate_arithmetic(
    expression: expressions.ArithmeticBinary, environment: Environment
) -> Value:
    typ = environment.scope.get_type(expression)

    def elementary_function(left: Value, right: Value) -> Value:
        return _convert_numeric(
            expression.operator.native_function(left.as_number, right.as_number), typ
        )

    return _evaluate_binary(elementary_function, expression, environment)


@_evaluate.register
def _evaluate_equality(
    expression: expressions.Equality, environment: Environment
) -> Value:
    def elementary_function(left: Value, right: Value) -> Value:
        return Boolean(expression.operator.native_function(left, right))

    return _evaluate_binary(elementary_function, expression, environment)


@_evaluate.register
def _evaluate_comparison(
    expression: expressions.Comparison, environment: Environment
) -> Value:
    def elementary_function(left: Value, right: Value) -> Value:
        return Boolean(
            expression.operator.native_function(left.as_number, right.as_number)
        )

    return _evaluate_binary(elementary_function, expression, environment)


@_evaluate.register
def _evaluate_condition(
    expression: expressions.Conditional, environment: Environment
) -> Value:
    if evaluate(expression.condition, environment).as_bool:
        return _evaluate(expression.consequence, environment)
    else:
        return _evaluate(expression.alternative, environment)


def _evaluate_unary(
    elementary_function: t.Callable[[Value], Value],
    expression: expressions.UnaryExpression,
    environment: Environment,
) -> Value:
    return elementary_function(_evaluate(expression.operand, environment))


@_evaluate.register
def _evaluate_round(
    expression: expressions.ArithmeticUnary, environment: Environment
) -> Value:
    def elementary_function(operand: Value) -> Value:
        return Integer(expression.operator.native_function(operand.as_number))

    return _evaluate_unary(elementary_function, expression, environment)


@_evaluate.register
def _evaluate_not(expression: expressions.Not, environment: Environment) -> Value:
    def elementary_function(operand: Value) -> Value:
        return Boolean(not operand.as_bool)

    return _evaluate_unary(elementary_function, expression, environment)


def evaluate(expression: expressions.Expression, environment: Environment) -> Value:
    return _evaluate(expression, environment)
