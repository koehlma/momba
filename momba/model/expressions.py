# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import decimal
import enum
import fractions
import math
import warnings

from . import errors, operators, properties, types

if t.TYPE_CHECKING:
    from . import context, distributions


class Expression(properties.Property, abc.ABC):
    @property
    @abc.abstractmethod
    def children(self) -> t.Sequence[Expression]:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_constant_in(self, scope: context.Scope) -> bool:
        """
        Returns `True` only if the expression has a constant value in the given scope.

        Arguments:
            scope: The scope to use.
        """
        raise NotImplementedError()

    def traverse(self) -> t.Iterator[Expression]:
        yield self
        for child in self.children:
            yield from child.traverse()

    @property
    def subexpressions(self) -> t.AbstractSet[Expression]:
        return frozenset(self.traverse())

    @property
    def is_sampling_free(self) -> bool:
        return all(not isinstance(e, Sample) for e in self.traverse())

    @property
    def used_names(self) -> t.AbstractSet[Name]:
        return frozenset(child for child in self.traverse() if isinstance(child, Name))


class _Leaf(Expression):
    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


class Constant(_Leaf, abc.ABC):
    def is_constant_in(self, scope: context.Scope) -> bool:
        return True


@d.dataclass(frozen=True)
class BooleanConstant(Constant):
    boolean: bool

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.BOOL


class NumericConstant(Constant, abc.ABC):
    @property
    @abc.abstractmethod
    def as_float(self) -> float:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def as_fraction(self) -> fractions.Fraction:
        raise NotImplementedError()


TRUE: Expression = BooleanConstant(True)
FALSE: Expression = BooleanConstant(False)


@d.dataclass(frozen=True)
class IntegerConstant(NumericConstant):
    integer: int

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.INT

    @property
    def as_float(self) -> float:
        warnings.warn(
            "converting an integer constant to a float may result in a loss of precision"
        )
        return float(self.integer)

    @property
    def as_fraction(self) -> fractions.Fraction:
        return fractions.Fraction(self.integer)


_NAMED_REAL_MAP: t.Dict[str, NamedReal] = {}


class NamedReal(enum.Enum):
    PI = "π", math.pi
    E = "e", math.e

    symbol: str
    float_value: float

    def __init__(self, symbol: str, float_value: float) -> None:
        self.symbol = symbol
        self.float_value = float_value
        _NAMED_REAL_MAP[symbol] = self


Real = t.Union[NamedReal, fractions.Fraction]


@d.dataclass(frozen=True)
class RealConstant(NumericConstant):
    real: Real

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    @property
    def as_float(self) -> float:
        warnings.warn(
            "converting a real constant to a float may result in a loss of precision"
        )
        if isinstance(self.real, NamedReal):
            return self.real.float_value
        return float(self.real)

    @property
    def as_fraction(self) -> fractions.Fraction:
        if isinstance(self.real, fractions.Fraction):
            return self.real
        else:
            warnings.warn(
                "converting a named real constant do a fraction does result in a loss of precision"
            )
            return fractions.Fraction(self.real.float_value)


@d.dataclass(frozen=True)
class Name(_Leaf):
    identifier: str

    def is_constant_in(self, scope: context.Scope) -> bool:
        return scope.lookup(self.identifier).is_constant_in(scope)

    def infer_type(self, scope: context.Scope) -> types.Type:
        return scope.lookup(self.identifier).typ


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class BinaryExpression(Expression):
    operator: operators.BinaryOperator
    left: Expression
    right: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.left, self.right

    def is_constant_in(self, scope: context.Scope) -> bool:
        return self.left.is_constant_in(scope) and self.right.is_constant_in(scope)

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


class Boolean(BinaryExpression):
    operator: operators.BooleanOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if left_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {left_type}")
        right_type = scope.get_type(self.right)
        if right_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {right_type}")
        return types.BOOL


_REAL_RESULT_OPERATORS = {
    operators.ArithmeticBinaryOperator.REAL_DIV,
    operators.ArithmeticBinaryOperator.LOG,
    operators.ArithmeticBinaryOperator.POW,
}


class ArithmeticBinary(BinaryExpression):
    operator: operators.ArithmeticBinaryOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        if not left_type.is_numeric or not right_type.is_numeric:
            raise errors.InvalidTypeError(
                "operands of arithmetic expressions must have a numeric type"
            )
        is_int = (
            types.INT.is_assignable_from(left_type)
            and types.INT.is_assignable_from(right_type)
            and self.operator not in _REAL_RESULT_OPERATORS
        )
        if is_int:
            return types.INT
        else:
            return types.REAL


class Equality(BinaryExpression):
    operator: operators.EqualityOperator

    def get_common_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        if left_type.is_assignable_from(right_type):
            return left_type
        elif right_type.is_assignable_from(left_type):
            return right_type
        raise AssertionError(
            "type-inference should ensure that some of the above is true"
        )

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        left_assignable_right = left_type.is_assignable_from(right_type)
        right_assignable_left = right_type.is_assignable_from(left_type)
        if left_assignable_right or right_assignable_left:
            return types.BOOL
        # XXX: JANI specifies that “left and right must be assignable to some common type”
        # not sure whether this implementation reflects this specification
        raise errors.InvalidTypeError(
            "invalid combination of type for equality comparison"
        )


class Comparison(BinaryExpression):
    operator: operators.ComparisonOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if not left_type.is_numeric:
            raise errors.InvalidTypeError(f"expected numeric type but got {left_type}")
        right_type = scope.get_type(self.right)
        if not right_type.is_numeric:
            raise errors.InvalidTypeError(f"expected numeric type but got {right_type}")
        return types.BOOL


@d.dataclass(frozen=True)
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.condition, self.consequence, self.alternative

    def is_constant_in(self, scope: context.Scope) -> bool:
        return (
            self.condition.is_constant_in(scope)
            and self.consequence.is_constant_in(scope)
            and self.alternative.is_constant_in(scope)
        )

    def infer_type(self, scope: context.Scope) -> types.Type:
        condition_type = scope.get_type(self.condition)
        if condition_type != types.BOOL:
            raise errors.InvalidTypeError(
                f"expected `types.BOOL` but got `{condition_type}`"
            )
        consequence_type = scope.get_type(self.consequence)
        alternative_type = scope.get_type(self.alternative)
        if consequence_type.is_assignable_from(alternative_type):
            return consequence_type
        elif alternative_type.is_assignable_from(consequence_type):
            return alternative_type
        else:
            raise errors.InvalidTypeError(
                "invalid combination of consequence and alternative types"
            )


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class UnaryExpression(Expression):
    operator: operators.UnaryOperator
    operand: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return (self.operand,)

    def is_constant_in(self, scope: context.Scope) -> bool:
        return self.operand.is_constant_in(scope)

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


class ArithmeticUnary(UnaryExpression):
    operator: operators.ArithmeticUnaryOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if not operand_type.is_numeric:
            raise errors.InvalidTypeError(
                f"expected a numeric type but got {operand_type}"
            )
        return self.operator.infer_result_type(operand_type)


class Not(UnaryExpression):
    operator: operators.NotOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if operand_type != types.BOOL:
            raise errors.InvalidTypeError(
                f"expected `types.BOOL` but got {operand_type}"
            )
        return types.BOOL


@d.dataclass(frozen=True)
class Sample(Expression):
    distribution: distributions.DistributionType
    arguments: t.Sequence[Expression]

    def __post_init__(self) -> None:
        if len(self.arguments) != self.distribution.arity:
            raise errors.InvalidTypeError(
                f"distribution {self.distribution} requires {self.distribution.arity} "
                f"arguments but {len(self.arguments)} were given"
            )

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.arguments

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    def infer_type(self, scope: context.Scope) -> types.Type:
        # we already know that the arity of the parameters and arguments match
        for argument, parameter_type in zip(
            self.arguments, self.distribution.parameter_types
        ):
            argument_type = scope.get_type(argument)
            if not parameter_type.is_assignable_from(argument_type):
                raise errors.InvalidTypeError(
                    f"parameter type `{parameter_type}` is not assignable "
                    f"from argument type `{argument_type}`"
                )
        return self.distribution.result_type


# requires JANI extension `nondet-selection`
@d.dataclass(frozen=True)
class Selection(Expression):
    name: str
    condition: Expression

    def infer_type(self, scope: context.Scope) -> types.Type:
        condition_type = scope.get_type(self.condition)
        if condition_type != types.BOOL:
            raise errors.InvalidTypeError("condition must have type `types.BOOL`")
        declaration = scope.lookup(self.name)
        assert isinstance(declaration, context.VariableDeclaration)
        return declaration.typ

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    @property
    def children(self) -> t.Sequence[Expression]:
        return (self.condition,)


@d.dataclass(frozen=True)
class Derivative(Expression):
    identifier: str

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


RealValue = t.Union[t.Literal["π", "e"], float, fractions.Fraction, decimal.Decimal]
NumericValue = t.Union[int, RealValue]
Value = t.Union[bool, NumericValue]

ValueOrExpression = t.Union[Expression, Value]


class ConversionError(ValueError):
    pass


def ensure_expr(value_or_expression: ValueOrExpression) -> Expression:
    if isinstance(value_or_expression, Expression):
        return value_or_expression
    elif isinstance(value_or_expression, bool):
        return BooleanConstant(value_or_expression)
    elif isinstance(value_or_expression, int):
        return IntegerConstant(value_or_expression)
    elif isinstance(value_or_expression, (float, fractions.Fraction, decimal.Decimal)):
        return RealConstant(fractions.Fraction(value_or_expression))
    elif isinstance(value_or_expression, str):
        try:
            return RealConstant(_NAMED_REAL_MAP[value_or_expression])
        except KeyError:
            pass
    raise ConversionError(
        f"unable to convert Python value {value_or_expression!r} to expression"
    )


def ite(
    condition: ValueOrExpression,
    consequence: ValueOrExpression,
    alternative: ValueOrExpression,
) -> Expression:
    return Conditional(
        ensure_expr(condition), ensure_expr(consequence), ensure_expr(alternative)
    )


BinaryConstructor = t.Callable[[ValueOrExpression, ValueOrExpression], Expression]


def _boolean_binary_expression(
    operator: operators.BooleanOperator, expressions: t.Sequence[ValueOrExpression]
) -> Expression:
    if len(expressions) == 1:
        return ensure_expr(expressions[0])
    result = Boolean(
        operator,
        ensure_expr(expressions[0]),
        ensure_expr(expressions[1]),
    )
    for operand in expressions[2:]:
        result = Boolean(operator, result, ensure_expr(operand))
    return result


def logic_or(*expressions: ValueOrExpression) -> Expression:
    return _boolean_binary_expression(operators.BooleanOperator.OR, expressions)


def logic_and(*expressions: ValueOrExpression) -> Expression:
    return _boolean_binary_expression(operators.BooleanOperator.AND, expressions)


def logic_xor(*expressions: ValueOrExpression) -> Expression:
    return _boolean_binary_expression(operators.BooleanOperator.XOR, expressions)


def logic_implies(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Boolean(
        operators.BooleanOperator.IMPLY, ensure_expr(left), ensure_expr(right)
    )


def logic_equiv(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Boolean(
        operators.BooleanOperator.EQUIV, ensure_expr(left), ensure_expr(right)
    )


def equals(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Equality(
        operators.EqualityOperator.EQ, ensure_expr(left), ensure_expr(right)
    )


def not_equals(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Equality(
        operators.EqualityOperator.NEQ, ensure_expr(left), ensure_expr(right)
    )


def less_than(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Comparison(
        operators.ComparisonOperator.LT, ensure_expr(left), ensure_expr(right)
    )


def less_or_equal_than(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Comparison(
        operators.ComparisonOperator.LE, ensure_expr(left), ensure_expr(right)
    )


def greater_than(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return Comparison(
        operators.ComparisonOperator.GE, ensure_expr(left), ensure_expr(right)
    )


def greater_or_equal_than(
    left: ValueOrExpression, right: ValueOrExpression
) -> Expression:
    return Comparison(
        operators.ComparisonOperator.GT, ensure_expr(left), ensure_expr(right)
    )


def add(left: ValueOrExpression, right: ValueOrExpression) -> Expression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.ADD, ensure_expr(left), ensure_expr(right)
    )


def sub(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.SUB, ensure_expr(left), ensure_expr(right)
    )


def mul(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MUL, ensure_expr(left), ensure_expr(right)
    )


def mod(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MOD, ensure_expr(left), ensure_expr(right)
    )


def real_div(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.REAL_DIV,
        ensure_expr(left),
        ensure_expr(right),
    )


def log(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.LOG, ensure_expr(left), ensure_expr(right)
    )


def power(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.POW, ensure_expr(left), ensure_expr(right)
    )


def minimum(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MIN, ensure_expr(left), ensure_expr(right)
    )


def maximum(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.MAX, ensure_expr(left), ensure_expr(right)
    )


def floor_div(left: ValueOrExpression, right: ValueOrExpression) -> BinaryExpression:
    return ArithmeticBinary(
        operators.ArithmeticBinaryOperator.FLOOR_DIV,
        ensure_expr(left),
        ensure_expr(right),
    )


UnaryConstructor = t.Callable[[ValueOrExpression], Expression]


def logic_not(operand: ValueOrExpression) -> Expression:
    return Not(operators.NotOperator.NOT, ensure_expr(operand))


def floor(operand: ValueOrExpression) -> Expression:
    return ArithmeticUnary(
        operators.ArithmeticUnaryOperator.FLOOR, ensure_expr(operand)
    )


def ceil(operand: ValueOrExpression) -> Expression:
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.CEIL, ensure_expr(operand))


def absolute(operand: ValueOrExpression) -> Expression:
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.ABS, ensure_expr(operand))


def sgn(operand: ValueOrExpression) -> Expression:
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.SGN, ensure_expr(operand))


def trunc(operand: ValueOrExpression) -> Expression:
    return ArithmeticUnary(operators.ArithmeticUnaryOperator.TRC, ensure_expr(operand))


def name(identifier: str) -> Name:
    return Name(identifier)
