# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import dataclasses
import enum
import math
import numbers

from . import context, errors, operators, properties, types

if t.TYPE_CHECKING:
    from . import distributions


class Expression(properties.Property, abc.ABC):
    @property
    @abc.abstractmethod
    def children(self) -> t.Sequence[Expression]:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_constant_in(self, scope: context.Scope) -> bool:
        """ Returns `True` only if the expression has a constant value in the given scope. """
        raise NotImplementedError()

    @abc.abstractmethod
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()

    @property
    def traverse(self) -> t.Iterable[Expression]:
        yield self
        for child in self.children:
            yield from child.traverse

    @property
    def subexpressions(self) -> t.AbstractSet[Expression]:
        return frozenset(self.traverse)

    @property
    def is_sampling_free(self) -> bool:
        return all(not isinstance(e, Sample) for e in self.traverse)

    def lor(self, other: Expression) -> Expression:
        if not isinstance(other, Expression):
            return NotImplemented
        return Boolean(operators.BooleanOperator.OR, self, other)

    def lnot(self) -> Expression:
        return lnot(self)

    def add(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.ADD, self, convert(other))

    def radd(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.ADD, convert(other), self)

    def sub(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.SUB, self, convert(other))

    def rsub(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.SUB, convert(other), self)

    def mul(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MUL, self, convert(other))

    def rmul(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MUL, convert(other), self)

    def mod(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MOD, self, convert(other))

    def rmod(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MOD, convert(other), self)

    def floordiv(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.FLOOR_DIV, convert(other), self)

    def eq(self, other: Expression) -> Expression:
        return Equality(operators.EqualityOperator.EQ, self, other)

    def neq(self, other: Expression) -> Expression:
        return Equality(operators.EqualityOperator.NEQ, self, other)

    def lt(self, other: Expression) -> Expression:
        return Comparison(operators.ComparisonOperator.LT, self, other)

    def le(self, other: Expression) -> Expression:
        return Comparison(operators.ComparisonOperator.LE, self, other)

    def ge(self, other: Expression) -> Expression:
        return Comparison(operators.ComparisonOperator.GE, self, other)

    def gt(self, other: Expression) -> Expression:
        return Comparison(operators.ComparisonOperator.GT, self, other)

    def land(self, other: Expression) -> Expression:
        return land(self, other)


class _Leaf(Expression):
    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


class Constant(_Leaf, abc.ABC):
    def is_constant_in(self, scope: context.Scope) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class BooleanConstant(Constant):
    boolean: bool

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.BOOL


class NumericConstant(Constant, abc.ABC):
    @property
    @abc.abstractmethod
    def as_float(self) -> float:
        raise NotImplementedError()


TRUE = BooleanConstant(True)
FALSE = BooleanConstant(False)


@dataclasses.dataclass(frozen=True)
class IntegerConstant(NumericConstant):
    integer: int

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.INT

    @property
    def as_float(self) -> float:
        # TODO: emit a warning to keep track of imprecisions
        return float(self.integer)


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


Real = t.Union[NamedReal, numbers.Real]


@dataclasses.dataclass(frozen=True)
class RealConstant(NumericConstant):
    real: Real

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    @property
    def as_float(self) -> float:
        # TODO: emit a warning to keep track of imprecisions
        if isinstance(self.real, NamedReal):
            return self.real.float_value
        return float(self.real)


@dataclasses.dataclass(frozen=True)
class Identifier(_Leaf):
    name: str

    def is_constant_in(self, scope: context.Scope) -> bool:
        return scope.lookup(self.name).is_constant_in(scope)

    def infer_type(self, scope: context.Scope) -> types.Type:
        return scope.lookup(self.name).typ


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@dataclasses.dataclass(frozen=True)
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


_REAL_OPERATORS = {
    operators.ArithmeticOperator.REAL_DIV,
    operators.ArithmeticOperator.LOG,
    operators.ArithmeticOperator.POW,
}


class Arithmetic(BinaryExpression):
    operator: operators.ArithmeticOperator

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
            and self.operator not in _REAL_OPERATORS
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
        # XXX: JANI specifies that “left and right must be assignable to some common type”
        if not left_type.is_assignable_from(
            right_type
        ) and not right_type.is_assignable_from(left_type):
            raise errors.InvalidTypeError(
                "invalid combination of type for equality comparison"
            )
        return types.BOOL


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


@dataclasses.dataclass(frozen=True)
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
@dataclasses.dataclass(frozen=True)
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


class Round(UnaryExpression):
    operator: operators.Round

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if not operand_type.is_numeric:
            raise errors.InvalidTypeError(
                f"expected a numeric type but got {operand_type}"
            )
        return types.INT


class Not(UnaryExpression):
    operator: operators.Not

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if operand_type != types.BOOL:
            raise errors.InvalidTypeError(
                f"expected `types.BOOL` but got {operand_type}"
            )
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class Sample(Expression):
    distribution: distributions.Distribution
    arguments: t.Sequence[Expression]

    def __post_init__(self) -> None:
        if len(self.arguments) != len(self.distribution.parameter_types):
            raise ValueError("parameter and arguments arity mismatch")

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
        return types.REAL


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class Derivative(Expression):
    identifier: str

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


def ite(
    condition: Expression, consequence: Expression, alternative: Expression
) -> Expression:
    return Conditional(condition, consequence, alternative)


PythonRealString = t.Literal["π", "e"]
PythonReal = t.Union[numbers.Real, float, PythonRealString, NamedReal]
PythonNumeric = t.Union[int, PythonReal]
PythonValue = t.Union[bool, PythonNumeric]


class ConversionError(ValueError):
    pass


def const(value: PythonValue) -> Constant:
    if isinstance(value, bool):
        return BooleanConstant(value)
    if isinstance(value, int):
        return IntegerConstant(value)
    elif isinstance(value, (numbers.Number, NamedReal)):
        return RealConstant(value)
    elif isinstance(value, str):
        return RealConstant(_NAMED_REAL_MAP[value])
    raise ConversionError(f"unable to convert Python value {value!r} to Momba value")


MaybeExpression = t.Union[PythonValue, Expression]


def convert(value: MaybeExpression) -> Expression:
    if isinstance(value, Expression):
        return value
    return const(value)


BinaryConstructor = t.Callable[[Expression, Expression], Expression]


def lor(*expressions: Expression) -> Expression:
    if len(expressions) == 2:
        return Boolean(operators.BooleanOperator.OR, expressions[0], expressions[1])
    result = convert(False)
    for disjunct in expressions:
        result = Boolean(operators.BooleanOperator.OR, result, disjunct)
    return result


def land(*expressions: Expression) -> Expression:
    if len(expressions) == 2:
        return Boolean(operators.BooleanOperator.AND, expressions[0], expressions[1])
    result = convert(True)
    for conjunct in expressions:
        result = Boolean(operators.BooleanOperator.AND, result, conjunct)
    return result


def xor(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.BooleanOperator.XOR, left, right)


def implies(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.BooleanOperator.IMPLY, left, right)


def equiv(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.BooleanOperator.EQUIV, left, right)


def eq(left: Expression, right: Expression) -> BinaryExpression:
    return Equality(operators.EqualityOperator.EQ, left, right)


def neq(left: Expression, right: Expression) -> BinaryExpression:
    return Equality(operators.EqualityOperator.NEQ, left, right)


def lt(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Comparison(operators.ComparisonOperator.LT, convert(left), convert(right))


def le(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Comparison(operators.ComparisonOperator.LE, convert(left), convert(right))


def ge(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Comparison(operators.ComparisonOperator.GE, convert(left), convert(right))


def gt(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Comparison(operators.ComparisonOperator.GT, convert(left), convert(right))


def add(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.ADD, convert(left), convert(right))


def sub(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.SUB, convert(left), convert(right))


def mul(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MUL, convert(left), convert(right))


def mod(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MOD, convert(left), convert(right))


def minimum(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MIN, convert(left), convert(right))


def maximum(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MAX, convert(left), convert(right))


def real_div(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(
        operators.ArithmeticOperator.REAL_DIV, convert(left), convert(right)
    )


def floor_div(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(
        operators.ArithmeticOperator.FLOOR_DIV, convert(left), convert(right)
    )


def power(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.POW, convert(left), convert(right))


def log(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.LOG, convert(left), convert(right))


UnaryConstructor = t.Callable[[Expression], Expression]


def lnot(operand: Expression) -> Expression:
    return Not(operators.Not.NOT, operand)


def floor(operand: MaybeExpression) -> Expression:
    return Round(operators.Round.FLOOR, convert(operand))


def ceil(operand: Expression) -> Expression:
    return Round(operators.Round.CEIL, convert(operand))


def normalize_xor(expr: Expression) -> Expression:
    assert isinstance(expr, Boolean) and expr.operator is operators.BooleanOperator.XOR
    return lor(land(lnot(expr.left), expr.right), land(expr.right, lnot(expr.left)))


def normalize_equiv(expr: Expression) -> Expression:
    assert (
        isinstance(expr, Boolean) and expr.operator is operators.BooleanOperator.EQUIV
    )
    return land(implies(expr.left, expr.right), implies(expr.right, expr.left))


def normalize_floor_div(expr: Expression) -> Expression:
    assert (
        isinstance(expr, Arithmetic)
        and expr.operator is operators.ArithmeticOperator.FLOOR_DIV
    )
    return floor(real_div(expr.left, expr.right))


logic_not = lnot
logic_or = lor
logic_and = land
logic_xor = xor
logic_implies = implies
logic_equiv = equiv


def logic_rimplies(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.BooleanOperator.IMPLY, right, left)


def identifier(name: str) -> Identifier:
    return Identifier(name)
