# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import dataclasses

from . import context, errors, operators, types, values

if t.TYPE_CHECKING:
    from . import distributions


class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def children(self) -> t.Sequence[Expression]:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_constant_in(self, scope: context.Scope) -> bool:
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

    def __and__(self, other: Expression) -> Expression:
        if not isinstance(other, Expression):
            return NotImplemented
        return Boolean(operators.Boolean.AND, self, other)

    def __or__(self, other: Expression) -> Expression:
        if not isinstance(other, Expression):
            return NotImplemented
        return Boolean(operators.Boolean.OR, self, other)

    def __ior__(self, other: Expression) -> Expression:
        return self | other

    def __invert__(self) -> Expression:
        return lnot(self)

    def __add__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.ADD, self, convert(other))

    def __radd__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.ADD, convert(other), self)

    def __sub__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.SUB, self, convert(other))

    def __rsub__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.SUB, convert(other), self)

    def __mul__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MUL, self, convert(other))

    def __rmul__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MUL, convert(other), self)

    def __mod__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MOD, self, convert(other))

    def __rmod__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.MOD, convert(other), self)

    def __floordiv__(self, other: MaybeExpression) -> Expression:
        return Arithmetic(operators.ArithmeticOperator.FLOOR_DIV, convert(other), self)

    def __lt__(self, other: MaybeExpression) -> Expression:
        return lt(self, convert(other))

    def __le__(self, other: MaybeExpression) -> Expression:
        return le(self, convert(other))

    def __gt__(self, other: MaybeExpression) -> Expression:
        return gt(self, convert(other))

    def __ge__(self, other: MaybeExpression) -> Expression:
        return ge(self, convert(other))

    def eq(self, other: Expression) -> Expression:
        return Equality(
            operators.EqualityOperator.EQ, self, other
        )

    def neq(self, other: Expression) -> Expression:
        return Equality(
            operators.EqualityOperator.NEQ, self, other
        )

    def lt(self, other: Expression) -> Expression:
        return Comparison(
            operators.Comparison.LT, self, other
        )

    def le(self, other: Expression) -> Expression:
        return Comparison(
            operators.Comparison.LE, self, other
        )

    def ge(self, other: Expression) -> Expression:
        return Comparison(
            operators.Comparison.GE, self, other
        )

    def gt(self, other: Expression) -> Expression:
        return Comparison(
            operators.Comparison.GT, self, other
        )

    def land(self, other: Expression) -> Expression:
        return land(self, other)


@dataclasses.dataclass(frozen=True)
class Constant(Expression):
    value: values.Value

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()

    def is_constant_in(self, scope: context.Scope) -> bool:
        return True

    def infer_type(self, scope: context.Scope) -> types.Type:
        return self.value.typ


@dataclasses.dataclass(frozen=True)
class Identifier(Expression):
    identifier: context.Identifier

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()

    def is_constant_in(self, scope: context.Scope) -> bool:
        return scope.lookup(self.identifier).is_constant_in(scope)

    def infer_type(self, scope: context.Scope) -> types.Type:
        declaration = scope.lookup(self.identifier)
        if isinstance(declaration, context.VariableDeclaration):
            return declaration.typ
        elif isinstance(declaration, context.ConstantDeclaration):
            return declaration.typ
        assert False


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
        return (
            self.left.is_constant_in(scope)
            and self.right.is_constant_in(scope)
        )

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


class Boolean(BinaryExpression):
    operator: operators.Boolean

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if left_type != types.BOOL:
            raise errors.InvalidTypeError(f'expected types.BOOL but got {left_type}')
        right_type = scope.get_type(self.right)
        if right_type != types.BOOL:
            raise errors.InvalidTypeError(f'expected types.BOOL but got {right_type}')
        return types.BOOL


_REAL_OPERATORS = {
    operators.ArithmeticOperator.REAL_DIV,
    operators.ArithmeticOperator.LOG,
    operators.ArithmeticOperator.POW
}


class Arithmetic(BinaryExpression):
    operator: operators.ArithmeticOperator

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        if not left_type.is_numeric or not right_type.is_numeric:
            raise errors.InvalidTypeError(
                'operands of arithmetic expressions must have a numeric type'
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
        assert False, 'type-inference should ensure that some of the above is true'

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        right_type = scope.get_type(self.right)
        # XXX: JANI specifies that “left and right must be assignable to some common type”
        if (not left_type.is_assignable_from(right_type)
                and not right_type.is_assignable_from(left_type)):
            raise errors.InvalidTypeError(
                'invalid combination of type for equality comparison'
            )
        return types.BOOL


class Comparison(BinaryExpression):
    operator: operators.Comparison

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = scope.get_type(self.left)
        if not left_type.is_numeric:
            raise errors.InvalidTypeError(
                f'expected numeric type but got {left_type}'
            )
        right_type = scope.get_type(self.right)
        if not right_type.is_numeric:
            raise errors.InvalidTypeError(
                f'expected numeric type but got {right_type}'
            )
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
                f'expected `types.BOOL` but got `{condition_type}`'
            )
        consequence_type = scope.get_type(self.consequence)
        alternative_type = scope.get_type(self.alternative)
        if consequence_type.is_assignable_from(alternative_type):
            return consequence_type
        elif alternative_type.is_assignable_from(consequence_type):
            return alternative_type
        else:
            raise errors.InvalidTypeError(
                'invalid combination of consequence and alternative types'
            )


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@dataclasses.dataclass(frozen=True)
class UnaryExpression(Expression):
    operator: operators.UnaryOperator
    operand: Expression

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.operand,

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
            raise errors.InvalidTypeError(f'expected a numeric type but got {operand_type}')
        return types.INT


class Not(UnaryExpression):
    operator: operators.Not

    def infer_type(self, scope: context.Scope) -> types.Type:
        operand_type = scope.get_type(self.operand)
        if operand_type != types.BOOL:
            raise errors.InvalidTypeError(f'expected `types.BOOL` but got {operand_type}')
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class Sample(Expression):
    distribution: distributions.Distribution
    arguments: t.Sequence[Expression]

    def __post_init__(self) -> None:
        if len(self.arguments) != len(self.distribution.parameter_types):
            raise ValueError('parameter and arguments arity mismatch')

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.arguments

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    def infer_type(self, scope: context.Scope) -> types.Type:
        # we already know that the arity of the parameters and arguments match
        for argument, parameter_type in zip(self.arguments, self.distribution.parameter_types):
            argument_type = scope.get_type(argument)
            if not parameter_type.is_assignable_from(argument_type):
                raise errors.InvalidTypeError(
                    f'parameter type `{parameter_type}` is not assignable '
                    f'from argument type `{argument_type}`'
                )
        return types.REAL


@dataclasses.dataclass(frozen=True)
class Selection(Expression):
    identifier: context.Identifier
    condition: Expression

    def infer_type(self, scope: context.Scope) -> types.Type:
        condition_type = scope.get_type(self.condition)
        if condition_type != types.BOOL:
            raise errors.InvalidTypeError('condition must have type `types.BOOL`')
        declaration = scope.lookup(self.identifier)
        assert isinstance(declaration, context.VariableDeclaration)
        return declaration.typ

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    @property
    def children(self) -> t.Sequence[Expression]:
        return self.condition,


@dataclasses.dataclass(frozen=True)
class Derivative(Expression):
    identifier: context.Identifier

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.REAL

    def is_constant_in(self, scope: context.Scope) -> bool:
        return False

    @property
    def children(self) -> t.Sequence[Expression]:
        return ()


def var(identifier: context.Identifier) -> Identifier:
    return Identifier(identifier)


def ite(condition: Expression, consequence: Expression, alternative: Expression) -> Expression:
    return Conditional(condition, consequence, alternative)


def const(value: values.PythonValue) -> Constant:
    return Constant(values.pack(value))


MaybeExpression = t.Union[values.PythonValue, Expression]


def convert(value: MaybeExpression) -> Expression:
    if isinstance(value, Expression):
        return value
    return const(value)


def identifier(name: str) -> Identifier:
    return Identifier(name)


BinaryConstructor = t.Callable[[Expression, Expression], Expression]


def lor(*expressions: Expression) -> Expression:
    if len(expressions) == 2:
        return Boolean(operators.Boolean.OR, expressions[0], expressions[1])
    result = convert(False)
    for disjunct in expressions:
        result |= disjunct
    return result


def land(*expressions: Expression) -> Expression:
    if len(expressions) == 2:
        return Boolean(operators.Boolean.AND, expressions[0], expressions[1])
    result = convert(True)
    for conjunct in expressions:
        result &= conjunct
    return result


def xor(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.Boolean.XOR, left, right)


def implies(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.Boolean.IMPLY, left, right)


def equiv(left: Expression, right: Expression) -> Expression:
    return Boolean(operators.Boolean.EQUIV, left, right)


def eq(left: Expression, right: Expression) -> BinaryExpression:
    return Equality(operators.EqualityOperator.EQ, left, right)


def neq(left: Expression, right: Expression) -> BinaryExpression:
    return Equality(operators.EqualityOperator.NEQ, left, right)


def lt(left: Expression, right: Expression) -> BinaryExpression:
    return Comparison(operators.Comparison.LT, left, right)


def le(left: Expression, right: Expression) -> BinaryExpression:
    return Comparison(operators.Comparison.LE, left, right)


def ge(left: Expression, right: Expression) -> BinaryExpression:
    return Comparison(operators.Comparison.GE, left, right)


def gt(left: Expression, right: Expression) -> BinaryExpression:
    return Comparison(operators.Comparison.GT, left, right)


def add(left: Expression, right: Expression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.ADD, left, right)


def sub(left: Expression, right: Expression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.SUB, left, right)


def mul(left: Expression, right: Expression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MUL, left, right)


def mod(left: Expression, right: Expression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MOD, left, right)


def minimum(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MIN, convert(left), convert(right))


def maximum(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.MAX, convert(left), convert(right))


def div(left: MaybeExpression, right: MaybeExpression) -> BinaryExpression:
    return Arithmetic(operators.ArithmeticOperator.REAL_DIV, convert(left), convert(right))


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
    assert isinstance(expr, Boolean) and expr.operator is operators.Boolean.XOR
    return lor(land(lnot(expr.left), expr.right), land(expr.right, lnot(expr.left)))


def normalize_equiv(expr: Expression) -> Expression:
    assert isinstance(expr, Boolean) and expr.operator is operators.Boolean.EQUIV
    return land(implies(expr.left, expr.right), implies(expr.right, expr.left))


def normalize_floor_div(expr: Expression) -> Expression:
    assert isinstance(expr, Arithmetic) and expr.operator is operators.ArithmeticOperator.FLOOR_DIV
    return floor(div(expr.left, expr.right))
