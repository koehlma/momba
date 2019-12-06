# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import typing

from . import distribution, operators, types, values


Identifier = str

_Declarations = typing.Dict[Identifier, 'types.Type']


class TypeSystemError(ValueError):
    pass


class InvalidTypeError(TypeSystemError):
    pass


class UndeclaredVariableError(TypeSystemError):
    pass


@dataclasses.dataclass(frozen=True)
class TypeContext:
    _declarations: _Declarations = dataclasses.field(default_factory=dict)

    def declare(self, identifier: Identifier, typ: types.Type) -> None:
        self._declarations[identifier] = typ

    def lookup(self, identifier: Identifier) -> types.Type:
        try:
            return self._declarations[identifier]
        except KeyError:
            raise UndeclaredVariableError(
                f'variable {identifier} has not been declared in type context'
            )


class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def children(self) -> typing.Sequence[Expression]:
        raise NotImplementedError()

    @property
    def is_constant(self) -> bool:
        return False

    @abc.abstractmethod
    def infer_type(self, ctx: TypeContext) -> types.Type:
        """ Infers the type of the expression.

        If no type can be inferred, an exception is raised.
        """
        raise NotImplementedError()

    @property
    def traverse(self) -> typing.Iterable[Expression]:
        yield self
        for child in self.children:
            yield from child.traverse

    @property
    def subexpressions(self) -> typing.AbstractSet[Expression]:
        return frozenset(self.traverse)

    @property
    def is_sampling_free(self) -> bool:
        return all(not isinstance(e, Sample) for e in self.traverse)

    def __and__(self, other: Expression) -> Expression:
        if not isinstance(other, Expression):
            return NotImplemented
        return Boolean(operators.BooleanOperator.AND, self, other)

    def __or__(self, other: Expression) -> Expression:
        if not isinstance(other, Expression):
            return NotImplemented
        return Boolean(operators.BooleanOperator.OR, self, other)

    def eq(self, other: Expression) -> Expression:
        return Equality(
            operators.EqualityOperator.EQ, self, other
        )

    def neq(self, other: Expression) -> Expression:
        return Equality(
            operators.EqualityOperator.NEQ, self, other
        )

    def lt(self, other: Expression) -> Expression:
        return NumericComparison(
            operators.ComparisonOperator.LT, self, other
        )

    def le(self, other: Expression) -> Expression:
        return NumericComparison(
            operators.ComparisonOperator.LE, self, other
        )


@dataclasses.dataclass(frozen=True)
class Constant(Expression):
    value: values.Value

    @property
    def children(self) -> typing.Sequence[Expression]:
        return ()

    @property
    def is_constant(self) -> bool:
        return True

    def infer_type(self, ctx: TypeContext) -> types.Type:
        return self.value.typ


@dataclasses.dataclass(frozen=True)
class Variable(Expression):
    identifier: Identifier

    @property
    def children(self) -> typing.Sequence[Expression]:
        return ()

    def infer_type(self, ctx: TypeContext) -> types.Type:
        return ctx.lookup(self.identifier)


# XXX: this class should be abstract, however, then it does not type-check
# https://github.com/python/mypy/issues/5374
@dataclasses.dataclass(frozen=True)
class BinaryExpression(Expression):
    operator: operators.BinaryOperator
    left: Expression
    right: Expression

    @property
    def children(self) -> typing.Sequence[Expression]:
        return self.left, self.right

    # XXX: this method shall be implemented by all subclasses
    def infer_type(self, ctx: TypeContext) -> types.Type:
        raise NotImplementedError()


class Boolean(BinaryExpression):
    operator: operators.BooleanOperator

    def infer_type(self, ctx: TypeContext) -> types.Type:
        left_type = self.left.infer_type(ctx)
        if left_type != types.BOOL:
            raise InvalidTypeError(f'expected types.BOOL but got {left_type}')
        right_type = self.right.infer_type(ctx)
        if right_type != types.BOOL:
            raise InvalidTypeError(f'expected types.BOOL but got {right_type}')
        return types.BOOL


class Arithmetic(BinaryExpression):
    operator: operators.ArithmeticOperator

    def infer_type(self, ctx: TypeContext) -> types.Type:
        left_type = self.left.infer_type(ctx)
        right_type = self.right.infer_type(ctx)
        if not left_type.is_numeric or not right_type.is_numeric:
            raise InvalidTypeError('operands of arithmetic expressions must have a numeric type')
        if types.INT.is_assignable_from(left_type) and types.INT.is_assignable_from(right_type):
            return types.INT
        return types.REAL


class Equality(BinaryExpression):
    operator: operators.EqualityOperator

    def infer_type(self, ctx: TypeContext) -> types.Type:
        left_type = self.left.infer_type(ctx)
        right_type = self.right.infer_type(ctx)
        # XXX: JANI specifies that “left and right must be assignable to some common type”
        if (not left_type.is_assignable_from(right_type)
                and not right_type.is_assignable_from(left_type)):
            raise InvalidTypeError(
                'invalid combination of type for equality comparison'
            )
        return types.BOOL


class NumericComparison(BinaryExpression):
    operator: operators.ComparisonOperator

    def infer_type(self, ctx: TypeContext) -> types.Type:
        left_type = self.left.infer_type(ctx)
        if left_type.is_numeric:
            raise InvalidTypeError(f'expected numeric type but got {left_type}')
        right_type = self.right.infer_type(ctx)
        if right_type.is_numeric:
            raise InvalidTypeError(f'expected numeric type but got {right_type}')
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression

    @property
    def children(self) -> typing.Sequence[Expression]:
        return self.condition, self.consequence, self.alternative

    def infer_type(self, ctx: TypeContext) -> types.Type:
        condition_type = self.condition.infer_type(ctx)
        if condition_type != types.BOOL:
            raise InvalidTypeError(
                f'expected `types.BOOL` but got `{condition_type}`'
            )
        consequence_type = self.consequence.infer_type(ctx)
        alternative_type = self.alternative.infer_type(ctx)
        if consequence_type.is_assignable_from(alternative_type):
            return consequence_type
        elif alternative_type.is_assignable_from(consequence_type):
            return alternative_type
        else:
            raise InvalidTypeError(
                'invalid combination of consequence and alternative types'
            )


@dataclasses.dataclass(frozen=True)
class Not(Expression):
    operand: Expression

    @property
    def children(self) -> typing.Sequence[Expression]:
        return self.operand,

    def infer_type(self, ctx: TypeContext) -> types.Type:
        operand_type = self.operand.infer_type(ctx)
        if operand_type != types.BOOL:
            raise InvalidTypeError(f'expected `types.BOOL` but got {operand_type}')
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class Sample(Expression):
    distribution: distribution.Distribution
    arguments: typing.Sequence[Expression]

    def __post_init__(self) -> None:
        if len(self.arguments) != len(self.distribution.parameter_types):
            raise ValueError('arity of parameters and arity does not match')

    @property
    def children(self) -> typing.Sequence[Expression]:
        return self.arguments

    def infer_type(self, ctx: TypeContext) -> types.Type:
        # we already know that the arity of the parameters and arguments match
        for argument, parameter_type in zip(self.arguments, self.distribution.parameter_types):
            argument_type = argument.infer_type(ctx)
            if not parameter_type.is_assignable_from(argument_type):
                raise InvalidTypeError(
                    f'parameter type `{parameter_type}` is not assignable '
                    f'from argument type `{argument_type}`'
                )
        return types.REAL


@dataclasses.dataclass(frozen=True)
class Selection(Expression):
    identifier: Identifier
    condition: Expression

    def infer_type(self, ctx: TypeContext) -> types.Type:
        condition_type = self.condition.infer_type(ctx)
        if condition_type != types.BOOL:
            raise InvalidTypeError('condition must have type `types.BOOL`')
        return ctx.lookup(self.identifier)

    @property
    def children(self) -> typing.Sequence[Expression]:
        return self.condition,


@dataclasses.dataclass(frozen=True)
class Derivative(Expression):
    identifier: Identifier

    def infer_type(self, ctx: TypeContext) -> types.Type:
        return types.REAL

    @property
    def children(self) -> typing.Sequence[Expression]:
        return ()


def infer_type_of(expr: Expression, ctx: typing.Optional[TypeContext] = None) -> types.Type:
    return expr.infer_type(ctx or TypeContext())


def var(identifier: Identifier) -> Expression:
    return Variable(identifier)


def ite(condition: Expression, consequence: Expression, alternative: Expression) -> Expression:
    return Conditional(condition, consequence, alternative)


def const(value: values.PythonValue) -> Constant:
    return Constant(values.pack(value))


MaybeExpression = typing.Union[values.PythonValue, Expression]


def cast(value: MaybeExpression) -> Expression:
    if isinstance(value, Expression):
        return value
    return const(value)
