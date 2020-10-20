# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc

from . import errors, expressions

if t.TYPE_CHECKING:
    from . import context


class BaseTypeError(errors.TypeConstructionError):
    pass


class InvalidBoundError(errors.TypeConstructionError):
    pass


Bound = t.Optional[t.Union["expressions.ValueOrExpression", "ellipsis"]]


class Type(abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return False

    @abc.abstractmethod
    def is_assignable_from(self, typ: Type) -> bool:
        raise NotImplementedError()

    def validate_in(self, scope: context.Scope) -> None:
        pass


class NumericType(Type, abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return True

    def bound(self, lower: Bound, upper: Bound) -> BoundedType:
        """
        Bounds the numeric type with the given bounds.
        """
        return BoundedType(
            self, BoundedType.convert_bound(lower), BoundedType.convert_bound(upper)
        )


@d.dataclass(frozen=True)
class IntegerType(NumericType):
    def __str__(self) -> str:
        return "types.INT"

    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


@d.dataclass(frozen=True)
class RealType(NumericType):
    def __str__(self) -> str:
        return "types.REAL"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


@d.dataclass(frozen=True)
class BoolType(Type):
    def __str__(self) -> str:
        return "types.BOOL"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ == BOOL


@d.dataclass(frozen=True)
class ClockType(NumericType):
    def __str__(self) -> str:
        return "types.CLOCK"

    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


@d.dataclass(frozen=True)
class ContinuousType(NumericType):
    def __str__(self) -> str:
        return "types.CONTINUOUS"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


INT = IntegerType()
REAL = RealType()
BOOL = BoolType()
CLOCK = ClockType()
CONTINUOUS = ContinuousType()


@d.dataclass(frozen=True)
class BoundedType(NumericType):
    base: NumericType

    lower_bound: t.Optional[expressions.Expression]
    upper_bound: t.Optional[expressions.Expression]

    def __str__(self) -> str:
        return f"{self.base}[{self.lower_bound}, {self.upper_bound}]"

    def __post_init__(self) -> None:
        if not isinstance(self.base, NumericType):
            raise BaseTypeError("base-type of bounded type must be numeric")
        if self.lower_bound is None and self.upper_bound is None:
            raise InvalidBoundError(
                "neither `lower_bound` nor `upper_bound` is present"
            )

    def validate_in(self, scope: context.Scope) -> None:
        if self.lower_bound is not None:
            if not scope.is_constant(self.lower_bound):
                raise InvalidBoundError("`lower_bound` has to be a constant")
            lower_bound_type = scope.get_type(self.lower_bound)
            if not self.base.is_assignable_from(lower_bound_type):
                raise InvalidBoundError(
                    f"type {lower_bound_type} of `lower_bound` is not "
                    f"assignable to base-type {self.base}"
                )
        if self.upper_bound is not None:
            if not scope.is_constant(self.upper_bound):
                raise InvalidBoundError("`upper_bound` has to be a constant")
            upper_bound_type = scope.get_type(self.upper_bound)
            if not self.base.is_assignable_from(upper_bound_type):
                raise InvalidBoundError(
                    f"type {upper_bound_type} of `upper_bound` is not "
                    f"assignable to base-type {self.base}"
                )

    @staticmethod
    def convert_bound(bound: Bound) -> t.Optional[expressions.Expression]:
        if bound is None or bound is Ellipsis:
            return None
        return expressions.ensure_expr(t.cast(expressions.ValueOrExpression, bound))

    def is_assignable_from(self, typ: Type) -> bool:
        return self.base.is_assignable_from(typ)


@d.dataclass(frozen=True)
class ArrayType(Type):
    element: Type

    def __str__(self) -> str:
        return f"Array({self.element})"

    def is_assignable_from(self, typ: Type) -> bool:
        return isinstance(typ, ArrayType) and self.element.is_assignable_from(
            typ.element
        )


def array_of(element: Type) -> ArrayType:
    return ArrayType(element)


@d.dataclass(frozen=True)
class SetType(Type):
    element: Type

    def __str__(self) -> str:
        return f"Set({self.element})"

    def is_assignable_from(self, typ: Type) -> bool:
        return isinstance(typ, SetType) and self.element.is_assignable_from(typ.element)


def set_of(element: Type) -> SetType:
    return SetType(element)


@d.dataclass(frozen=True)
class StateType(Type):
    def __str__(self) -> str:
        return "types.STATE"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ == STATE


STATE = StateType()


class InferTypeUnary(t.Protocol):
    def __call__(self, operand: Type) -> Type:
        pass


class InferTypeBinary(t.Protocol):
    def __call__(self, left: Type, right: Type) -> Type:
        pass
