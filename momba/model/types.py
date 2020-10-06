# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import dataclasses

from . import context, expressions


class Type(abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return False

    @abc.abstractmethod
    def is_assignable_from(self, typ: Type) -> bool:
        raise NotImplementedError()

    def validate_in(self, scope: context.Scope) -> None:
        pass


class Numeric(Type, abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return True

    def bound(self, lower: Bound, upper: Bound) -> BoundedType:
        """Bounds the numeric type with the given bounds."""
        return BoundedType(
            self, BoundedType.convert_bound(lower), BoundedType.convert_bound(upper)
        )


@dataclasses.dataclass(frozen=True)
class IntegerType(Numeric):
    def __str__(self) -> str:
        return "types.INT"

    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


@dataclasses.dataclass(frozen=True)
class RealType(Numeric):
    def __str__(self) -> str:
        return "types.REAL"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


@dataclasses.dataclass(frozen=True)
class BoolType(Type):
    def __str__(self) -> str:
        return "types.BOOL"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ == BOOL


@dataclasses.dataclass(frozen=True)
class ClockType(Numeric):
    def __str__(self) -> str:
        return "types.CLOCK"

    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


@dataclasses.dataclass(frozen=True)
class ContinuousType(Numeric):
    def __str__(self) -> str:
        return "types.CONTINUOUS"

    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


INT = IntegerType()
REAL = RealType()
BOOL = BoolType()
CLOCK = ClockType()
CONTINUOUS = ContinuousType()


Bound = t.Optional[t.Union["expressions.MaybeExpression", "ellipsis"]]


class TypeConstructionError(ValueError):
    pass


class BaseTypeError(TypeConstructionError):
    pass


class InvalidBoundError(TypeConstructionError):
    pass


@dataclasses.dataclass(frozen=True)
class BoundedType(Numeric):
    base: Numeric

    lower_bound: t.Optional[expressions.Expression]
    upper_bound: t.Optional[expressions.Expression]

    def __str__(self) -> str:
        return f"{self.base}[{self.lower_bound}, {self.upper_bound}]"

    def __post_init__(self) -> None:
        if not isinstance(self.base, Numeric):
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
                    "type of `lower_bound` is not assignable to base-type"
                )
        if self.upper_bound is not None:
            if not scope.is_constant(self.upper_bound):
                raise InvalidBoundError("`upper_bound` has to be a constant")
            upper_bound_type = scope.get_type(self.upper_bound)
            if not self.base.is_assignable_from(upper_bound_type):
                raise InvalidBoundError(
                    "type of `upper_bound` is not assignable to base-type"
                )

    @staticmethod
    def convert_bound(bound: Bound) -> t.Optional[expressions.Expression]:
        if bound is None or bound is Ellipsis:
            return None
        return expressions.convert(t.cast(expressions.MaybeExpression, bound))

    def is_assignable_from(self, typ: Type) -> bool:
        return self.base.is_assignable_from(typ)


@dataclasses.dataclass(frozen=True)
class ArrayType(Type):
    base: Type

    def __str__(self) -> str:
        return f"Array({self.base})"

    def is_assignable_from(self, typ: Type) -> bool:
        return isinstance(typ, ArrayType) and self.base.is_assignable_from(typ.base)


def array_of(base: Type) -> ArrayType:
    return ArrayType(base)
