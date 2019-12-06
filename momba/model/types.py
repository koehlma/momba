# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import typing

from . import values


class Type(abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return False

    @abc.abstractmethod
    def is_assignable_from(self, typ: Type) -> bool:
        raise NotImplementedError()


class Numeric(Type, abc.ABC):
    @property
    def is_numeric(self) -> bool:
        return True

    def __getitem__(self, bounds: Bounds) -> BoundedType:
        lower_bound = BoundedType.cast_bound(bounds[0])
        upper_bound = BoundedType.cast_bound(bounds[1])
        return BoundedType(self, lower_bound, upper_bound)


@dataclasses.dataclass(frozen=True)
class _Integer(Numeric):
    def is_assignable_from(self, typ: Type) -> bool:
        return typ == INT or (isinstance(typ, BoundedType) and typ.base == INT)


@dataclasses.dataclass(frozen=True)
class _Real(Numeric):
    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


@dataclasses.dataclass(frozen=True)
class _Bool(Type):
    def is_assignable_from(self, typ: Type) -> bool:
        return typ == BOOL


@dataclasses.dataclass(frozen=True)
class _Clock(Numeric):
    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


@dataclasses.dataclass(frozen=True)
class _Continuous(Numeric):
    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


INT = _Integer()
REAL = _Real()
BOOL = _Bool()
CLOCK = _Clock()
CONTINUOUS = _Continuous()


Bound = typing.Optional[typing.Union['values.PythonNumeric', 'ellipsis']]
Bounds = typing.Tuple[Bound, Bound]


class TypeConstructionError(ValueError):
    pass


class BaseTypeError(TypeConstructionError):
    pass


class InvalidBoundError(TypeConstructionError):
    pass


@dataclasses.dataclass(frozen=True)
class BoundedType(Numeric):
    base: Numeric

    lower_bound: typing.Optional[values.NumericValue]
    upper_bound: typing.Optional[values.NumericValue]

    def __post_init__(self) -> None:
        if not isinstance(self.base, Numeric):
            raise BaseTypeError('base-type of bounded type must be numeric')
        if self.lower_bound is None and self.upper_bound is None:
            raise InvalidBoundError('neither `lower_bound` nor `upper_bound` is present')
        if self.lower_bound is not None:
            if not isinstance(self.lower_bound, values.NumericValue):
                raise InvalidBoundError('`lower_bound` is not a numeric constant')
            if not self.base.is_assignable_from(self.lower_bound.typ):
                raise InvalidBoundError('type of `lower_bound` is not assignable to base-type')
        if self.upper_bound is not None:
            if not isinstance(self.upper_bound, values.NumericValue):
                raise InvalidBoundError('`upper_bound` is not a numeric constant')
            if not self.base.is_assignable_from(self.upper_bound.typ):
                raise InvalidBoundError('type of `upper_bound` is not assignable to base-type')

    @staticmethod
    def cast_bound(bound: Bound) -> typing.Optional[values.NumericValue]:
        if bound is None or bound is Ellipsis:
            return None
        return values.pack_numeric(typing.cast(values.PythonNumeric, bound))

    def is_assignable_from(self, typ: Type) -> bool:
        return self.base.is_assignable_from(typ)


@dataclasses.dataclass(frozen=True)
class ArrayType(Type):
    base: Type

    def is_assignable_from(self, typ: Type) -> bool:
        return isinstance(typ, ArrayType) and self.base.is_assignable_from(typ.base)


def array_of(base: Type) -> ArrayType:
    return ArrayType(base)
