# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import enum
import numbers
import typing

from .. import utils

from . import expressions


class Type(abc.ABC):
    @property
    @abc.abstractmethod
    def is_numeric(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_assignable_from(self, typ: Type) -> bool:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _BasicType(Type):
    class Kind(enum.Enum):
        BOOL = 'bool'
        INT = 'int'
        REAL = 'real'

    kind: Kind

    def __str__(self) -> str:
        return f'types.{self.kind.value.upper()}'

    def __getitem__(self, key: slice) -> BoundedType:
        lower_bound = BoundedType.cast_bound(key.start)
        upper_bound = BoundedType.cast_bound(key.stop)
        return BoundedType(self, lower_bound, upper_bound)

    @property
    def is_numeric(self) -> bool:
        return self in {INT, REAL}

    def is_assignable_from(self, typ: Type) -> bool:
        if self == typ:
            return True
        elif isinstance(typ, BoundedType) and self == INT:
            return typ.base == INT
        elif self == REAL:
            return typ.is_numeric
        return False


BOOL = _BasicType(_BasicType.Kind.BOOL)
INT = _BasicType(_BasicType.Kind.INT)
REAL = _BasicType(_BasicType.Kind.REAL)


Bound = typing.Optional[
    typing.Union[int, str, numbers.Number, expressions.Expression]
]


class BaseTypeError(ValueError):
    pass


class InvalidBoundError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class BoundedType(Type):
    base: _BasicType

    lower_bound: typing.Optional[expressions.Expression]
    upper_bound: typing.Optional[expressions.Expression]

    def __post_init__(self) -> None:
        if self.base not in {INT, REAL}:
            raise BaseTypeError('base-type of bounded type must be INT or REAL')
        if self.lower_bound is None and self.upper_bound is None:
            raise InvalidBoundError('neither `lower_bound` nor `upper_bound` is present')
        if self.lower_bound is not None:
            if not self.lower_bound.is_constant:
                raise InvalidBoundError('`lower_bound` is not a constant expression')
            if not self.base.is_assignable_from(expressions.infer_type_of(self.lower_bound)):
                raise InvalidBoundError('type of `lower_bound` is not assignable to base-type')
        if self.upper_bound is not None:
            if not self.upper_bound.is_constant:
                raise InvalidBoundError('`upper_bound` is not a constant expression')
            if not self.base.is_assignable_from(expressions.infer_type_of(self.upper_bound)):
                raise InvalidBoundError('type of `upper_bound` is not assignable to base-type')

    @staticmethod
    def cast_bound(bound: Bound) -> typing.Optional[expressions.Expression]:
        if isinstance(bound, int):
            return expressions.IntegerConstant(bound)
        elif isinstance(bound, expressions.Expression):
            return bound
        elif isinstance(bound, (str, numbers.Number)):
            return expressions.RealConstant(bound)
        elif bound is None:
            return None
        else:
            raise InvalidBoundError(f'unable to cast the given bound into an expression')

    @property
    def is_numeric(self) -> bool:
        return self.base.is_numeric

    def is_assignable_from(self, typ: Type) -> bool:
        return self.base.is_assignable_from(typ)


@dataclasses.dataclass(frozen=True)
class _ClockType(Type):
    def __str__(self) -> str:
        return f'types.CLOCK'

    @property
    def is_numeric(self) -> bool:
        return True

    def is_assignable_from(self, typ: Type) -> bool:
        if isinstance(typ, BoundedType):
            return typ.base == INT
        return typ == INT


CLOCK = _ClockType()


@dataclasses.dataclass(frozen=True)
class _ContinuousType(Type):
    def __str__(self) -> str:
        return f'types.CONTINUOUS'

    @property
    def is_numeric(self) -> bool:
        return True

    def is_assignable_from(self, typ: Type) -> bool:
        return typ.is_numeric


CONTINUOUS = _ContinuousType()


@dataclasses.dataclass(frozen=True)
class ArrayType(Type):
    # TODO: is this allowed to be any type including CLOCK and CONTINUOUS?
    base: Type

    @property
    def is_numeric(self) -> bool:
        return False

    @utils.lru_cache
    def is_assignable_from(self, typ: Type) -> bool:
        return isinstance(typ, ArrayType) and self.base.is_assignable_from(typ.base)


def array_of(typ: Type) -> ArrayType:
    return ArrayType(typ)
