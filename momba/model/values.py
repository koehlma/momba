# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import enum
import dataclasses
import numbers
import typing

from . import types


class Value(abc.ABC):
    @property
    @abc.abstractmethod
    def typ(self) -> types.Type:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class BooleanValue(Value):
    boolean: bool

    @property
    def typ(self) -> types.Type:
        return types.BOOL


TRUE = BooleanValue(True)
FALSE = BooleanValue(False)


class NumericValue(Value, abc.ABC):
    pass


@dataclasses.dataclass(frozen=True)
class IntegerValue(NumericValue):
    integer: int

    @property
    def typ(self) -> types.Type:
        return types.INT


class NamedReal(enum.Enum):
    PI = 'π'
    E = 'e'


@dataclasses.dataclass(frozen=True)
class RealValue(NumericValue):
    real: typing.Union[NamedReal, numbers.Number]

    @property
    def typ(self) -> types.Type:
        return types.REAL


PythonRealString = typing.Literal['π', 'e']
PythonReal = typing.Union[numbers.Number, float, PythonRealString, NamedReal]
PythonNumeric = typing.Union[int, PythonReal]
PythonValue = typing.Union[bool, PythonNumeric]


class ConversionError(ValueError):
    pass


def pack(value: PythonValue) -> Value:
    if isinstance(value, bool):
        return BooleanValue(value)
    return pack_numeric(value)


def pack_numeric(value: PythonNumeric) -> NumericValue:
    if isinstance(value, int):
        return IntegerValue(value)
    elif isinstance(value, (numbers.Number, NamedReal)):
        return RealValue(value)
    elif isinstance(value, str):
        return RealValue(NamedReal(value))
    raise ConversionError(f'unable to convert Python value {value!r} to Momba value')


def unpack(value: Value) -> PythonValue:
    if isinstance(value, BooleanValue):
        return value.boolean
    return unpack_numeric(typing.cast(NumericValue, value))


def unpack_numeric(value: NumericValue) -> PythonNumeric:
    if isinstance(value, IntegerValue):
        return value.integer
    elif isinstance(value, RealValue):
        return value.real
    raise ConversionError(f'unable to convert Momba value {value!r} to Python value')
