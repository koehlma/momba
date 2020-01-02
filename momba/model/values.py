# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import enum
import dataclasses
import math
import numbers

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


@dataclasses.dataclass(frozen=True)
class RealValue(NumericValue):
    real: t.Union[NamedReal, numbers.Real]

    @property
    def typ(self) -> types.Type:
        return types.REAL

    @property
    def as_float(self) -> float:
        if isinstance(self.real, NamedReal):
            return self.real.float_value
        return float(self.real)


PythonRealString = t.Literal["π", "e"]
PythonReal = t.Union[numbers.Real, float, PythonRealString, NamedReal]
PythonNumeric = t.Union[int, PythonReal]
PythonValue = t.Union[bool, PythonNumeric]


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
        return RealValue(_NAMED_REAL_MAP[value])
    raise ConversionError(f"unable to convert Python value {value!r} to Momba value")


def unpack(value: Value) -> PythonValue:
    if isinstance(value, BooleanValue):
        return value.boolean
    return unpack_numeric(t.cast(NumericValue, value))


def unpack_numeric(value: NumericValue) -> PythonNumeric:
    if isinstance(value, IntegerValue):
        return value.integer
    elif isinstance(value, RealValue):
        return value.real
    raise ConversionError(f"unable to convert Momba value {value!r} to Python value")
