# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t


@d.dataclass(frozen=True)
class Value:
    """
    Represents a value.

    Attributes
    ----------
    as_int:
        Asserts that the value is an integer and returns the integer.
    as_float:
        Asserts that the value is a float and returns the float.
    as_bool:
        Asserts that the value is a bool and returns the bool.
    as_array:
        Asserts that the value is an array and returns a tuple of values.
    """

    _value: t.Any

    def __repr__(self) -> str:
        return f"Value({self._value})"

    @property
    def as_int(self) -> int:
        assert isinstance(self._value, int)
        return self._value

    @property
    def as_float(self) -> float:
        assert isinstance(self._value, float)
        return self._value

    @property
    def as_bool(self) -> bool:
        assert isinstance(self._value, bool)
        return self._value

    @property
    def as_array(self) -> t.Sequence[Value]:
        return tuple(map(Value, self._value))
