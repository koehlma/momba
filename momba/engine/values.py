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
    """

    _value: t.Any

    def __repr__(self) -> str:
        return f"Value({self._value})"

    @property
    def is_int(self) -> bool:
        """
        Returns :code:`True` if the value is an integer.
        """
        return isinstance(self._value, int)

    @property
    def is_float(self) -> bool:
        """
        Returns :code:`True` if the value is a float.
        """
        return isinstance(self._value, float)

    @property
    def is_bool(self) -> bool:
        """
        Returns :code:`True` if the value is a boolean.
        """
        return isinstance(self._value, bool)

    @property
    def is_array(self) -> bool:
        """
        Returns :code:`True` if the value is an array.
        """
        return isinstance(self._value, list)

    @property
    def as_int(self) -> int:
        """
        Asserts that the value is an integer and returns the integer.
        """
        assert self.is_int
        return self._value

    @property
    def as_float(self) -> float:
        """
        Asserts that the value is a float and returns the float.
        """
        assert isinstance(self._value, float)
        return self._value

    @property
    def as_bool(self) -> bool:
        """
        Asserts that the value is a boolean and returns the boolean.
        """
        assert isinstance(self._value, bool)
        return self._value

    @property
    def as_array(self) -> t.Tuple[Value, ...]:
        """
        Asserts that the value is an array and returns a tuple of values.
        """
        assert isinstance(self._value, list)
        return tuple(map(Value, self._value))
