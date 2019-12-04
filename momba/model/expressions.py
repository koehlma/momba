# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import numbers
import typing

from . import types


class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def is_constant(self) -> bool:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def typ(self) -> types.Type:
        raise NotImplementedError()


class Constant(Expression, abc.ABC):
    @property
    def is_constant(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class IntegerConstant(Constant):
    integer: int

    @property
    def typ(self) -> types.Type:
        return types.INT


@dataclasses.dataclass(frozen=True)
class BoolConstant(Constant):
    boolean: bool

    @property
    def typ(self) -> types.Type:
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class RealConstant(Constant):
    # TODO: this should not be just any string
    real: typing.Union[str, numbers.Number]

    @property
    def typ(self) -> types.Type:
        return types.REAL
