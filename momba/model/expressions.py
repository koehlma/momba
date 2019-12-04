# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import abc
import dataclasses
import numbers
import typing

from . import types


Identifier = str

TypeEnv = typing.Mapping[Identifier, 'types.Type']

EMPTY_ENV: TypeEnv = {}


class UnboundVariableError(Exception):
    pass


class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def is_constant(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def infer_type(self, env: TypeEnv) -> types.Type:
        raise NotImplementedError()


class Constant(Expression, abc.ABC):
    @property
    def is_constant(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class IntegerConstant(Constant):
    integer: int

    def infer_type(self, env: TypeEnv) -> types.Type:
        return types.INT


@dataclasses.dataclass(frozen=True)
class BoolConstant(Constant):
    boolean: bool

    def infer_type(self, env: TypeEnv) -> types.Type:
        return types.BOOL


@dataclasses.dataclass(frozen=True)
class RealConstant(Constant):
    # TODO: this should not be just any string
    real: typing.Union[str, numbers.Number]

    def infer_type(self, env: TypeEnv) -> types.Type:
        return types.REAL


class Variable(Expression):
    identifier: Identifier

    @property
    def is_constant(self) -> bool:
        return False

    def infer_type(self, env: TypeEnv) -> types.Type:
        try:
            return env[self.identifier]
        except KeyError:
            raise UnboundVariableError(
                f'variable {self.identifier} is not bound in type environment'
            )


def infer_type_of(expr: Expression, env: typing.Optional[TypeEnv] = None) -> types.Type:
    return expr.infer_type(env or EMPTY_ENV)
