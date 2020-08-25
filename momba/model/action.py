# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc


if t.TYPE_CHECKING:
    from . import context, expressions, types


@d.dataclass(frozen=True)
class ActionParameter:
    typ: types.Type

    comment: t.Optional[str] = None


@d.dataclass(frozen=True)
class ActionType:
    name: str

    parameters: t.Tuple[ActionParameter, ...] = ()

    comment: t.Optional[str] = None

    @property
    def has_parameters(self) -> bool:
        return bool(self.parameters)

    def create_pattern(self, *arguments: ActionArgument) -> ActionPattern:
        return ActionPattern(self, arguments=arguments)


class ActionArgument(abc.ABC):
    pass


@d.dataclass(frozen=True)
class WriteArgument(ActionArgument):
    # is evaluated in the automata's scope, not in the edge scope
    expression: expressions.Expression


@d.dataclass(frozen=True)
class ReadArgument(ActionArgument):
    # has to be declared in the automata's scope, may be transient
    name: str


@d.dataclass(frozen=True)
class GuardArgument(ActionArgument):
    name: str


@d.dataclass(frozen=True)
class ActionPattern:
    action_type: ActionType

    arguments: t.Tuple[ActionArgument, ...] = ()

    def __post_init__(self) -> None:
        if len(self.action_type.parameters) != len(self.arguments):
            raise ValueError("number of parameters andarguments does not match")

    def declare_in(self, scope: context.Scope) -> None:
        """ Declares the identifiers of the pattern in the given scope. """
        for argument, parameter in zip(self.arguments, self.action_type.parameters):
            if isinstance(argument, GuardArgument):
                if scope.is_declared(argument.name):
                    assert scope.lookup(argument.name).typ == parameter.typ
                else:
                    scope.declare(argument.name, parameter.typ)
            elif isinstance(argument, ReadArgument):
                # FIXME: proper error reporting, should be declared, e.g., as transient
                assert scope.is_declared(argument.name)
