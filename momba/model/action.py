# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses


if t.TYPE_CHECKING:
    from . import context, types


@dataclasses.dataclass(frozen=True)
class ActionParameter:
    typ: types.Type

    comment: t.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ActionType:
    name: str

    # XXX: unable to type, could be any kind of hashable sequence
    parameters: t.Tuple[ActionParameter, ...] = ()

    comment: t.Optional[str] = None

    @property
    def has_parameters(self) -> bool:
        return bool(self.parameters)

    def create_pattern(self, *identifiers: str) -> ActionPattern:
        return ActionPattern(self, identifiers=identifiers)


@dataclasses.dataclass(frozen=True)
class ActionPattern:
    action_type: ActionType

    # XXX: unable to type, could be any kind of hashable sequence
    identifiers: t.Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.action_type.parameters) != len(self.identifiers):
            raise ValueError(f"number of parameters and identifiers does not match")

    def declare_in(self, scope: context.Scope) -> None:
        """ Declares the identifiers of the pattern in the given scope. """
        for name, parameter in zip(self.identifiers, self.action_type.parameters):
            scope.declare(name, parameter.typ)
