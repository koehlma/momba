# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses


if t.TYPE_CHECKING:
    from . import context, types


@dataclasses.dataclass(frozen=True)
class ActionType:
    name: str
    parameters: t.Sequence[ActionParameter]

    comment: t.Optional[str] = None

    @property
    def has_parameters(self) -> bool:
        return bool(self.parameters)


@dataclasses.dataclass(frozen=True)
class ActionParameter:
    typ: types.Type

    comment: t.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ActionPattern:
    action_type: ActionType

    arguments: t.Sequence[str] = ()

    def __post_init__(self) -> None:
        assert len(self.action_type.parameters) == len(self.arguments)

    def apply(self, scope: context.Scope) -> None:
        for name, parameter in zip(self.arguments, self.action_type.parameters):
            scope.declare_variable(name, parameter.typ)
