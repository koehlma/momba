# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from . import errors, types
from .actions import ActionPattern
from .automata import Instance
from .expressions import Expression

if t.TYPE_CHECKING:
    from . import context


@d.dataclass(frozen=True, eq=False)
class Link:
    vector: t.Mapping[Instance, ActionPattern]
    result: t.Optional[ActionPattern] = None
    condition: t.Optional[Expression] = None

    def construct_scope(self, ctx: context.Context) -> context.Scope:
        scope = ctx.global_scope.create_child_scope()
        for pattern in self.vector.values():
            pattern.declare_in(scope)
        return scope


class Network:
    name: t.Optional[str]

    ctx: context.Context

    _initial_restriction: t.Optional[Expression]

    _links: t.Set[Link]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None) -> None:
        self.ctx = ctx
        self.name = name
        self._initial_restriction = None
        self._links = set()

    @property
    def links(self) -> t.AbstractSet[Link]:
        return self._links

    @property
    def instances(self) -> t.AbstractSet[Instance]:
        return frozenset(
            instance for link in self._links for instance in link.vector.keys()
        )

    @property
    def initial_restriction(self) -> t.Optional[Expression]:
        return self._initial_restriction

    @initial_restriction.setter
    def initial_restriction(self, initial_restriction: Expression) -> None:
        if self._initial_restriction is not None:
            raise errors.InvalidOperationError(
                "restriction of initial valuations has already been set"
            )
        if self.ctx.global_scope.get_type(initial_restriction) != types.BOOL:
            raise errors.InvalidTypeError(
                "restriction of initial valuations must have type `types.BOOL`"
            )
        self._initial_restriction = initial_restriction

    def create_link(
        self,
        vector: t.Mapping[Instance, ActionPattern],
        *,
        result: t.Optional[ActionPattern] = None,
        condition: t.Optional[Expression] = None
    ) -> Link:
        link = Link(vector, result=result, condition=condition)
        self._links.add(link)
        return link
