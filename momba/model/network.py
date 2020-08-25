# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses

from . import errors, types
from .automata import Instance

if t.TYPE_CHECKING:
    from .action import ActionPattern
    from . import context
    from .expressions import Expression


@dataclasses.dataclass(frozen=True, eq=False)
class Synchronization:
    vector: t.Mapping[Instance, ActionPattern]
    result: t.Optional[ActionPattern] = None
    condition: t.Optional[Expression] = None

    def construct_scope(self, ctx: context.Context) -> context.Scope:
        scope = ctx.global_scope.create_child_scope()
        for pattern in self.vector.values():
            pattern.declare_in(scope)
        return scope


@dataclasses.dataclass(frozen=True, eq=False)
class Composition:
    instances: t.FrozenSet[Instance]
    synchronizations: t.Set[Synchronization] = dataclasses.field(default_factory=set)

    def create_synchronization(
        self,
        vector: t.Mapping[Instance, ActionPattern],
        *,
        result_pattern: t.Optional[ActionPattern] = None,
    ) -> None:
        for instance in vector.keys():
            if instance not in self.instances:
                raise errors.ModelingError(
                    f"instance {instance} is not part of composition"
                )
        self.synchronizations.add(Synchronization(vector, result_pattern))


class Network:
    """
    The core class representing a network of interacting SHAs.
    """

    name: t.Optional[str]

    ctx: context.Context

    _initial_restriction: t.Optional[Expression]
    _system: t.Set[Composition]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None,) -> None:
        self.ctx = ctx
        self.name = name
        self._initial_restriction = None
        self._system = set()

    @property
    def system(self) -> t.AbstractSet[Composition]:
        return self._system

    @property
    def links(self) -> t.AbstractSet[Synchronization]:
        return frozenset(
            link
            for composition in self._system
            for link in composition.synchronizations
        )

    @property
    def instances(self) -> t.AbstractSet[Instance]:
        return frozenset(
            instance
            for composition in self._system
            for instance in composition.instances
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

    def create_composition(self, instances: t.AbstractSet[Instance]) -> Composition:
        for instance in instances:
            if instance.automaton not in self.ctx.automata:
                raise errors.ModelingError(
                    f"automaton {instance.automaton} is not part of the network"
                )
        composition = Composition(frozenset(instances))
        self._system.add(composition)
        return composition
