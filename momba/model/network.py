# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses

from . import errors, expressions, types
from .automata import Automaton, Instance

if t.TYPE_CHECKING:
    from . import context
    from .expressions import Expression


@dataclasses.dataclass(frozen=True, eq=False)
class Synchronization:
    vector: t.Mapping[Instance, str]
    result: t.Optional[str] = None


@dataclasses.dataclass(frozen=True, eq=False)
class Composition:
    instances: t.FrozenSet[Instance]
    synchronizations: t.Set[Synchronization] = dataclasses.field(default_factory=set)

    def create_synchronization(
        self, vector: t.Mapping[Instance, str], result: t.Optional[str] = None
    ) -> None:
        for instance in vector.keys():
            if instance not in self.instances:
                raise errors.ModelingError(
                    f"instance {instance} is not part of composition"
                )
        self.synchronizations.add(Synchronization(vector, result))


class Network:
    """
    The core class representing a network of interacting SHAs.
    """

    name: t.Optional[str]

    ctx: context.Context

    _restrict_initial: t.Optional[Expression]
    _system: t.Set[Composition]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None,) -> None:
        self.ctx = ctx
        self.name = name
        self._restrict_initial = None
        self._system = set()

    @property
    def system(self) -> t.AbstractSet[Composition]:
        return self._system

    @property
    def restrict_initial(self) -> t.Optional[Expression]:
        return self._restrict_initial

    @restrict_initial.setter
    def restrict_initial(self, restrict_initial: Expression) -> None:
        if self._restrict_initial is not None:
            raise errors.InvalidOperationError(
                f"restriction of initial valuations has already been set"
            )
        if self.ctx.global_scope.get_type(restrict_initial) != types.BOOL:
            raise errors.InvalidTypeError(
                f"restriction of initial valuations must have type `types.BOOL`"
            )
        self._restrict_initial = restrict_initial

    @property
    def automata(self) -> t.AbstractSet[Automaton]:
        """
        The set of :py_class:`momba.Automaton` making up the model.
        """
        return self.ctx.automata

    def create_automaton(self, *, name: t.Optional[str] = None) -> Automaton:
        return self.ctx.create_automaton(name=name)

    def declare_variable(self, identifier: str, typ: types.Type) -> None:
        self.ctx.global_scope.declare_variable(identifier, typ)

    def declare_constant(
        self,
        identifier: str,
        typ: types.Type,
        value: t.Optional[expressions.MaybeExpression] = None,
    ) -> None:
        if value is None:
            self.ctx.global_scope.declare_constant(identifier, typ, None)
        else:
            self.ctx.global_scope.declare_constant(
                identifier, typ, expressions.convert(value)
            )

    def create_composition(self, instances: t.AbstractSet[Instance]) -> Composition:
        for instance in instances:
            if instance.automaton not in self.ctx.automata:
                if instance.automaton not in self.automata:
                    raise errors.ModelingError(
                        f"automaton {instance.automaton} is not part of the network"
                    )
        composition = Composition(frozenset(instances))
        self._system.add(composition)
        return composition
