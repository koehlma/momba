# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses

from . import context, errors, expressions, types
from .automata import Automaton, Instance

if t.TYPE_CHECKING:
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

    ctx: context.Context

    _restrict_initial: t.Optional[Expression]
    _automata: t.Set[Automaton]
    _system: t.Set[Composition]

    def __init__(self, model_type: context.ModelType = context.ModelType.SHA) -> None:
        self.ctx = context.Context(model_type)
        self._restrict_initial = None
        self._automata = set()
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
        return self._automata

    def create_automaton(self, *, name: t.Optional[str] = None) -> Automaton:
        automaton = Automaton(self.ctx, name=name)
        self._automata.add(automaton)
        return automaton

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
            if instance.automaton not in self.automata:
                raise errors.ModelingError(
                    f"automaton {instance.automaton} is not part of the network"
                )
        composition = Composition(frozenset(instances))
        self._system.add(composition)
        return composition
