# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import collections
import dataclasses

from . import effects, context, errors, types

if t.TYPE_CHECKING:
    # XXX: stupid stuff to make mypy and the linter happy
    from . import expressions  # noqa: F401


Action = str


def action(name: str) -> Action:
    return Action(name)


@dataclasses.dataclass(frozen=True, eq=False)
class Instance:
    automaton: Automaton

    input_enable: t.FrozenSet[str] = frozenset()


ProgressInvariant = t.Optional["expressions.Expression"]
TransientValues = t.AbstractSet[effects.Assignment]


@dataclasses.dataclass(frozen=True, eq=False)
class Location:
    """
    Represents a location of a SHA.

    Attributes:
        name:
            The unique name of the location.
        progress_invariant:
            The *time-progression invariant* of the location. Has to be a boolean
            expression in the scope the location is used.
        transient_values:
            Assignments for transient variables.
    """

    name: t.Optional[str] = None
    progress_invariant: ProgressInvariant = None
    transient_values: TransientValues = frozenset()

    def validate(self, scope: context.Scope) -> None:
        if self.progress_invariant is not None:
            if scope.ctx.model_type not in context.TA_MODEL_TYPES:
                raise errors.ModelingError(
                    f"location invariant is not allowed for model type {scope.ctx.model_type}"
                )
            if scope.get_type(self.progress_invariant) != types.BOOL:
                raise errors.InvalidTypeError(
                    f"type of invariant in location {self} is not `types.BOOL`"
                )
        if self.transient_values:
            if scope.ctx.model_type not in context.TA_MODEL_TYPES:
                raise errors.ModelingError(
                    f"transient values are not allowed for model type {scope.ctx.model_type}"
                )
            if not effects.are_compatible(self.transient_values):
                raise errors.IncompatibleAssignmentsError(
                    f"incompatible assignments for transient values"
                )
            for assignment in self.transient_values:
                if assignment.index != 0:
                    raise errors.ModelingError(
                        f"index of assignments for transient values must be zero"
                    )
                assignment.validate(scope)


@dataclasses.dataclass(frozen=True)
class Destination:
    location: Location
    probability: t.Optional[expressions.Expression] = None
    assignments: t.AbstractSet[effects.Assignment] = frozenset()

    def validate(self, scope: context.Scope) -> None:
        if not effects.are_compatible(self.assignments):
            raise errors.IncompatibleAssignmentsError(
                f"assignments on edge {self} are not compatible"
            )
        if self.probability is not None:
            if not scope.get_type(self.probability).is_numeric:
                raise errors.InvalidTypeError(f"probability value must be numeric")
        for assignment in self.assignments:
            assignment.validate(scope)


@dataclasses.dataclass(frozen=True)
class Edge:
    location: Location
    destinations: t.AbstractSet[Destination]
    action: t.Optional[Action] = None
    guard: t.Optional[expressions.Expression] = None
    rate: t.Optional[expressions.Expression] = None

    def validate(self, scope: context.Scope) -> None:
        if self.guard is not None and scope.get_type(self.guard) != types.BOOL:
            raise errors.InvalidTypeError(
                f"type of guard on edge {self} is not `types.BOOL`"
            )
        if self.rate is not None and not scope.get_type(self.rate).is_numeric:
            raise errors.InvalidTypeError(f"type of rate on edge {self} is not numeric")
        for destination in self.destinations:
            destination.validate(scope)


class Automaton:
    ctx: context.Context
    scope: context.Scope

    name: t.Optional[str]

    _locations: t.Set[Location]
    _initial_locations: t.Set[Location]
    _restrict_initial: t.Optional[expressions.Expression]
    _edges: t.Set[Edge]
    _incoming: t.DefaultDict[Location, t.Set[Edge]]
    _outgoing: t.DefaultDict[Location, t.Set[Edge]]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None) -> None:
        self.ctx = ctx
        self.scope = self.ctx.new_scope()
        self.name = name
        self._locations = set()
        self._initial_locations = set()
        self._restrict_initial = None
        self._edges = set()
        self._incoming = collections.defaultdict(set)
        self._outgoing = collections.defaultdict(set)

    @property
    def locations(self) -> t.AbstractSet[Location]:
        return self._locations

    @property
    def initial_locations(self) -> t.AbstractSet[Location]:
        return self._initial_locations

    @property
    def restrict_initial(self) -> t.Optional[expressions.Expression]:
        return self._restrict_initial

    @restrict_initial.setter
    def restrict_initial(self, restrict_initial: expressions.Expression) -> None:
        if self._restrict_initial is not None:
            raise errors.InvalidOperationError(
                f"restriction of initial valuations has already been set"
            )
        if self.scope.get_type(restrict_initial) != types.BOOL:
            raise errors.InvalidTypeError(
                f"restriction of initial valuations must have type `types.BOOL`"
            )
        self._restrict_initial = restrict_initial

    @property
    def edges(self) -> t.AbstractSet[Edge]:
        return self._edges

    def add_location(self, location: Location, *, initial: bool = False) -> None:
        location.validate(self.scope)
        self._locations.add(location)
        if initial:
            self._initial_locations.add(location)

    def create_location(
        self,
        name: t.Optional[str] = None,
        *,
        progress_invariant: t.Optional[expressions.Expression] = None,
        transient_values: t.AbstractSet[effects.Assignment] = frozenset(),
        initial: bool = False,
    ) -> Location:
        """
        Adds a location to the automaton.

        :param location: The :class:`Location` to add.
        """
        location = Location(name, progress_invariant, transient_values)
        self.add_location(location, initial=initial)
        return location

    def add_initial_location(self, location: Location) -> None:
        self.add_location(location, initial=True)

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the automaton.
        """
        edge.validate(self.scope)
        edge.location.validate(self.scope)
        for destination in edge.destinations:
            destination.location.validate(self.scope)
        self._edges.add(edge)
        self._locations.add(edge.location)
        self._outgoing[edge.location].add(edge)
        for destination in edge.destinations:
            self._locations.add(destination.location)
            self._incoming[destination.location].add(edge)

    def create_edge(
        self,
        source: Location,
        destinations: t.AbstractSet[Destination],
        *,
        action: t.Optional[Action] = None,
        guard: t.Optional[expressions.Expression] = None,
        rate: t.Optional[expressions.Expression] = None,
    ) -> None:
        """
        Creates a new edge with the given parameters.

        See :class:`Edge` for more details.
        """
        edge = Edge(source, frozenset(destinations), action, guard, rate)
        self.add_edge(edge)

    def incoming(self, location: Location) -> t.AbstractSet[Edge]:
        """
        Returns the set of outgoing edges from the given location.
        """
        return self._incoming[location]

    def outgoing(self, location: Location) -> t.AbstractSet[Edge]:
        """
        Returns the set of incoming edges to the given location.
        """
        return self._outgoing[location]

    def declare_variable(self, name: str, typ: types.Type) -> None:
        """
        Declares a variable in the local scope of the automaton.
        """
        self.scope.declare_variable(name, typ)

    def create_instance(
        self, *, input_enable: t.AbstractSet[str] = frozenset()
    ) -> Instance:
        """
        Creates an instance of the automaton for composition.
        """
        return Instance(self, input_enable=frozenset(input_enable))


Assignments = t.Union[
    t.AbstractSet[effects.Assignment], t.Mapping[str, "expressions.Expression"]
]


def create_destination(
    location: Location,
    *,
    probability: t.Optional[expressions.Expression] = None,
    assignments: Assignments = frozenset(),
) -> Destination:
    if isinstance(assignments, t.Mapping):
        return Destination(
            location,
            probability,
            assignments=frozenset(
                effects.Assignment(effects.Identifier(name), value)
                for name, value in assignments.items()
            ),
        )
    else:
        return Destination(location, probability, assignments)
