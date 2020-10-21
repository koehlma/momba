# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import collections

from mxu.maps import FrozenMap

from . import actions, effects, errors, expressions, types

if t.TYPE_CHECKING:
    from . import context


Annotation = t.Mapping[str, t.Union[int, str, float]]


@d.dataclass(frozen=True, eq=False)
class Instance:
    automaton: Automaton

    arguments: t.Tuple[expressions.Expression, ...] = ()

    input_enable: t.FrozenSet[str] = frozenset()


ProgressInvariant = t.Optional[expressions.Expression]
TransientValues = t.AbstractSet["effects.Assignment"]


@d.dataclass(frozen=True, eq=False)
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
            if not scope.ctx.model_type.is_timed:
                raise errors.ModelingError(
                    f"location invariant is not allowed for model type {scope.ctx.model_type}"
                )
            if scope.get_type(self.progress_invariant) != types.BOOL:
                raise errors.InvalidTypeError(
                    f"type of invariant in location {self} is not `types.BOOL`"
                )
        if self.transient_values:
            if not scope.ctx.model_type.is_timed:
                raise errors.ModelingError(
                    f"transient values are not allowed for model type {scope.ctx.model_type}"
                )
            if not effects.are_compatible(self.transient_values):
                raise errors.IncompatibleAssignmentsError(
                    "incompatible assignments for transient values"
                )
            for assignment in self.transient_values:
                if assignment.index != 0:
                    raise errors.ModelingError(
                        "index of assignments for transient values must be zero"
                    )
                assignment.validate(scope)


@d.dataclass(frozen=True)
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
                raise errors.InvalidTypeError("probability value must be numeric")
        for assignment in self.assignments:
            assignment.validate(scope)


@d.dataclass(frozen=True)
class Edge:
    location: Location
    destinations: t.AbstractSet[Destination]
    action_pattern: t.Optional[actions.ActionPattern] = None
    guard: t.Optional[expressions.Expression] = None
    rate: t.Optional[expressions.Expression] = None
    annotation: t.Optional[Annotation] = None

    def create_edge_scope(self, parent: context.Scope) -> context.Scope:
        scope = parent.create_child_scope()
        if self.action_pattern is not None:
            self.action_pattern.declare_in(scope)
        return scope

    def validate(self, scope: context.Scope) -> None:
        if self.rate is not None and not scope.get_type(self.rate).is_numeric:
            raise errors.InvalidTypeError(f"type of rate on edge {self} is not numeric")
        edge_scope = self.create_edge_scope(scope)
        if self.guard is not None and edge_scope.get_type(self.guard) != types.BOOL:
            raise errors.InvalidTypeError(
                f"type of guard on edge {self} is not `types.BOOL`"
            )
        for destination in self.destinations:
            destination.validate(edge_scope)


class Automaton:
    ctx: context.Context
    scope: context.Scope

    name: t.Optional[str]

    _parameters: t.List[str]

    _locations: t.Set[Location]
    _initial_locations: t.Set[Location]
    _initial_restriction: t.Optional[expressions.Expression]
    _edges: t.List[Edge]
    _incoming_edges: t.DefaultDict[Location, t.Set[Edge]]
    _outgoing_edges: t.DefaultDict[Location, t.Set[Edge]]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None) -> None:
        self.ctx = ctx
        self.scope = self.ctx.global_scope.create_child_scope()
        self.name = name
        self._parameters = []
        self._locations = set()
        self._initial_locations = set()
        self._initial_restriction = None
        self._edges = []
        self._incoming_edges = collections.defaultdict(set)
        self._outgoing_edges = collections.defaultdict(set)

    def __repr__(self) -> str:
        return f"<Automaton @ 0x{id(self):X} name={self.name!r}>"

    @property
    def parameters(self) -> t.Sequence[str]:
        return self._parameters

    @property
    def locations(self) -> t.AbstractSet[Location]:
        return self._locations

    @property
    def initial_locations(self) -> t.AbstractSet[Location]:
        return self._initial_locations

    @property
    def initial_restriction(self) -> t.Optional[expressions.Expression]:
        return self._initial_restriction

    @initial_restriction.setter
    def initial_restriction(self, initial_restriction: expressions.Expression) -> None:
        if self._initial_restriction is not None:
            raise errors.InvalidOperationError(
                "restriction of initial valuations has already been set"
            )
        if self.scope.get_type(initial_restriction) != types.BOOL:
            raise errors.InvalidTypeError(
                "restriction of initial valuations must have type `types.BOOL`"
            )
        self._initial_restriction = initial_restriction

    @property
    def edges(self) -> t.Sequence[Edge]:
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
        if edge.action_pattern is not None:
            self.ctx.add_action_type(edge.action_pattern.action_type)
        edge.validate(self.scope)
        edge.location.validate(self.scope)
        for destination in edge.destinations:
            destination.location.validate(self.scope)
        self._edges.append(edge)
        self._locations.add(edge.location)
        self._outgoing_edges[edge.location].add(edge)
        for destination in edge.destinations:
            self._locations.add(destination.location)
            self._incoming_edges[destination.location].add(edge)

    def create_edge(
        self,
        source: Location,
        destinations: t.AbstractSet[Destination],
        *,
        action_pattern: t.Optional[actions.ActionPattern] = None,
        guard: t.Optional[expressions.Expression] = None,
        rate: t.Optional[expressions.Expression] = None,
        annotation: t.Optional[Annotation] = None,
    ) -> None:
        """
        Creates a new edge with the given parameters.

        See :class:`Edge` for more details.
        """
        edge = Edge(
            source,
            frozenset(destinations),
            action_pattern,
            guard,
            rate,
            FrozenMap(annotation),
        )
        self.add_edge(edge)

    def get_incoming_edges(self, location: Location) -> t.AbstractSet[Edge]:
        """
        Returns the set of outgoing edges from the given location.
        """
        return self._incoming_edges[location]

    def get_outgoing_edges(self, location: Location) -> t.AbstractSet[Edge]:
        """
        Returns the set of incoming edges to the given location.
        """
        return self._outgoing_edges[location]

    def declare_variable(
        self,
        name: str,
        typ: types.Type,
        *,
        is_transient: t.Optional[bool] = None,
        initial_value: t.Optional[expressions.Expression] = None,
    ) -> None:
        """
        Declares a variable in the local scope of the automaton.
        """
        self.scope.declare_variable(
            name, typ, is_transient=is_transient, initial_value=initial_value
        )

    def declare_parameter(
        self,
        name: str,
        typ: types.Type,
        *,
        default_value: t.Optional[expressions.Expression] = None,
    ) -> None:
        """
        Declarse a parameter for the automaton.
        """
        self.scope.declare_constant(name, typ, value=default_value)
        self._parameters.append(name)

    def create_instance(
        self,
        *,
        parameters: t.Sequence[expressions.Expression] = (),
        input_enable: t.AbstractSet[str] = frozenset(),
    ) -> Instance:
        """
        Creates an instance of the automaton for composition.
        """
        assert len(parameters) == len(self.parameters)
        return Instance(
            self, arguments=tuple(parameters), input_enable=frozenset(input_enable)
        )


Assignments = t.Union[
    t.AbstractSet["effects.Assignment"], t.Mapping[str, "expressions.Expression"]
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
                effects.Assignment(effects.Name(name), value)
                for name, value in assignments.items()
            ),
        )
    else:
        return Destination(location, probability, assignments)
