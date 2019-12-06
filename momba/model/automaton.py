# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses
import typing

from . import assignments, context, errors, expressions, types


@dataclasses.dataclass(frozen=True)
class Location:
    name: str
    invariant: expressions.Expression = expressions.cast(True)

    def validate(self, scope: context.Scope) -> None:
        if scope.get_type(self.invariant) != types.BOOL:
            raise errors.InvalidTypeError(
                f'type of invariant in location {self} is not `types.BOOL`'
            )


@dataclasses.dataclass(frozen=True)
class Destination:
    location: Location
    assignments: typing.AbstractSet[assignments.Assignment] = dataclasses.field(
        default_factory=frozenset
    )

    def validate(self, scope: context.Scope) -> None:
        if not assignments.are_compatible(self.assignments):
            raise errors.IncompatibleAssignmentsError(
                f'assignments on edge {self} are not compatible'
            )
        for assignment in self.assignments:
            assignment.validate(scope)


@dataclasses.dataclass(frozen=True)
class Edge:
    location: Location
    guard: expressions.Expression
    destinations: typing.AbstractSet[Destination]

    def validate(self, scope: context.Scope) -> None:
        if scope.get_type(self.guard) != types.BOOL:
            raise errors.InvalidTypeError(
                f'type of guard on edge {self} is not `types.BOOL`'
            )
        for destination in self.destinations:
            destination.validate(scope)


class Automaton:
    ctx: context.Context
    scope: context.Scope

    comment: typing.Optional[str]

    _locations: typing.Set[Location]
    _initial_locations: typing.Set[Location]
    _restrict_initial: typing.Optional[expressions.Expression]
    _edges: typing.Set[Edge]

    def __init__(self, ctx: typing.Optional[context.Context] = None):
        self.ctx = ctx or context.Context()
        self.scope = self.ctx.new_scope()
        self.comment = None
        self._locations = set()
        self._initial_locations = set()
        self._restrict_initial = None
        self._edges = set()

    @property
    def locations(self) -> typing.AbstractSet[Location]:
        return self._locations

    @property
    def initial_locations(self) -> typing.AbstractSet[Location]:
        return self._initial_locations

    @property
    def restrict_initial(self) -> typing.Optional[expressions.Expression]:
        return self._restrict_initial

    @restrict_initial.setter
    def restrict_initial(self, restrict_initial: expressions.Expression) -> None:
        if self._restrict_initial is not None:
            raise errors.InvalidOperationError(
                f'restriction of initial valuations has already been set'
            )
        if self.scope.get_type(restrict_initial) != types.BOOL:
            raise errors.InvalidTypeError(
                f'restriction of initial valuations must have type `types.BOOL`'
            )
        self._restrict_initial = restrict_initial

    @property
    def edges(self) -> typing.AbstractSet[Edge]:
        return self._edges

    def add_location(self, location: Location) -> None:
        location.validate(self.scope)
        self._locations.add(location)

    def add_initial_location(self, location: Location) -> None:
        self.add_location(location)
        self._initial_locations.add(location)

    def add_edge(self, edge: Edge) -> None:
        edge.validate(self.scope)
        edge.location.validate(self.scope)
        for destination in edge.destinations:
            destination.location.validate(self.scope)
        self._edges.add(edge)
        self._locations.add(edge.location)
        for destination in edge.destinations:
            self._locations.add(destination.location)
