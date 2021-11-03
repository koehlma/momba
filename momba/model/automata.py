# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import collections

from mxu.maps import FrozenMap

from . import actions, errors, expressions, observations, types

if t.TYPE_CHECKING:
    from . import context


Annotation = t.Mapping[str, t.Union[int, str, float]]


@d.dataclass(frozen=True, eq=False)
class Instance:
    """
    Represents an automaton instance.

    Attributes
    ----------
    automaton:
        The instaniated automaton.
    arguments:
        The arguments passed to the parameters of the automaton.
    input_enabled:
        The set of action types for which the instance is input enabled.
    comment:
        An optional comment describing the instance.
    """

    automaton: Automaton
    arguments: t.Tuple[expressions.Expression, ...] = ()
    input_enable: t.FrozenSet[actions.ActionType] = frozenset()
    comment: t.Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.arguments) != len(self.automaton.parameters):
            raise errors.ModelingError("invalid number of arguments for automaton")


ProgressInvariant = t.Optional[expressions.Expression]
TransientValues = t.AbstractSet["Assignment"]


@d.dataclass(frozen=True, eq=False)
class Location:
    """
    Represents a location of an automaton.

    Attributes:
        name:
            The optional name of the location.
        progress_invariant:
            An optional expression `progress_invariant` specifies the invariant
            subject to which the time may progress in the location. As per the
            JANI specification, the invariant is only allowed for models using
            real-valued clocks (see :class:`~momba.model.ModelType`).
        transient_values:
            A set of assignments for transient variables. These assignments
            define the values of the transient variables when in the location.
    """

    name: t.Optional[str] = None

    progress_invariant: ProgressInvariant = None
    transient_values: TransientValues = frozenset()

    def validate(self, automaton: Automaton) -> None:
        """
        Validates the location for the given automaton.

        Raises :class:`~errors.ModelingError` if the location is invalid
        to add to the automaton according to the JANI specification. For
        instance, if the location has a `progress_invariant` but the model
        type does not support clocks.
        """
        scope = automaton.scope
        if self.progress_invariant is not None:
            if not scope.ctx.model_type.uses_clocks:
                raise errors.ModelingError(
                    f"location invariant is not allowed for model type {scope.ctx.model_type}"
                )
            if scope.get_type(self.progress_invariant) != types.BOOL:
                raise errors.InvalidTypeError(
                    f"type of invariant in location {self} is not `types.BOOL`"
                )
        if self.transient_values:
            if not are_compatible(self.transient_values):
                raise errors.IncompatibleAssignmentsError(
                    f"incompatible assignments for transient values in location {self}"
                )
            for assignment in self.transient_values:
                if assignment.index != 0:
                    raise errors.ModelingError(
                        f"index of assignment {assignment} to transient variable is non-zero"
                    )
                assignment.validate(scope)


@d.dataclass(frozen=True)
class Assignment:
    """
    Represents an assignment.

    Attributes
    ----------
    target:
        The target of the assignment.
    value:
        The value to assign.
    index:
        The index of the assignment.
    """

    target: expressions.Expression
    value: expressions.Expression
    index: int = 0

    def validate(self, scope: context.Scope) -> None:
        """
        Validates the assignment in the given scope.

        Raises :class:`~errors.ModelingError` if the target is not
        assignable from the value type.
        """
        target_type = self.target.infer_target_type(scope)
        value_type = scope.get_type(self.value)
        if not target_type.is_assignable_from(value_type):
            raise errors.InvalidTypeError(
                f"cannot assign {value_type} to {target_type}"
            )


def are_compatible(assignments: t.Iterable[Assignment]) -> bool:
    """
    Checks whether the given assignments are compatible according
    to the JANI specification.
    """
    groups: t.DefaultDict[int, t.Set[expressions.Expression]] = collections.defaultdict(
        set
    )
    for assignment in assignments:
        target = assignment.target
        if target in groups[assignment.index]:
            return False
        groups[assignment.index].add(target)
    return True


@d.dataclass(frozen=True)
class Destination:
    """
    Represents a destination of an edge.

    Attributes
    ----------
    location:
        The target location of the destination.
    probability:
        An optional expression for the probability of the destination.
    assignments:
        A set of assignments to be executed when going to the destination.
    """

    location: Location
    probability: t.Optional[expressions.Expression] = None
    assignments: t.Tuple[Assignment, ...] = ()

    def _validate(self, automaton: Automaton, scope: context.Scope) -> None:
        self.location.validate(automaton)
        if self.location not in automaton.locations:
            raise errors.ModelingError(
                f"source location of edge {self} is not a location of the automaton {automaton}"
            )
        if not are_compatible(self.assignments):
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
    """
    Represents an edge of an automaton.

    Attributes
    ----------
    location:
        The source location of the edge.
    destinations:
        The destinations of the edge.
    action_pattern:
        The optional action pattern of the edge.
    guard:
        The optional guard of the edge.
    rate:
        The optional rate of the edge.
    annotation:
        An optional annotation of the edge.
    """

    location: Location
    destinations: t.Tuple[Destination, ...]
    action_pattern: t.Optional[actions.ActionPattern] = None
    guard: t.Optional[expressions.Expression] = None
    rate: t.Optional[expressions.Expression] = None
    annotation: t.Optional[Annotation] = None
    observation: t.FrozenSet[observations.Observation] = frozenset()

    def create_edge_scope(self, parent: context.Scope) -> context.Scope:
        """
        Creates an *edge scope* with the given parent scope.

        .. warning::
            Used for *value passing* an experimental Momba feature. Value
            passing is not part of the official JANI specification.
        """
        scope = parent.create_child_scope()
        if self.action_pattern is not None:
            self.action_pattern.declare_in(scope)
        return scope

    def validate(self, automaton: Automaton) -> None:
        """
        Validates the edge for the given automaton.

        Raises :class:`~errors.ModelingError` if the edge is invalid to add
        to the automaton according to the JANI specification. For instance, if
        source location is not a location of the automaton.
        """
        scope = automaton.scope
        self.location.validate(automaton)
        if self.location not in automaton.locations:
            raise errors.ModelingError(
                f"source location of edge {self} is not a location of the automaton {automaton}"
            )
        if self.rate is not None and not scope.get_type(self.rate).is_numeric:
            raise errors.InvalidTypeError(f"type of rate on edge {self} is not numeric")
        edge_scope = self.create_edge_scope(scope)
        if self.guard is not None and edge_scope.get_type(self.guard) != types.BOOL:
            raise errors.InvalidTypeError(
                f"type of guard on edge {self} is not `types.BOOL`"
            )
        for destination in self.destinations:
            destination._validate(automaton, edge_scope)


class Automaton:
    """
    Represents an automaton.

    Attributes
    ----------
    ctx:
        The :class:`Context` associated with the automaton.
    scope:
        The local :class:`Scope` of the automaton.
    name:
        The optional name of the automaton.
    """

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
        return f"<Automaton name={self.name!r} at 0x{id(self):x}>"

    @property
    def locations(self) -> t.AbstractSet[Location]:
        """
        The set of locations of the automaton.
        """
        return self._locations

    @property
    def initial_locations(self) -> t.AbstractSet[Location]:
        """
        The set of initial locations of the automaton.
        """
        return self._initial_locations

    @property
    def edges(self) -> t.Sequence[Edge]:
        """
        The set of edges of the automaton.
        """
        return self._edges

    @property
    def initial_restriction(self) -> t.Optional[expressions.Expression]:
        """
        An optional restriction to be satisfied by initial states.

        This property can be set *only once* per automaton. The expression
        has to be boolean.

        Raises :class:`~errors.ModelingError` when set twice or to a
        non-boolean expression.
        """
        return self._initial_restriction

    @initial_restriction.setter
    def initial_restriction(self, initial_restriction: expressions.Expression) -> None:
        if self._initial_restriction is not None:
            raise errors.ModelingError(
                "restriction of initial environment has already been set"
            )
        if self.scope.get_type(initial_restriction) != types.BOOL:
            raise errors.ModelingError(
                "restriction of initial valuations must have type `types.BOOL`"
            )
        self._initial_restriction = initial_restriction

    @property
    def parameters(self) -> t.Sequence[str]:
        """
        The sequence of parameters of the automaton.
        """
        return self._parameters

    def add_location(self, location: Location, *, initial: bool = False) -> None:
        """
        Adds a location to the automaton.

        The flag `initial` specifies whether the location
        is an initial location.

        Raises :class:`~errors.ModelingError` when the location cannot be added
        to the automaton. See the method :meth:`Location.validate` for details.
        """
        location.validate(self)
        self._locations.add(location)
        if initial:
            self._initial_locations.add(location)

    def create_location(
        self,
        name: t.Optional[str] = None,
        *,
        progress_invariant: t.Optional[expressions.Expression] = None,
        transient_values: t.AbstractSet[Assignment] = frozenset(),
        initial: bool = False,
    ) -> Location:
        """
        Creates a location with the given name and adds it to the automaton.

        The optional expression `progress_invariant` specifies the invariant
        subject to which the time may progress in the location. As per the
        JANI specification, the invariant is only allowed for models using
        real-valued clocks (see :class:`~momba.model.ModelType`).

        The parameter `transient_values` should be a set of assignments for
        transient variables. These assignments define the values of the
        transient variables when in the location.

        The flag `initial` specifies whether the location
        is an initial location.

        Raises the same exceptions as :meth:`add_location`.
        """
        location = Location(name, progress_invariant, transient_values)
        self.add_location(location, initial=initial)
        return location

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the automaton.

        Raises :class:`~errors.ModelingError` when the edge cannot be added
        to the automaton. See the method :meth:`Edge.validate` for details.
        """
        edge.validate(self)
        self._edges.append(edge)
        self._locations.add(edge.location)
        self._outgoing_edges[edge.location].add(edge)
        for destination in edge.destinations:
            self._locations.add(destination.location)
            self._incoming_edges[destination.location].add(edge)

    def create_edge(
        self,
        source: Location,
        destinations: t.Iterable[Destination],
        *,
        action_pattern: t.Optional[actions.ActionPattern] = None,
        guard: t.Optional[expressions.Expression] = None,
        rate: t.Optional[expressions.Expression] = None,
        annotation: t.Optional[Annotation] = None,
        observations: t.AbstractSet[observations.Observation] = frozenset(),
    ) -> None:
        """
        Creates an edge and adds it to the automaton.

        The parameters are passed to :class:`Edge`.

        Raises the same exceptions as :meth:`add_edge`.
        """
        edge = Edge(
            source,
            tuple(destinations),
            action_pattern,
            guard,
            rate,
            FrozenMap(annotation),
            frozenset(observations),
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

        Passes the parameters to :meth:`Scope.declare_variable`.
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
        Declarse a parameter of the automaton.

        Passes the parameters to :meth:`Scope.declare_constant` where
        `value` is `default_value`.
        """
        self.scope.declare_constant(name, typ, value=default_value)
        self._parameters.append(name)

    def create_instance(
        self,
        *,
        arguments: t.Sequence[expressions.ValueOrExpression] = (),
        input_enable: t.AbstractSet[actions.ActionType] = frozenset(),
        comment: t.Optional[str] = None,
    ) -> Instance:
        """
        Creates an instance of the automaton for composition.

        Passes the parameters to :class:`Instance`.
        """
        return Instance(
            self,
            arguments=tuple(map(expressions.ensure_expr, arguments)),
            input_enable=frozenset(input_enable),
        )


Assignments = t.Union[
    t.Iterable["Assignment"], t.Mapping[str, "expressions.Expression"]
]


def create_destination(
    location: Location,
    *,
    probability: t.Optional[expressions.Expression] = None,
    assignments: Assignments = (),
) -> Destination:
    """
    Creates a destination with the given target location.

    This is a convenience function for creating destinations where
    assignments can be provied as a mapping. We recommend using it
    instead of creating :class:`Destination` objects directly.
    """
    if isinstance(assignments, t.Mapping):
        return Destination(
            location,
            probability,
            assignments=tuple(
                Assignment(expressions.Name(name), value)
                for name, value in assignments.items()
            ),
        )
    else:
        return Destination(location, probability, tuple(assignments))
