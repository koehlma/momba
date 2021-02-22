# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import json
import functools

from .. import model
from ..model import expressions
from ..utils.distribution import Distribution

from .time import DiscreteTime, TimeType
from .translator import Translation, translate_network
from .values import Value


TimeTypeT = t.TypeVar("TimeTypeT", bound=TimeType)


@d.dataclass(frozen=True)
class Action:
    """
    Represents an action.

    The *arguments* of an action are usually empty. They are used for
    the experimental value-passing feature of Momba which has not been
    stabilized yet.

    Attributes
    ----------
    action_type:
        The :class:`~momba.model.ActionType` of the action.
    arguments:
        The arguments of the action (a tuple of values).
    """

    action_type: model.ActionType
    arguments: t.Tuple[Value, ...]


@d.dataclass(frozen=True, repr=False)
class Destination(t.Generic[TimeTypeT]):
    """
    Represents a destination of a transition.

    Attributes
    ----------
    explorer:
        The :class:`Explorer` associated with the destination.
    """

    explorer: Explorer[TimeTypeT]

    _state: t.Any
    _transition: t.Any
    _destination: t.Any

    @property
    def probability(self) -> float:
        """
        The probability associated with the destination.
        """
        return self._destination.probability()

    @functools.cached_property
    def state(self) -> State[TimeTypeT]:
        """
        The target :class:`State` associated with the destination.
        """
        return State(
            self.explorer,
            self._destination.successor(self._state, self._transition),
        )


def _action(action: t.Any, explorer: Explorer[TimeTypeT]) -> t.Optional[Action]:
    if action.is_silent():
        return None
    label = action.label()
    assert isinstance(label, str)
    arguments = action.arguments()
    return Action(
        explorer.network.ctx.get_action_type_by_name(label),
        tuple(arguments),
    )


@d.dataclass(frozen=True, repr=False)
class Transition(t.Generic[TimeTypeT]):
    """
    Represents a joint transition of an automaton network.

    Attributes
    ----------
    explorer:
        The :class:`Explorer` associated with the transition.
    instances:
        The automaton instances participating in the transition.
    action:
        The action associated with the transition.
    action_vector:
        The actions with which the respective instances participate.

        Is a mapping from instances to actions.
    edge_vector:
        The edges with which the respective instances participate.

        Is a mapping from instances to edges.
    destinations:
        The destinations of the transition.
    """

    explorer: Explorer[TimeTypeT]
    source: State[TimeTypeT]

    _state: t.Any
    _transition: t.Any

    @property
    def instances(self) -> t.AbstractSet[model.Instance]:
        instances = set()
        for edge_reference in json.loads(self._transition.edge_vector()):
            automaton_name = edge_reference["location"]["automaton"]["name"]
            instance = self.explorer._translation.instance_name_to_instance[
                automaton_name
            ]
            instances.add(instance)
        return instances

    @property
    def action(self) -> t.Optional[Action]:
        action = self._transition.result_action()
        if action.is_silent():
            return None
        label = action.label()
        assert isinstance(label, str)
        arguments = action.arguments()
        return Action(
            self.explorer.network.ctx.get_action_type_by_name(label),
            tuple(arguments),
        )

    @property
    def action_vector(self) -> t.Mapping[model.Instance, t.Optional[Action]]:
        action_vector = {}
        for edge_reference, action in zip(
            json.loads(self._transition.edge_vector()), self._transition.action_vector()
        ):
            automaton_name = edge_reference["location"]["automaton"]["name"]
            instance = self.explorer._translation.instance_name_to_instance[
                automaton_name
            ]
            action_vector[instance] = _action(action, self.explorer)
        return action_vector

    @property
    def edge_vector(self) -> t.Mapping[model.Instance, model.Edge]:
        edge_vector = {}
        for edge_reference in json.loads(self._transition.edge_vector()):
            automaton_name = edge_reference["location"]["automaton"]["name"]
            location_name = edge_reference["location"]["name"]
            edge_index = edge_reference["index"]
            instance = self.explorer._translation.instance_name_to_instance[
                automaton_name
            ]
            location = self.explorer._translation.reversed_instance_to_location_names[
                instance
            ][location_name]
            location_edges = instance.automaton.get_outgoing_edges(location)
            counter = 0
            for edge in instance.automaton.edges:
                if edge not in location_edges:
                    continue
                if counter == edge_index:
                    edge_vector[instance] = edge
                    break
                else:
                    counter += 1
        return edge_vector

    @functools.cached_property
    def destinations(self) -> Distribution[Destination[TimeTypeT]]:
        destinations = tuple(
            Destination(self.explorer, self._state, self._transition, destination)
            for destination in self._transition.destinations(self._state)
        )
        return Distribution(
            {destination: destination.probability for destination in destinations}
        )


@d.dataclass(frozen=True, repr=False)
class State(t.Generic[TimeTypeT]):
    """
    Represents a state of an automaton network.

    Attributes
    ----------
    explorer:
        The :class:`Explorer` associated with the state.
    """

    explorer: Explorer[TimeTypeT]

    _state: t.Any

    @functools.cached_property
    def global_env(self) -> t.Mapping[str, Value]:
        """
        The global environment, i.e., a mapping from global variables to values.
        """
        return {
            name: Value(self._state.get_global_value(declaration.identifier))
            for name, declaration in self.explorer._translation.declarations.globals_table.items()
            if not declaration.is_transient
        }

    def get_local_env(self, instance: model.Instance) -> t.Mapping[str, Value]:
        """
        Returns the local environment of the provided automaton instance.
        """
        return {
            name: Value(self._state.get_global_value(declaration.identifier))
            for name, declaration in self.explorer._translation.declarations.locals_table[
                instance
            ].items()
        }

    @functools.cached_property
    def locations(self) -> t.Mapping[model.Instance, model.Location]:
        """
        The locations of the respective automata instances.

        A mapping from instances to locations.
        """
        return {
            instance: self.explorer._translation.reversed_instance_to_location_names[
                instance
            ][self._state.get_location_of(name)]
            for instance, name in self.explorer._translation.instance_names.items()
        }

    @functools.cached_property
    def transitions(self) -> t.Sequence[Transition[TimeTypeT]]:
        """
        Outgoing transitions of the state.
        """
        return tuple(
            Transition(self.explorer, self, self._state, transition)
            for transition in self._state.transitions()
        )

    @property
    def valuations(self) -> TimeTypeT:
        raise NotImplementedError()


class Explorer(t.Generic[TimeTypeT]):
    """
    Main interface to the state space exploration engine.

    .. warning::
        A network must not be modified once an explorer has been created
        for it.
        Modifying the network nonetheless may lead to all kinds of
        unspecified behavior.

    Paramaters
    ----------

    Attributes
    ----------
    network:
        The :class:`~momba.model.Network` the explorer has been created for.
    time_type:
        The :class:`TimeType` of the explorer.
    """

    network: model.Network
    time_type: t.Type[TimeTypeT]

    _translation: Translation
    _compiled: t.Any

    def __init__(
        self,
        network: model.Network,
        time_type: t.Type[TimeTypeT],
        *,
        parameters: t.Optional[t.Mapping[str, expressions.ValueOrExpression]] = None,
    ) -> None:
        self.network = network
        self.time_type = time_type
        self._translation = translate_network(
            network,
            parameters={
                name: expressions.ensure_expr(expr)
                for name, expr in (parameters or {}).items()
            },
        )
        self._compiled = self.time_type.compile(network, self._translation)

    @staticmethod
    def new_discrete_time(
        network: model.Network,
        *,
        parameters: t.Optional[t.Mapping[str, expressions.ValueOrExpression]] = None,
    ) -> Explorer[DiscreteTime]:
        """
        Creates a new discrete time explorer.
        """
        return Explorer(network, DiscreteTime, parameters=parameters)

    @functools.cached_property
    def initial_states(self) -> t.AbstractSet[State[TimeTypeT]]:
        """
        The initial states of the network.
        """
        return frozenset(
            State(self, state) for state in self._compiled.initial_states()
        )
