# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import functools

from .. import model
from ..model import expressions
from ..errors import InvalidModelType
from ..utils.distribution import Distribution

from .time import DiscreteTime, TimeType
from .translator import Translation, translate_network
from .values import Value

try:
    import momba_engine as _engine
except ImportError:
    raise ImportError(
        "Missing optional dependency `momba_engine`.\n"
        "Using Momba's engine requires installing `momba_engine`."
    )


TimeTypeT = t.TypeVar("TimeTypeT", bound=TimeType)


@d.dataclass(frozen=True)
class Action:
    """
    Represents an action.

    The arguments of an action are usually empty.
    They are used for the experimental value-passing feature
    of Momba which has not been stabilized yet.

    Attributes
    ----------
    action_type:
        The :class:`~momba.model.ActionType` of the action.
    arguments:
        The arguments of the action.
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
    probability
        The probability associated with the destination.
    state
        The target :class:`State` associated with the destination.
    """

    explorer: Explorer[TimeTypeT]

    _state: t.Any
    _transition: t.Any
    _destination: t.Any

    @property
    def probability(self) -> float:
        return self._destination.probability()

    @functools.cached_property
    def state(self) -> State[TimeTypeT]:
        return State(
            self.explorer,
            self._destination.successor(self._state, self._transition),
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
        raise NotImplementedError()

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
    def action_vector(self) -> t.Mapping[model.Instance, Action]:
        raise NotImplementedError()

    @property
    def edge_vector(self) -> t.Mapping[model.Instance, model.Edge]:
        raise NotImplementedError()

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
    global_env:
        The global environment, i.e., a mapping from global variables
        to values.
    locations:
        The locations of the respective automata instances.

        A mapping from instances to locations.
    transitions:
        Outgoing transitions of the state.
    """

    explorer: Explorer[TimeTypeT]

    _state: t.Any

    @functools.cached_property
    def global_env(self) -> t.Mapping[str, Value]:
        return {
            name: Value(self._state.get_global_value(declaration.identifier))
            for name, declaration in self.explorer._translation.declarations.globals_table.items()
            if not declaration.is_transient
        }

    def get_local_env(self, instance: model.Instance) -> t.Mapping[str, Value]:
        return {
            name: Value(self._state.get_global_value(declaration.identifier))
            for name, declaration in self.explorer._translation.declarations.locals_table[
                instance
            ].items()
        }

    @functools.cached_property
    def locations(self) -> t.Mapping[model.Instance, model.Location]:
        return {
            instance: self.explorer._translation.reversed_instance_to_location_names[
                instance
            ][self._state.get_location_of(name)]
            for instance, name in self.explorer._translation.instance_names.items()
        }

    @functools.cached_property
    def transitions(self) -> t.Sequence[Transition[TimeTypeT]]:
        return tuple(
            Transition(self.explorer, self, self._state, transition)
            for transition in self._state.transitions()
        )

    @property
    def valuations(self) -> TimeTypeT:
        raise NotImplementedError()


def _compile(translation: Translation) -> t.Any:
    return _engine.MDPExplorer(translation.json_network)


@d.dataclass(frozen=True, eq=False, repr=False)
class Explorer(t.Generic[TimeTypeT]):
    """
    To create an instance of this class use :meth:`new_discrete_time`
    or :meth:`new_zone_time`.
    Never create an instance of this class directly.
    Always use one of the provided :code:`new` methods.

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
    """

    network: model.Network
    time_type: t.Type[TimeTypeT]

    _translation: Translation
    _compiled: t.Any

    @staticmethod
    def new_discrete_time(
        network: model.Network,
        *,
        parameters: t.Optional[t.Mapping[str, expressions.ValueOrExpression]] = None,
    ) -> Explorer[DiscreteTime]:
        """
        Creates a new :class:`Explorer` for a discrete time model.
        """
        if not network.ctx.model_type.is_discrete_time:
            raise InvalidModelType(
                f"{network.ctx.model_type} is not a discrete time model type"
            )
        translation = translate_network(
            network,
            parameters={
                name: expressions.ensure_expr(expr)
                for name, expr in (parameters or {}).items()
            },
        )
        return Explorer(network, DiscreteTime, translation, _compile(translation))

    @staticmethod
    def new_continuous_time(
        network: model.Network,
        *,
        parameters: t.Optional[t.Mapping[str, expressions.ValueOrExpression]] = None,
    ) -> Explorer[DiscreteTime]:
        """
        Creates a new :class:`Explorer` for a timed model using `float` zones
        for the representation of time.
        """
        raise NotImplementedError()

    @functools.cached_property
    def initial_states(self) -> t.AbstractSet[State[TimeTypeT]]:
        return frozenset(
            State(self, state) for state in self._compiled.initial_states()
        )
