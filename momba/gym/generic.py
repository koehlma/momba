# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import enum
import random

from .. import engine, model
from ..engine import explore

from . import abstract


class Actions(enum.Enum):
    """Specifies how edges of the controlled automaton are chosen."""

    EDGE_BY_INDEX = enum.auto()
    """The edge is chosen based on its index."""

    EDGE_BY_LABEL = enum.auto()
    """The edge is chosen based on its label. """


class Observations(enum.Enum):
    """Specifies what is observable by the agent."""

    GLOBAL_ONLY = enum.auto()
    """Only global variables are observable."""

    LOCAL_AND_GLOBAL = enum.auto()
    """Local and global variables are observable."""

    OMNISCIENT = enum.auto()
    """All (non-transient) variables are observable."""


class _ActionResolver(abc.ABC):
    @property
    @abc.abstractmethod
    def num_actions(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def resolve(
        self,
        transitions: t.Sequence[engine.Transition[engine.DiscreteTime]],
        action: int,
    ) -> t.Sequence[engine.Transition[engine.DiscreteTime]]:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class _EdgeByIndexResolver(_ActionResolver):
    instance: model.Instance
    num_actions: int

    @classmethod
    def create(cls, instance: model.Instance) -> _EdgeByIndexResolver:
        return cls(instance, len(instance.automaton.edges))

    def resolve(
        self,
        transitions: t.Sequence[engine.Transition[engine.DiscreteTime]],
        action: int,
    ) -> t.Sequence[engine.Transition[engine.DiscreteTime]]:
        return [
            transition
            for transition in transitions
            if transition.index_vector.get(self.instance, -1) == action
        ]


@d.dataclass(frozen=True)
class _EdgeByLabelResolver(_ActionResolver):
    instance: model.Instance
    num_actions: int
    action_mapping: t.Mapping[int, model.ActionType] = d.field(default_factory=dict)

    @classmethod
    def create(
        cls, ctx: model.Context, instance: model.Instance
    ) -> _EdgeByLabelResolver:
        action_types: t.Set[model.ActionType] = set()
        for edge in instance.automaton.edges:
            if edge.action_pattern is not None:
                action_types.add(edge.action_pattern.action_type)
        num_actions = len(action_types)
        action_mapping: t.Dict[int, model.ActionType] = {}
        # `ctx.action_types` should be ordered as in the JANI file
        for action_type in ctx.action_types.values():
            action_mapping[len(action_mapping)] = action_type
        return cls(instance, num_actions, action_mapping)

    def resolve(
        self,
        transitions: t.Sequence[engine.Transition[engine.DiscreteTime]],
        action: int,
    ) -> t.Sequence[engine.Transition[engine.DiscreteTime]]:
        action_type = self.action_mapping[action]
        result = []
        for transition in transitions:
            instance_action = transition.action_vector.get(self.instance, None)
            if (
                instance_action is not None
                and instance_action.action_type == action_type
            ):
                result.append(transition)
        return transitions


def _create_action_resolver(
    actions: Actions, ctx: model.Context, instance: model.Instance
) -> _ActionResolver:
    if actions is Actions.EDGE_BY_INDEX:
        return _EdgeByIndexResolver.create(instance)
    else:
        assert actions is Actions.EDGE_BY_LABEL
        return _EdgeByLabelResolver.create(ctx, instance)


def _count_features(value: engine.Value) -> int:
    if value.is_int or value.is_bool or value.is_float:
        return 1
    else:
        return sum(map(_count_features, value.as_array))


def _extend_state_vector(vector: t.List[float], value: engine.Value) -> None:
    if value.is_array:
        for value in value.as_array:
            _extend_state_vector(vector, value)
    else:
        vector.append(float(value._value))


@d.dataclass(frozen=True)
class Rewards:
    """Specifies the rewards for a reachability objective."""

    goal_reached: float = 100
    """The reward when a goal state is reached. """

    dead_end: float = -100
    """The reward when a dead end or bad state is reached."""

    step_taken: float = 0
    """The reward when a valid decision has been taken."""

    invalid_action: float = -100
    """The reward when an invalid decision has been taken."""


DEFAULT_REWARD_STRUCTURE = Rewards()


@d.dataclass(frozen=True)
class Objective:
    r"""A reachability objective.

    Represents a reachability objective of the form
    :math:`\lnot\phi\mathbin{\mathbf{U}}\psi`, i.e., :math:`\lnot\phi` has to
    be true until the goal :math:`\psi` is reached. Used in
    conjunction with :class:`Rewards` to provide rewards.
    """

    goal_predicate: model.Expression
    """A boolean expression for :math:`\psi`."""

    dead_predicate: model.Expression
    """A boolean expression for :math:`\phi`."""


def _extract_property(prop: model.Expression) -> Objective:
    if isinstance(prop, model.properties.Aggregate):
        assert prop.function in {
            model.operators.AggregationFunction.MIN,
            model.operators.AggregationFunction.MAX,
            model.operators.AggregationFunction.VALUES,
        }, f"Unsupported aggregation function {prop.function}"
        assert isinstance(
            prop.predicate, model.properties.StateSelector
        ), f"Unsupported state predicate {prop.predicate} in aggregation"
        assert (
            prop.predicate.predicate is model.properties.StatePredicate.INITIAL
        ), "Unsupported state predicate for aggregation."
        prop = prop.values
    if isinstance(prop, model.properties.Probability):
        prop = prop.formula
    if isinstance(prop, model.properties.UnaryPathFormula):
        assert (
            prop.operator is model.operators.UnaryPathOperator.EVENTUALLY
        ), "Unsupported unary path formula."
        return Objective(
            goal_predicate=prop.formula, dead_predicate=model.ensure_expr(False)
        )
    elif isinstance(prop, model.properties.BinaryPathFormula):
        assert (
            prop.operator is model.operators.BinaryPathOperator.UNTIL
        ), "Unsupported binary path formula."
        left = prop.left
        right = prop.right
        return Objective(
            goal_predicate=right, dead_predicate=model.expressions.logic_not(left)
        )
    else:
        raise Exception("Unsupported property!")


@d.dataclass(frozen=True)
class _Context:
    explorer: engine.Explorer[engine.DiscreteTime]

    controlled_instance: model.Instance

    initial_state: engine.State[engine.DiscreteTime]

    objective: Objective

    dead_predicate: explore.CompiledGlobalExpression[engine.DiscreteTime]
    goal_predicate: explore.CompiledGlobalExpression[engine.DiscreteTime]

    rewards: Rewards

    actions: Actions
    observations: Observations

    action_resolver: _ActionResolver

    global_variables: t.Tuple[str, ...]
    local_variables: t.Tuple[str, ...]
    other_variables: t.Mapping[model.Instance, t.Tuple[str, ...]]

    num_features: int

    @classmethod
    def create(
        cls,
        explorer: engine.Explorer[engine.DiscreteTime],
        controlled_instance: model.Instance,
        property_name: str,
        *,
        rewards: t.Optional[Rewards] = None,
        actions: Actions = Actions.EDGE_BY_INDEX,
        observations: Observations = Observations.GLOBAL_ONLY,
    ) -> _Context:
        network = explorer.network
        initial_states = explorer.initial_states
        objective = _extract_property(
            network.ctx.get_property_definition_by_name(property_name).expression
        )
        goal_predicate = explorer.compile_global_expression(objective.goal_predicate)
        dead_predicate = explorer.compile_global_expression(objective.dead_predicate)
        if len(initial_states) != 1:
            raise Exception("Invalid number of initial states.")
        (initial_state,) = initial_states

        num_features = 0

        global_variables = []
        for declaration in network.ctx.global_scope.variable_declarations:
            if declaration.is_transient:
                continue
            global_variables.append(declaration.identifier)
            value = initial_state.global_env[declaration.identifier]
            num_features += _count_features(value)

        local_variables = []
        if observations in {Observations.LOCAL_AND_GLOBAL, Observations.OMNISCIENT}:
            for (
                declaration
            ) in controlled_instance.automaton.scope.variable_declarations:
                if declaration.is_transient:
                    continue
                local_variables.append(declaration.identifier)
                value = initial_state.get_local_env(controlled_instance)[
                    declaration.identifier
                ]
                num_features += _count_features(value)

        other_variables: t.Dict[model.Instance, t.Tuple[str, ...]] = {}
        if observations is Observations.OMNISCIENT:
            for instance in network.instances:
                instance_variables: t.List[str] = []
                for declaration in instance.automaton.scope.variable_declarations:
                    if declaration.is_transient:
                        continue
                    instance_variables.append(declaration.identifier)
                    value = initial_state.get_local_env(instance)[
                        declaration.identifier
                    ]
                    num_features += _count_features(value)
                if instance_variables:
                    other_variables[instance] = tuple(sorted(instance_variables))

        return cls(
            explorer=explorer,
            controlled_instance=controlled_instance,
            initial_state=initial_state,
            objective=objective,
            goal_predicate=goal_predicate,
            dead_predicate=dead_predicate,
            actions=actions,
            observations=observations,
            global_variables=tuple(sorted(global_variables)),
            local_variables=tuple(sorted(local_variables)),
            other_variables=other_variables,
            rewards=rewards or Rewards(),
            action_resolver=_create_action_resolver(
                actions, network.ctx, controlled_instance
            ),
            num_features=num_features,
        )


class GenericExplorer(abstract.Explorer):
    _ctx: _Context

    state: engine.State[engine.DiscreteTime]

    def __init__(
        self,
        *,
        _ctx: _Context,
        _state: t.Optional[engine.State[engine.DiscreteTime]] = None,
    ) -> None:
        self._ctx = _ctx
        self.state = _state or _ctx.initial_state

    @classmethod
    def create(
        cls,
        explorer: engine.Explorer[engine.DiscreteTime],
        controlled_instance: model.Instance,
        property_name: str,
        *,
        rewards: t.Optional[Rewards] = None,
        actions: Actions = Actions.EDGE_BY_INDEX,
        observations: Observations = Observations.GLOBAL_ONLY,
    ) -> GenericExplorer:
        ctx = _Context.create(
            explorer,
            controlled_instance,
            property_name,
            rewards=rewards,
            actions=actions,
            observations=observations,
        )
        return cls(_ctx=ctx, _state=ctx.initial_state)

    def _explore_until_choice(self) -> None:
        while (
            not any(
                self._ctx.controlled_instance in transition.index_vector
                for transition in self.state.transitions
            )
            and not self.has_terminated
        ):
            self.state = random.choice(self.state.transitions).destinations.pick().state

    @property
    def num_actions(self) -> int:
        return self._ctx.action_resolver.num_actions

    @property
    def num_features(self) -> int:
        return self._ctx.num_features

    @property
    def has_terminated(self) -> bool:
        is_dead = self._ctx.dead_predicate.evaluate(self.state).as_bool
        return not self.state.transitions or is_dead or self.has_reached_goal

    @property
    def state_vector(self) -> t.Sequence[float]:
        vector: t.List[float] = []
        for variable in self._ctx.global_variables:
            value = self.state.global_env[variable]
            _extend_state_vector(vector, value)
        return vector

    @property
    def has_reached_goal(self) -> bool:
        return self._ctx.goal_predicate.evaluate(self.state).as_bool

    @property
    def available_actions(self) -> t.Sequence[bool]:
        available_actions: t.Set[int] = set()
        for transition in self.state.transitions:
            available_actions.add(
                transition.index_vector.get(self._ctx.controlled_instance, -1)
            )
        available_actions.discard(-1)
        return [action in available_actions for action in range(self.num_actions)]

    def step(self, action: int) -> float:
        if self.has_terminated:
            if self.has_reached_goal:
                return self._ctx.rewards.goal_reached
            else:
                return self._ctx.rewards.dead_end
        selected_transitions = self._ctx.action_resolver.resolve(
            self.state.transitions, action
        )
        if not selected_transitions:
            return self._ctx.rewards.invalid_action
        else:
            self.state = random.choice(selected_transitions).destinations.pick().state
            self._explore_until_choice()
        if self.has_reached_goal:
            return self._ctx.rewards.goal_reached
        elif self.has_terminated:
            return self._ctx.rewards.dead_end
        else:
            return self._ctx.rewards.step_taken

    def reset(self, *, explore_until_choice: bool = True) -> None:
        self.state = self._ctx.initial_state
        if explore_until_choice:
            self._explore_until_choice()

    def fork(self) -> GenericExplorer:
        return GenericExplorer(_ctx=self._ctx, _state=self.state)
