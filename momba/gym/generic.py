# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum
import random
import warnings

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


class _ActionResolver:
    num_actions: int

    def available(self, state: engine.State[engine.DiscreteTime]) -> t.Sequence[bool]:
        raise NotImplementedError()

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

    def available(self, state: engine.State[engine.DiscreteTime]) -> t.Sequence[bool]:
        available_actions: t.Set[int] = set()
        for transition in state.transitions:
            available_actions.add(transition.index_vector.get(self.instance, -1))
        available_actions.discard(-1)
        return [action in available_actions for action in range(self.num_actions)]

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
    reverse_action_mapping: t.Mapping[model.ActionType, int] = d.field(
        default_factory=dict
    )

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
            if action_type in action_types:
                action_mapping[len(action_mapping)] = action_type
        return cls(
            instance,
            num_actions,
            action_mapping,
            reverse_action_mapping={
                typ: number for number, typ in action_mapping.items()
            },
        )

    def available(self, state: engine.State[engine.DiscreteTime]) -> t.Sequence[bool]:
        available_actions: t.Set[int] = set()
        for transition in state.transitions:
            instance_action = transition.action_vector.get(self.instance, None)
            if instance_action is not None:
                available_actions.add(
                    self.reverse_action_mapping[instance_action.action_type]
                )
            available_actions.add(transition.index_vector.get(self.instance, -1))
        available_actions.discard(-1)
        return [action in available_actions for action in range(self.num_actions)]

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
    r"""A reach-avoid objective.

    Represents a reach-avoid objective of the form
    :math:`\lnot\phi\mathbin{\mathbf{U}}\psi`, i.e., :math:`\lnot\phi` has to
    be true until the goal :math:`\psi` is reached. Used in
    conjunction with :class:`Rewards` to provide rewards.
    """

    goal_predicate: model.Expression
    r"""A boolean expression for :math:`\psi`."""

    dead_predicate: model.Expression
    r"""A boolean expression for :math:`\phi`."""


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
            for instance in sorted(
                network.instances, key=lambda instance: str(instance.automaton.name)
            ):
                if instance is controlled_instance:
                    # do not include the local variables twice
                    continue
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

    def _has_choice(self, state: engine.State[engine.DiscreteTime]) -> bool:
        return any(
            self._ctx.controlled_instance in transition.index_vector
            for transition in state.transitions
        )

    def _is_final(self, state: engine.State[engine.DiscreteTime]) -> bool:
        return self._has_choice(state) or self._has_terminated(state)

    def _explore_until_choice(self) -> None:
        while not self._is_final(self.state):
            if len(self.state.transitions) > 1:
                warnings.warn(
                    "Uncontrolled nondeterminism has been resolved uniformly."
                )
            self.state = random.choice(self.state.transitions).destinations.pick().state

    def _explore_successors(
        self, state: engine.State[engine.DiscreteTime]
    ) -> t.Dict[engine.State[engine.DiscreteTime], float]:
        pending = [(1.0, state)]
        result: t.Dict[engine.State[engine.DiscreteTime], float] = {}
        while pending:
            probability, state = pending.pop()
            if self._is_final(state):
                if state not in result:
                    result[state] = 0.0
                result[state] += probability
            else:
                if len(state.transitions) > 1:
                    warnings.warn(
                        "Uncontrolled nondeterminism has been resolved uniformly."
                    )
                transition_probability = probability / len(state.transitions)
                for transition in state.transitions:
                    for destination in transition.destinations.support:
                        pending.append(
                            (
                                transition_probability
                                * float(
                                    transition.destinations.get_probability(destination)
                                ),
                                destination.state,
                            )
                        )
        return result

    @property
    def num_actions(self) -> int:
        return self._ctx.action_resolver.num_actions

    @property
    def num_features(self) -> int:
        return self._ctx.num_features

    def _state_vector(
        self, state: engine.State[engine.DiscreteTime]
    ) -> t.Sequence[float]:
        vector: t.List[float] = []
        for variable in self._ctx.global_variables:
            value = state.global_env[variable]
            _extend_state_vector(vector, value)
        local_env = state.get_local_env(self._ctx.controlled_instance)
        for variable in self._ctx.local_variables:
            _extend_state_vector(vector, local_env[variable])
        for instance, variables in self._ctx.other_variables.items():
            local_env = state.get_local_env(instance)
            for variable in variables:
                _extend_state_vector(vector, local_env[variable])
        return tuple(vector)

    @property
    def state_vector(self) -> t.Sequence[float]:
        return self._state_vector(self.state)

    def _has_terminated(self, state: engine.State[engine.DiscreteTime]) -> bool:
        is_dead = self._ctx.dead_predicate.evaluate(state).as_bool
        return not state.transitions or is_dead or self._has_reached_goal(state)

    def _has_reached_goal(self, state: engine.State[engine.DiscreteTime]) -> bool:
        return self._ctx.goal_predicate.evaluate(state).as_bool

    def _get_reward(self, state: engine.State[engine.DiscreteTime]) -> float:
        if self._has_reached_goal(state):
            return self._ctx.rewards.goal_reached
        elif self._has_terminated(state):
            return self._ctx.rewards.dead_end
        else:
            return self._ctx.rewards.step_taken

    @property
    def has_terminated(self) -> bool:
        return self._has_terminated(self.state)

    @property
    def has_reached_goal(self) -> bool:
        return self._has_reached_goal(self.state)

    @property
    def available_actions(self) -> t.Sequence[bool]:
        return tuple(self._ctx.action_resolver.available(self.state))

    @property
    def available_transitions(self) -> t.Sequence[abstract.Transition]:
        if self.has_terminated:
            return []
        result = []
        for action, available in enumerate(self.available_actions):
            if not available:
                continue
            transitions = self._ctx.action_resolver.resolve(
                self.state.transitions, action
            )
            if len(transitions) > 1:
                warnings.warn(
                    "Uncontrolled nondeterminism has been resolved uniformly."
                )
            transition_probability = 1.0 / len(transitions)
            destinations: t.Dict[t.Sequence[float], abstract.Destination] = {}
            for transition in transitions:
                for destination in transition.destinations.support:
                    for (
                        final_successor,
                        probability,
                    ) in self._explore_successors(destination.state).items():
                        final_successor_vector = self._state_vector(final_successor)
                        reward = self._get_reward(final_successor)
                        probability *= transition_probability
                        probability *= transition.destinations.get_probability(
                            destination
                        )
                        if final_successor_vector in destinations:
                            probability += destinations[
                                final_successor_vector
                            ].probability
                        destinations[final_successor_vector] = abstract.Destination(
                            final_successor_vector, reward, probability
                        )
            result.append(
                abstract.Transition(action, destinations=tuple(destinations.values()))
            )
        return result

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
            if len(selected_transitions) > 1:
                warnings.warn(
                    "Uncontrolled nondeterminism has been resolved uniformly."
                )
            self.state = random.choice(selected_transitions).destinations.pick().state
            self._explore_until_choice()
        return self._get_reward(self.state)

    def reset(self, *, explore_until_choice: bool = True) -> None:
        self.state = self._ctx.initial_state
        if explore_until_choice:
            self._explore_until_choice()

    def fork(self) -> GenericExplorer:
        return GenericExplorer(_ctx=self._ctx, _state=self.state)
