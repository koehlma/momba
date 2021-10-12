# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import random

from .. import engine, model
from ..engine import explore

from . import api


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
class RewardStructure:
    goal_reached: float = 100
    dead_end: float = -100
    step_taken: float = 0
    invalid_action: float = -100


@d.dataclass(frozen=True)
class Property:
    goal_predicate: model.Expression
    dead_predicate: model.Expression


def _extract_property(prop: model.Expression) -> Property:
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
        # we assume that this a propositional formula
        return Property(
            goal_predicate=prop.formula, dead_predicate=model.ensure_expr(False)
        )
    elif isinstance(prop, model.properties.BinaryPathFormula):
        assert (
            prop.operator is model.operators.BinaryPathOperator.UNTIL
        ), "Unsupported binary path formula."
        # we assume that this a propositional formula
        left = prop.left
        right = prop.right
        return Property(
            goal_predicate=right, dead_predicate=model.expressions.logic_not(left)
        )
    else:
        raise Exception("Unsupported property!")


@d.dataclass(frozen=True)
class GenericContext:
    explorer: engine.Explorer[engine.DiscreteTime]

    controlled_instance: model.Instance

    initial_state: engine.State[engine.DiscreteTime]

    generic_property: Property

    dead_predicate: explore.CompiledGlobalExpression[engine.DiscreteTime]
    goal_predicate: explore.CompiledGlobalExpression[engine.DiscreteTime]

    global_variables: t.Tuple[str, ...]

    rewards: RewardStructure

    num_actions: int
    num_features: int

    @classmethod
    def create(
        cls,
        explorer: engine.Explorer[engine.DiscreteTime],
        controlled_instance: model.Instance,
        property_name: str,
        *,
        rewards: t.Optional[RewardStructure] = None,
    ) -> GenericContext:
        network = explorer.network
        initial_states = explorer.initial_states
        generic_property = _extract_property(
            network.ctx.get_property_definition_by_name(property_name).expression
        )
        goal_predicate = explorer.compile_global_expression(
            generic_property.goal_predicate
        )
        dead_predicate = explorer.compile_global_expression(
            generic_property.dead_predicate
        )
        if len(initial_states) != 1:
            raise Exception("Invalid number of initial states.")
        (initial_state,) = initial_states
        global_variables = []
        num_actions = len(controlled_instance.automaton.edges)
        num_features = 0
        for declaration in network.ctx.global_scope.variable_declarations:
            if declaration.is_transient:
                continue
            global_variables.append(declaration.identifier)
            value = initial_state.global_env[declaration.identifier]
            num_features += _count_features(value)
        return cls(
            explorer=explorer,
            controlled_instance=controlled_instance,
            initial_state=initial_state,
            generic_property=generic_property,
            goal_predicate=goal_predicate,
            dead_predicate=dead_predicate,
            global_variables=tuple(sorted(global_variables)),
            rewards=rewards or RewardStructure(),
            num_actions=num_actions,
            num_features=num_features,
        )


class GenericExplorer(api.Explorer):
    ctx: GenericContext

    state: engine.State[engine.DiscreteTime]

    def __init__(
        self,
        ctx: GenericContext,
        *,
        state: t.Optional[engine.State[engine.DiscreteTime]] = None,
    ) -> None:
        self.ctx = ctx
        self.state = state or ctx.initial_state

    def _explore_until_choice(self) -> None:
        while (
            not any(
                self.ctx.controlled_instance in transition.index_vector
                for transition in self.state.transitions
            )
            and not self.has_terminated
        ):
            self.state = random.choice(self.state.transitions).destinations.pick().state

    @property
    def num_actions(self) -> int:
        return self.ctx.num_actions

    @property
    def num_features(self) -> int:
        return self.ctx.num_features

    @property
    def has_terminated(self) -> bool:
        is_dead = self.ctx.dead_predicate.evaluate(self.state).as_bool
        return not self.state.transitions or is_dead or self.has_reached_goal

    @property
    def state_vector(self) -> t.Sequence[float]:
        vector: t.List[float] = []
        for variable in self.ctx.global_variables:
            value = self.state.global_env[variable]
            _extend_state_vector(vector, value)
        return vector

    @property
    def has_reached_goal(self) -> bool:
        return self.ctx.goal_predicate.evaluate(self.state).as_bool

    @property
    def available_actions(self) -> t.Sequence[bool]:
        available_actions: t.Set[int] = set()
        for transition in self.state.transitions:
            available_actions.add(
                transition.index_vector.get(self.ctx.controlled_instance, -1)
            )
        available_actions.discard(-1)
        return [action in available_actions for action in range(self.num_actions)]

    def step(self, action: int) -> float:
        if self.has_terminated:
            if self.has_reached_goal:
                return self.ctx.rewards.goal_reached
            else:
                return self.ctx.rewards.dead_end
        selected_transitions = [
            transition
            for transition in self.state.transitions
            if transition.index_vector.get(self.ctx.controlled_instance, -1) == action
        ]
        if not selected_transitions:
            return self.ctx.rewards.invalid_action
        else:
            self.state = random.choice(selected_transitions).destinations.pick().state
            self._explore_until_choice()
        if self.has_reached_goal:
            return self.ctx.rewards.goal_reached
        elif self.has_terminated:
            return self.ctx.rewards.dead_end
        else:
            return self.ctx.rewards.step_taken

    def reset(self, *, explore_until_choice: bool = True) -> None:
        self.state = self.ctx.initial_state
        if explore_until_choice:
            self._explore_until_choice()

    def fork(self) -> GenericExplorer:
        return GenericExplorer(self.ctx, state=self.state)
