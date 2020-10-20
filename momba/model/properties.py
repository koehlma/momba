# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Michaela Klauck <klauck@cs.uni-saarland.de>
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import enum

from . import errors, operators, types

if t.TYPE_CHECKING:
    from . import context, expressions


class Property(abc.ABC):
    @abc.abstractmethod
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class Aggregate(Property):
    function: operators.AggregationFunction
    values: Property
    predicate: Property

    def infer_type(self, scope: context.Scope) -> types.Type:
        predicate_type = self.predicate.infer_type(scope)
        if not predicate_type == types.BOOL:
            raise errors.InvalidTypeError(
                f"expected types.BOOL but got {predicate_type}"
            )
        values_type = self.values.infer_type(scope)
        if values_type not in self.function.allowed_values_type:
            raise errors.InvalidTypeError(
                f"invalid type {values_type} of values in filter function"
            )
        return self.function.infer_result_type(values_type)


class StatePredicate(enum.Enum):
    INITIAL = "initial"
    DEADLOCK = "deadlock"
    TIMELOCK = "timelock"


@d.dataclass(frozen=True)
class StateSelector(Property):
    predicate: StatePredicate

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.BOOL


INITIAL_STATES = StateSelector(StatePredicate.INITIAL)
DEADLOCK_STATES = StateSelector(StatePredicate.DEADLOCK)
TIMELOCK_STATES = StateSelector(StatePredicate.TIMELOCK)


@d.dataclass(frozen=True)
class Probability(Property):
    operator: operators.MinMax
    formula: Property

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if not formula_type == types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        return types.REAL


@d.dataclass(frozen=True)
class PathQuantifier(Property):
    quantifier: operators.Quantifier
    formula: Property

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if not formula_type == types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        return types.BOOL


class AccumulationInstant(enum.Enum):
    STEPS = "steps"
    TIME = "time"
    EXIT = "exit"


@d.dataclass(frozen=True)
class ExpectedReward(Property):
    operator: operators.MinMax
    reward: Property
    accumulate: t.Optional[t.FrozenSet[AccumulationInstant]] = None
    reachability: t.Optional[Property] = None
    step_instant: t.Optional[expressions.Expression] = None
    time_instant: t.Optional[expressions.Expression] = None
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        # TODO: check the types of the provided arguments
        return types.REAL


@d.dataclass(frozen=True)
class RewardInstant:
    expression: expressions.Expression
    accumulate: t.FrozenSet[AccumulationInstant]
    instant: expressions.Expression


@d.dataclass(frozen=True)
class SteadyState(Property):
    operator: operators.MinMax
    formula: Property
    accumulate: t.Optional[t.FrozenSet[AccumulationInstant]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        # TODO: check the types of the provided arguments
        return types.REAL


@d.dataclass(frozen=True)
class BinaryPathFormula(Property):
    operator: operators.BinaryPathOperator
    left: Property
    right: Property
    step_bounds: t.Optional[Interval] = None
    time_bounds: t.Optional[Interval] = None
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        left_type = self.left.infer_type(scope)
        if left_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {left_type}")
        right_type = self.left.infer_type(scope)
        if right_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {right_type}")
        # TODO: check the types of the other arguments
        return types.BOOL


@d.dataclass(frozen=True)
class UnaryPathFormula(Property):
    operator: operators.UnaryPathOperator
    formula: Property
    step_bounds: t.Optional[Interval] = None
    time_bounds: t.Optional[Interval] = None
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if formula_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        # TODO: check the types of the other arguments
        return types.BOOL


@d.dataclass(frozen=True)
class Interval:
    lower: t.Optional[expressions.Expression] = None
    upper: t.Optional[expressions.Expression] = None
    lower_exclusive: t.Optional[expressions.Expression] = None
    upper_exclusive: t.Optional[expressions.Expression] = None


@d.dataclass(frozen=True)
class RewardBound:
    expression: expressions.Expression
    accumulate: t.FrozenSet[AccumulationInstant]
    bounds: Interval


def aggregate(
    function: operators.AggregationFunction,
    values: Property,
    states: Property = INITIAL_STATES,
) -> Property:
    return Aggregate(function, values, states)


def min_prob(formula: Property) -> Property:
    return Probability(operators.MinMax.MIN, formula)


def max_prob(formula: Property) -> Property:
    return Probability(operators.MinMax.MAX, formula)


def forall_paths(formula: Property) -> Property:
    return PathQuantifier(operators.Quantifier.FORALL, formula)


def exists_path(formula: Property) -> Property:
    return PathQuantifier(operators.Quantifier.EXISTS, formula)


def min_expected_reward(
    reward: expressions.Expression,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
    reachability: t.Optional[Property] = None,
    step_instant: t.Optional[expressions.Expression] = None,
    time_instant: t.Optional[expressions.Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None,
) -> Property:
    return ExpectedReward(
        operators.MinMax.MIN,
        reward,
        accumulate=None if accumulate is None else frozenset(accumulate),
        reachability=reachability,
        step_instant=step_instant,
        time_instant=time_instant,
        reward_instants=reward_instants,
    )


def max_expected_reward(
    reward: expressions.Expression,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
    reachability: t.Optional[Property] = None,
    step_instant: t.Optional[expressions.Expression] = None,
    time_instant: t.Optional[expressions.Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None,
) -> Property:
    return ExpectedReward(
        operators.MinMax.MAX,
        reward,
        accumulate=None if accumulate is None else frozenset(accumulate),
        reachability=reachability,
        step_instant=step_instant,
        time_instant=time_instant,
        reward_instants=reward_instants,
    )


def min_steady_state(
    formula: Property,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
) -> Property:
    return SteadyState(
        operators.MinMax.MIN,
        formula,
        accumulate=None if accumulate is None else frozenset(accumulate),
    )


def max_steady_state(
    formula: Property,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
) -> Property:
    return SteadyState(
        operators.MinMax.MAX,
        formula,
        accumulate=None if accumulate is None else frozenset(accumulate),
    )


def until(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> Property:
    return BinaryPathFormula(
        operators.BinaryPathOperator.UNTIL,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def weak_until(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> Property:
    return BinaryPathFormula(
        operators.BinaryPathOperator.WEAK_UNTIL,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def release(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> Property:
    return BinaryPathFormula(
        operators.BinaryPathOperator.RELEASE,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def eventually(
    formula: Property,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> Property:
    return UnaryPathFormula(
        operators.UnaryPathOperator.EVENTUALLY,
        formula,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def globally(
    formula: Property,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> Property:
    return UnaryPathFormula(
        operators.UnaryPathOperator.GLOBALLY,
        formula,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )
