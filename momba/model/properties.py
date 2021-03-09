# -*- coding:utf-8 -*-
#
# Copyright (C) 2020-2021, Saarland University
# Copyright (C) 2020-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>
# Copyright (C) 2020-2021, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum

from . import errors, expressions, operators, types

if t.TYPE_CHECKING:
    from . import context


@d.dataclass(frozen=True)
class Aggregate(expressions.Expression):
    """
    Applies an aggregation function over a set of states.

    Attributes
    ----------
    function:
        The aggregation function to apply.
    values:
        The values to aggregate over.
    predcate:
        The predicate used to identify the states to aggregate over.
    """

    function: operators.AggregationFunction
    values: expressions.Expression
    predicate: expressions.Expression

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

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return (self.predicate,)


class StatePredicate(enum.Enum):
    """
    An enum of state predicates to be used with :class:`Aggregate`.
    """

    INITIAL = "initial"
    """ The state is an initial state. """

    DEADLOCK = "deadlock"
    """ The state is a deadlock state. """

    TIMELOCK = "timelock"
    """ The state is a timelock state. """


@d.dataclass(frozen=True)
class StateSelector(expressions.Expression):
    """
    State selector expression using :class:`StatePredicate`.

    Attributes
    ----------
    predicate:
        A :class:`StatePredicate`.
    """

    predicate: StatePredicate

    def infer_type(self, scope: context.Scope) -> types.Type:
        return types.BOOL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return ()


INITIAL_STATES = StateSelector(StatePredicate.INITIAL)
DEADLOCK_STATES = StateSelector(StatePredicate.DEADLOCK)
TIMELOCK_STATES = StateSelector(StatePredicate.TIMELOCK)


@d.dataclass(frozen=True)
class Probability(expressions.Expression):
    """
    Probability property.

    Attributes
    ----------
    operator:
        *Min* or *max* probability (:class:`~momba.model.operators.MinMax`).
    formula:
        Boolean expression to compute the probability for.
    """

    operator: operators.MinMax
    formula: expressions.Expression

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if not formula_type == types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        return types.REAL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return (self.formula,)


@d.dataclass(frozen=True)
class PathQuantifier(expressions.Expression):
    """
    A temporal path quantifier property.

    Attributes
    ----------
    quantifier:
        The quantifier (:class:`~momba.model.operators.Quantifier`).
    formula:
        The inner formula.
    """

    quantifier: operators.Quantifier
    formula: expressions.Expression

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if not formula_type == types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        return types.BOOL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return (self.formula,)


class AccumulationInstant(enum.Enum):
    """
    En enumeration of reward accumulation instants.
    """

    STEPS = "steps"
    """ Accumulate at each step. """

    TIME = "time"
    """ Accumulate with the progression of time. """

    EXIT = "exit"
    """ Accumulate after exiting a state. """


@d.dataclass(frozen=True)
class ExpectedReward(expressions.Expression):
    """
    Expected reward property.

    Attributes
    ----------
    operator:
        *Min* or *max* probability (:class:`~momba.model.operators.MinMax`).
    reward:
        Expression to compute the reward.
    accumulate:
        A set of accumulation instants.
    reachability:
    step_instant:
    time_instant:
    reward_instants:
    """

    operator: operators.MinMax
    reward: expressions.Expression
    accumulate: t.Optional[t.FrozenSet[AccumulationInstant]] = None
    reachability: t.Optional[expressions.Expression] = None
    step_instant: t.Optional[expressions.Expression] = None
    time_instant: t.Optional[expressions.Expression] = None
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        # TODO: check the types of the provided arguments
        return types.REAL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        children: t.List[expressions.Expression] = []
        if self.reachability is not None:
            children.append(self.reachability)
        if self.step_instant is not None:
            children.append(self.step_instant)
        if self.time_instant is not None:
            children.append(self.time_instant)
        if self.reward_instants is not None:
            for reward_instant in self.reward_instants:
                children.extend(reward_instant.children)
        return children


@d.dataclass(frozen=True)
class RewardInstant:
    """
    A reward instant.

    Attributes
    ----------
    expression:
    accumulate:
    instant:
    """

    expression: expressions.Expression
    accumulate: t.FrozenSet[AccumulationInstant]
    instant: expressions.Expression

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return (self.expression, self.instant)


@d.dataclass(frozen=True)
class SteadyState(expressions.Expression):
    """
    A *steady-state* property.

    Attributes
    ----------
    operator:
    formula:
    accumulate:
    """

    operator: operators.MinMax
    formula: expressions.Expression
    accumulate: t.Optional[t.FrozenSet[AccumulationInstant]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        # TODO: check the types of the provided arguments
        return types.REAL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return (self.formula,)


@d.dataclass(frozen=True)
class BinaryPathFormula(expressions.Expression):
    """
    A temporal binary path formula.

    Attributes
    ----------
    operator:
    left:
    right:
    step_bounds:
    time_bounds:
    reward_bounds:
    """

    operator: operators.BinaryPathOperator
    left: expressions.Expression
    right: expressions.Expression
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

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        children: t.List[expressions.Expression] = [self.left, self.right]
        if self.step_bounds is not None:
            children.extend(self.step_bounds.expressions)
        if self.time_bounds is not None:
            children.extend(self.time_bounds.expressions)
        if self.reward_bounds is not None:
            for reward_bound in self.reward_bounds:
                children.extend(reward_bound.expressions)
        return children


@d.dataclass(frozen=True)
class UnaryPathFormula(expressions.Expression):
    """
    A temporal unary path formula.

    Attributes
    ----------
    operator:
    formula:
    step_bounds:
    time_bounds:
    reward_bounds:
    """

    operator: operators.UnaryPathOperator
    formula: expressions.Expression
    step_bounds: t.Optional[Interval] = None
    time_bounds: t.Optional[Interval] = None
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None

    def infer_type(self, scope: context.Scope) -> types.Type:
        formula_type = self.formula.infer_type(scope)
        if formula_type != types.BOOL:
            raise errors.InvalidTypeError(f"expected types.BOOL but got {formula_type}")
        # TODO: check the types of the other arguments
        return types.BOOL

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        children: t.List[expressions.Expression] = [self.formula]
        if self.step_bounds is not None:
            children.extend(self.step_bounds.expressions)
        if self.time_bounds is not None:
            children.extend(self.time_bounds.expressions)
        if self.reward_bounds is not None:
            for reward_bound in self.reward_bounds:
                children.extend(reward_bound.expressions)
        return children


@d.dataclass(frozen=True)
class Interval:
    """
    An interval.

    Attributes
    ----------
    lower:
        The lower bound of the interval or :code:`None`.
    upper:
        The upper bound of the interval or :code:`None`.
    lower_exclusive:
        Whether the lower bound is exclusive.
    upper_exclusive:
        Whether the upper bound is exclusive.
    """

    lower: t.Optional[expressions.Expression] = None
    upper: t.Optional[expressions.Expression] = None
    lower_exclusive: t.Optional[expressions.Expression] = None
    upper_exclusive: t.Optional[expressions.Expression] = None

    @property
    def expressions(self) -> t.Sequence[expressions.Expression]:
        return [
            expr
            for expr in [
                self.lower,
                self.upper,
                self.lower_exclusive,
                self.upper_exclusive,
            ]
            if expr is not None
        ]


@d.dataclass(frozen=True)
class RewardBound:
    """
    A *reward bound*.

    Attributes
    ----------
    expression:
    accumulate:
    bounds:
    """

    expression: expressions.Expression
    accumulate: t.FrozenSet[AccumulationInstant]
    bounds: Interval

    @property
    def expressions(self) -> t.Sequence[expressions.Expression]:
        expressions = [self.expression]
        expressions.extend(self.bounds.expressions)
        return expressions


def aggregate(
    function: operators.AggregationFunction,
    values: expressions.Expression,
    states: expressions.Expression = INITIAL_STATES,
) -> expressions.Expression:
    """
    Creates an :class:`Aggregate` property.
    """
    return Aggregate(function, values, states)


def min_prob(formula: expressions.Expression) -> expressions.Expression:
    """
    Constructs a :math:`P_\\mathit{min}` property.
    """
    return Probability(operators.MinMax.MIN, formula)


def max_prob(formula: expressions.Expression) -> expressions.Expression:
    """
    Constructs a :math:`P_\\mathit{max}` property.
    """
    return Probability(operators.MinMax.MAX, formula)


def forall_paths(formula: expressions.Expression) -> expressions.Expression:
    """
    CTL :math:`\\forall` path operator.
    """
    return PathQuantifier(operators.Quantifier.FORALL, formula)


def exists_path(formula: expressions.Expression) -> expressions.Expression:
    """
    CTL :math:`\\exists` exists operator.
    """
    return PathQuantifier(operators.Quantifier.EXISTS, formula)


def min_expected_reward(
    reward: expressions.Expression,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
    reachability: t.Optional[expressions.Expression] = None,
    step_instant: t.Optional[expressions.Expression] = None,
    time_instant: t.Optional[expressions.Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None,
) -> expressions.Expression:
    """
    Constructs a :math:`E_\\mathit{min}` property.
    """
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
    reachability: t.Optional[expressions.Expression] = None,
    step_instant: t.Optional[expressions.Expression] = None,
    time_instant: t.Optional[expressions.Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None,
) -> expressions.Expression:
    """
    Constructs a :math:`E_\\mathit{max}` property.
    """
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
    formula: expressions.Expression,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
) -> expressions.Expression:
    """
    Constructs a :math:`S_\\mathit{min}` property.
    """
    return SteadyState(
        operators.MinMax.MIN,
        formula,
        accumulate=None if accumulate is None else frozenset(accumulate),
    )


def max_steady_state(
    formula: expressions.Expression,
    *,
    accumulate: t.Optional[t.AbstractSet[AccumulationInstant]] = None,
) -> expressions.Expression:
    """
    Constructs a :math:`S_\\mathit{max}` property.
    """
    return SteadyState(
        operators.MinMax.MAX,
        formula,
        accumulate=None if accumulate is None else frozenset(accumulate),
    )


def until(
    left: expressions.Expression,
    right: expressions.Expression,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> expressions.Expression:
    """
    Constructs a temporal *until* property.
    """
    return BinaryPathFormula(
        operators.BinaryPathOperator.UNTIL,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def weak_until(
    left: expressions.Expression,
    right: expressions.Expression,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> expressions.Expression:
    """
    Constructs a temporal *weak-until* property.
    """
    return BinaryPathFormula(
        operators.BinaryPathOperator.WEAK_UNTIL,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def release(
    left: expressions.Expression,
    right: expressions.Expression,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> expressions.Expression:
    """
    Constructs a temporal *release* property.
    """
    return BinaryPathFormula(
        operators.BinaryPathOperator.RELEASE,
        left,
        right,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def eventually(
    formula: expressions.Expression,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> expressions.Expression:
    """
    Constructs a temporal *evenutally* property.
    """
    return UnaryPathFormula(
        operators.UnaryPathOperator.EVENTUALLY,
        formula,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )


def globally(
    formula: expressions.Expression,
    *,
    step_bounds: t.Optional[Interval] = None,
    time_bounds: t.Optional[Interval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None,
) -> expressions.Expression:
    """
    Constructs a temporal *globally* property.
    """
    return UnaryPathFormula(
        operators.UnaryPathOperator.GLOBALLY,
        formula,
        step_bounds=step_bounds,
        time_bounds=time_bounds,
        reward_bounds=reward_bounds,
    )
