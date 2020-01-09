# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import enum
import dataclasses
import typing as t

from . import operators

if t.TYPE_CHECKING:
    from .expressions import Expression


class Property:
    ...


@dataclasses.dataclass(frozen=True)
class PropertyInterval:
    lower: t.Optional[Expression] = None
    lower_exclusive: t.Optional[Expression] = None
    upper: t.Optional[Expression] = None
    upper_exclusive: t.Optional[Expression] = None


class RewardAccumulationInstant(enum.Enum):
    STEPS = "steps"
    TIME = "time"
    EXIT = "exit"


class StatePredicates(Property, enum.Enum):
    INITIAL = "initial"
    DEADLOCK = "deadlock"
    TIMELOCK = "timelock"


@dataclasses.dataclass(frozen=True)
class RewardBound:
    expression: Expression
    accumulate: t.Sequence[RewardAccumulationInstant]
    bounds: PropertyInterval


@dataclasses.dataclass(frozen=True)
class RewardInstant:
    expression: Expression
    accumulate: t.Sequence[RewardAccumulationInstant]
    instant: Expression


@dataclasses.dataclass(frozen=True)
class Filter(Property):
    function: operators.FilterFunction
    values: Property
    states: Property


@dataclasses.dataclass(frozen=True)
class ProbabilityProp(Property):
    operator: operators.Probability
    expression: Property


@dataclasses.dataclass(frozen=True)
class PathFormula(Property):
    operator: operators.PathOperator
    expression: Property


@dataclasses.dataclass(frozen=True)
class Expected(Property):
    operator: operators.Expected
    expression: Property
    accumulate: t.Optional[t.Sequence[RewardAccumulationInstant]] = None
    reach: t.Optional[Property] = None
    step_instant: t.Optional[Expression] = None
    time_instant: t.Optional[Expression] = None
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None


@dataclasses.dataclass(frozen=True)
class Steady(Property):
    operator: operators.Steady
    expression: Property
    accumulate: t.Optional[t.Sequence[RewardAccumulationInstant]] = None


@dataclasses.dataclass(frozen=True)
class Timed(Property):
    operator: operators.TimeOperator
    left: Property
    right: Property
    step_bounds: t.Optional[PropertyInterval] = None
    time_bounds: t.Optional[PropertyInterval] = None
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None


def filter(
    func: operators.FilterFunction, values: Expression, states: Expression
) -> Property:
    return Filter(func, values, states)


def min_prob(expression: Property) -> Property:
    return ProbabilityProp(operators.Probability.PMIN, expression)


def max_prob(expression: Property) -> Property:
    return ProbabilityProp(operators.Probability.PMAX, expression)


def forall(expression: Property) -> Property:
    return PathFormula(operators.PathOperator.FORALL, expression)


def exists(expression: Property) -> Property:
    return PathFormula(operators.PathOperator.EXISTS, expression)


def min_expected(
    expression: Expression,
    *,
    accumulate: t.Optional[t.Sequence[RewardAccumulationInstant]] = None,
    reach: t.Optional[Property] = None,
    step_instant: t.Optional[Expression] = None,
    time_instant: t.Optional[Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None
) -> Property:
    return Expected(
        operators.Expected.EMIN,
        expression,
        accumulate,
        reach,
        step_instant,
        time_instant,
        reward_instants,
    )


def max_expected(
    expression: Expression,
    *,
    accumulate: t.Optional[t.Sequence[RewardAccumulationInstant]] = None,
    reach: t.Optional[Property] = None,
    step_instant: t.Optional[Expression] = None,
    time_instant: t.Optional[Expression] = None,
    reward_instants: t.Optional[t.Sequence[RewardInstant]] = None
) -> Property:
    return Expected(
        operators.Expected.EMAX,
        expression,
        accumulate,
        reach,
        step_instant,
        time_instant,
        reward_instants,
    )


def s_min(
    expression: Property, accumulate: t.Sequence[RewardAccumulationInstant]
) -> Property:
    return Steady(operators.Steady.SMIN, expression, accumulate)


def s_max(
    expression: Property, accumulate: t.Sequence[RewardAccumulationInstant]
) -> Property:
    return Steady(operators.Steady.SMAX, expression, accumulate)


def until(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[PropertyInterval] = None,
    time_bounds: t.Optional[PropertyInterval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None
) -> Property:
    return Timed(
        operators.TimeOperator.UNTIL,
        left,
        right,
        step_bounds,
        time_bounds,
        reward_bounds,
    )


def weak_until(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[PropertyInterval] = None,
    time_bounds: t.Optional[PropertyInterval] = None,
    reward_bounds: t.Optional[t.Sequence[RewardBound]] = None
) -> Property:
    return Timed(
        operators.TimeOperator.WEAKU,
        left,
        right,
        step_bounds,
        time_bounds,
        reward_bounds,
    )
