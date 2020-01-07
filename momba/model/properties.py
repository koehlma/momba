# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import enum
import dataclasses
import typing as t

from typing import List
from . import operators

if t.TYPE_CHECKING:
    from .expressions import Expression


class Property:
    ...


class NamedProperty:
    _name: str
    _prop: Property

    def __init__(self, prop: Property, name: str) -> None:
        self._name = name
        self._prop = prop

    @property
    def name(self) -> str:
        return self._name

    @property
    def prop(self) -> Property:
        return self._prop


@dataclasses.dataclass(frozen=True)
class PropertyInterval:
    lower: Expression
    lower_exclusive: Expression
    upper: Expression
    upper_exclusive: Expression


class RewardAccumulation(enum.Enum):
    STEPS = "steps"
    TIME = "time"
    EXIT = "exit"


class StatePredicates(enum.Enum):
    INITIAL = "initial"
    DEADLOCK = "deadlock"
    TIMELOCK = "timelock"


@dataclasses.dataclass(frozen=True)
class RewardInstant:
    expression: Expression
    accumulate: RewardAccumulation
    bounds: PropertyInterval


@dataclasses.dataclass(frozen=True)
class RewardBound:
    expression: Expression
    accumulate: RewardAccumulation
    instant: Expression


@dataclasses.dataclass(frozen=True)
class Filter(Property):
    function: operators.FilterFunction
    values: Property
    states: Property


@dataclasses.dataclass(frozen=True)
class Prob(Property):
    operator: operators.PropertyOperator
    expression: Expression


@dataclasses.dataclass(frozen=True)
class PathFormula(Property):
    operator: operators.PathOperator
    expression: Expression


@dataclasses.dataclass(frozen=True)
class Expected(Property):
    operator: operators.PropertyOperator
    expression: Expression
    acc: RewardAccumulation
    reach: Property
    step_instant: Expression
    time_instant: Expression
    reward_instants: t.Sequence[RewardInstant]


@dataclasses.dataclass(frozen=True)
class Steady(Property):
    operator: operators.PropertyOperator
    expression: Expression
    acc: RewardAccumulation


@dataclasses.dataclass(frozen=True)
class TimedProp(Property):
    operator: operators.TimeOperator
    left: Property
    right: Property
    step_bounds: t.Optional[PropertyInterval]
    time_bounds: PropertyInterval
    reward_bounds: List[RewardBound]


def filter(
    func: operators.FilterFunction, values: Expression, states: Expression
) -> Property:
    return Filter(func, values, states)


def minProb(expression: Expression) -> Property:
    return Prob(operators.PropertyOperator.PMIN, expression)


def maxProb(expression: Expression) -> Property:
    return Prob(operators.PropertyOperator.PMAX, expression)


def forall(expression: Expression) -> Property:
    return PathFormula(operators.PathOperator.FORALL, expression)


def exists(expression: Expression) -> Property:
    return PathFormula(operators.PathOperator.EXISTS, expression)


def minExp(
    expression: Expression,
    *,
    acc: RewardAccumulation,
    reach: Property,
    step_instant: Expression,
    time_instant: Expression,
    reward_instants: List[RewardInstant]
) -> Property:
    return Expected(
        operators.PropertyOperator.EMIN,
        expression,
        acc,
        reach,
        step_instant,
        time_instant,
        reward_instants,
    )


def maxExp(
    expression: Expression,
    *,
    acc: RewardAccumulation,
    reach: Property,
    step_instant: Expression,
    time_instant: Expression,
    reward_instants: List[RewardInstant]
) -> Property:
    return Expected(
        operators.PropertyOperator.EMAX,
        expression,
        acc,
        reach,
        step_instant,
        time_instant,
        reward_instants,
    )


def sMin(expression: Expression, acc: RewardAccumulation) -> Property:
    return Steady(operators.PropertyOperator.SMIN, expression, acc)


def sMax(expression: Expression, acc: RewardAccumulation) -> Property:
    return Steady(operators.PropertyOperator.SMAX, expression, acc)


def until(
    left: Property,
    right: Property,
    *,
    step_bounds: t.Optional[PropertyInterval] = None,
    time_bounds: PropertyInterval,
    reward_bounds: List[RewardBound]
) -> Property:
    return TimedProp(
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
    step_bounds: PropertyInterval,
    time_bounds: PropertyInterval,
    reward_bounds: List[RewardBound]
) -> Property:
    return TimedProp(
        operators.TimeOperator.WEAKU,
        left,
        right,
        step_bounds,
        time_bounds,
        reward_bounds,
    )
