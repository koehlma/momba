# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import fractions
import random

from ..kit import dbm

from ..utils.distribution import Distribution

from . import base, ta


StateT = t.TypeVar("StateT", bound=t.Hashable)
ActionT = t.TypeVar("ActionT", bound=t.Hashable)
ClockT = t.TypeVar("ClockT", bound=t.Hashable)


@d.dataclass(frozen=True)
class Destination(t.Generic[StateT, ClockT]):
    reset: t.FrozenSet[ClockT]
    location: ta.Location[StateT, ClockT]


@d.dataclass(frozen=True)
class Edge(t.Generic[StateT, ActionT, ClockT]):
    source: ta.Location[StateT, ClockT]
    action: t.Optional[ActionT]
    guard: t.FrozenSet[dbm.Constraint[ClockT]]
    destinations: Distribution[Destination[StateT, ClockT]]


class PTA(
    base.TS[ta.Location[StateT, ClockT], Edge[StateT, ActionT, ClockT]],
    t.Generic[StateT, ActionT, ClockT],
):
    @property
    @abc.abstractmethod
    def clock_variables(self) -> t.AbstractSet[ClockT]:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class SimulationStep(t.Generic[StateT, ActionT, ClockT]):
    time: fractions.Fraction
    source: ta.Location[StateT, ClockT]
    decision: Decision[StateT, ActionT, ClockT]
    destination: Destination[StateT, ClockT]
    valuation: t.Mapping[ClockT, fractions.Fraction]


@d.dataclass(frozen=True)
class Option(t.Generic[StateT, ActionT, ClockT]):
    edge: Edge[StateT, ActionT, ClockT]
    time_lower_bound: dbm.Bound
    time_upper_bound: t.Optional[dbm.Bound] = None


@d.dataclass(frozen=True)
class Decision(t.Generic[StateT, ActionT, ClockT]):
    edge: Edge[StateT, ActionT, ClockT]
    time: fractions.Fraction


class Oracle(t.Protocol[StateT, ActionT, ClockT]):
    def __call__(
        self,
        location: ta.Location[StateT, ClockT],
        valuation: t.Mapping[ClockT, fractions.Fraction],
        options: t.AbstractSet[Option[StateT, ActionT, ClockT]],
    ) -> Decision[StateT, ActionT, ClockT]:
        pass


def uniform_oracle(
    location: ta.Location[StateT, ClockT],
    valuation: t.Mapping[ClockT, fractions.Fraction],
    options: t.AbstractSet[Option[StateT, ActionT, ClockT]],
) -> Decision[StateT, ActionT, ClockT]:
    option = random.choice(list(options))
    assert (
        option.time_upper_bound is not None
    ), "infinite time upper bounds not supported by the uniform oracle"
    time = option.time_lower_bound.constant + fractions.Fraction(random.random()) * (
        option.time_upper_bound.constant - option.time_lower_bound.constant
    )
    return Decision(option.edge, time)


def _compute_upper_bound(
    constraints: t.Iterable[dbm.Constraint[ClockT]],
    valuation: t.Mapping[ClockT, fractions.Fraction],
) -> t.Optional[dbm.Bound]:
    upper_bound: t.Optional[dbm.Bound] = None
    for constraint in constraints:
        if constraint.right != dbm.ZERO_CLOCK:
            continue
        clock = constraint.left
        if isinstance(clock, dbm.ZeroClock):
            continue
        difference = constraint.constant - valuation[clock]
        if upper_bound is None or difference < upper_bound.constant:
            upper_bound = dbm.Bound(difference, is_strict=constraint.is_strict)
        elif difference == upper_bound.constant:
            if not upper_bound.is_strict and constraint.is_strict:
                upper_bound = dbm.Bound(difference, is_strict=constraint.is_strict)
    return upper_bound


_ZERO_BOUND = dbm.Bound(fractions.Fraction(0), is_strict=False)


def _compute_lower_bound(
    constraints: t.Iterable[dbm.Constraint[ClockT]],
    valuation: t.Mapping[ClockT, fractions.Fraction],
) -> dbm.Bound:
    lower_bound: t.Optional[dbm.Bound] = None
    for constraint in constraints:
        bound = constraint.bound
        left, right = constraint.left, constraint.right
        if not isinstance(right, dbm.ZeroClock):
            if not isinstance(left, dbm.ZeroClock):
                difference = valuation[left] - valuation[right]
                if bound.is_strict and difference >= bound.constant:
                    break
                elif difference > bound.constant:
                    break
            delta = -(valuation[right] + bound.constant)
            assert delta >= 0
            if lower_bound is None or lower_bound.constant < delta:
                lower_bound = dbm.Bound(delta, is_strict=bound.is_strict)
            elif lower_bound.constant == delta:
                if bound.is_strict and not lower_bound.is_strict:
                    lower_bound = dbm.Bound(delta, is_strict=bound.is_strict)
    return lower_bound or _ZERO_BOUND


@d.dataclass(frozen=True)
class PTASimulator(t.Generic[StateT, ActionT, ClockT]):
    pta: PTA[StateT, ActionT, ClockT]

    oracle: Oracle[StateT, ActionT, ClockT] = uniform_oracle

    def _compute_option(
        self,
        edge: Edge[StateT, ActionT, ClockT],
        valuation: t.Mapping[ClockT, fractions.Fraction],
    ) -> t.Optional[Option[StateT, ActionT, ClockT]]:
        upper_bound = _compute_upper_bound(
            edge.source.invariant | edge.guard, valuation
        )
        lower_bound = _compute_lower_bound(
            edge.source.invariant | edge.guard, valuation
        )
        # FIXME: care about strict bounds
        if upper_bound is not None and lower_bound.constant > upper_bound.constant:
            return None
        return Option(edge, time_lower_bound=lower_bound, time_upper_bound=upper_bound)

    def run(
        self, steps: t.Optional[int] = None
    ) -> t.Iterator[SimulationStep[StateT, ActionT, ClockT]]:
        clock_variables = self.pta.clock_variables
        valuation: t.Dict[ClockT, fractions.Fraction] = {}
        time = fractions.Fraction(0)
        for variable in clock_variables:
            valuation[variable] = fractions.Fraction(0)
        (current_location,) = self.pta.initial_locations
        step = 0
        while steps is None or step < steps:
            options: t.Set[Option[StateT, ActionT, ClockT]] = set()
            edges = self.pta.get_edges_from(current_location)
            for edge in edges:
                option = self._compute_option(edge, valuation)
                if option is not None:
                    options.add(option)
            if not options:
                return
            choice = self.oracle(current_location, valuation, options)
            for clock in clock_variables:
                valuation[clock] += choice.time
            time += choice.time
            destination = choice.edge.destinations.pick()
            for clock in destination.reset:
                valuation[clock] = fractions.Fraction(0)
            yield SimulationStep(
                time, current_location, choice, destination, dict(valuation)
            )
            current_location = destination.location
            step += 1
