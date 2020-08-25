# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import fractions
import random

from ..utils.distribution import Distribution

from . import base, ta


StateT = t.TypeVar("StateT", bound=t.Hashable)
ActionT = t.TypeVar("ActionT", bound=t.Hashable)
ClockVariableT = t.TypeVar("ClockVariableT", bound=t.Hashable)


@d.dataclass(frozen=True)
class Destination(t.Generic[StateT, ClockVariableT]):
    reset: t.FrozenSet[ClockVariableT]
    location: ta.Location[StateT, ClockVariableT]


@d.dataclass(frozen=True)
class Edge(t.Generic[StateT, ActionT, ClockVariableT]):
    source: ta.Location[StateT, ClockVariableT]
    action: t.Optional[ActionT]
    guard: t.FrozenSet[ta.Constraint[ClockVariableT]]
    destinations: Distribution[Destination[StateT, ClockVariableT]]


class PTA(
    base.TS[ta.Location[StateT, ClockVariableT], Edge[StateT, ActionT, ClockVariableT]],
    t.Generic[StateT, ActionT, ClockVariableT],
):
    @property
    @abc.abstractmethod
    def clock_variables(self) -> t.AbstractSet[ClockVariableT]:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class SimulationStep(t.Generic[StateT, ActionT, ClockVariableT]):
    time: fractions.Fraction
    source: ta.Location[StateT, ClockVariableT]
    decision: Decision[StateT, ActionT, ClockVariableT]
    destination: Destination[StateT, ClockVariableT]
    valuation: t.Mapping[ClockVariableT, fractions.Fraction]


@d.dataclass(frozen=True)
class Option(t.Generic[StateT, ActionT, ClockVariableT]):
    edge: Edge[StateT, ActionT, ClockVariableT]
    time_lower_bound: ta.Bound
    time_upper_bound: t.Optional[ta.Bound] = None


@d.dataclass(frozen=True)
class Decision(t.Generic[StateT, ActionT, ClockVariableT]):
    edge: Edge[StateT, ActionT, ClockVariableT]
    time: fractions.Fraction


class Oracle(t.Protocol[StateT, ActionT, ClockVariableT]):
    def __call__(
        self,
        location: ta.Location[StateT, ClockVariableT],
        valuation: t.Mapping[ClockVariableT, fractions.Fraction],
        options: t.AbstractSet[Option[StateT, ActionT, ClockVariableT]],
    ) -> Decision[StateT, ActionT, ClockVariableT]:
        pass


def uniform_oracle(
    location: ta.Location[StateT, ClockVariableT],
    valuation: t.Mapping[ClockVariableT, fractions.Fraction],
    options: t.AbstractSet[Option[StateT, ActionT, ClockVariableT]],
) -> Decision[StateT, ActionT, ClockVariableT]:
    option = random.choice(list(options))
    assert (
        option.time_upper_bound is not None
    ), "infinite time upper bounds not supported by the uniform oracle"
    time = option.time_lower_bound.bound + fractions.Fraction(random.random()) * (
        option.time_upper_bound.bound - option.time_lower_bound.bound
    )
    return Decision(option.edge, time)


def _compute_upper_bound(
    constraints: t.Iterable[ta.Constraint[ClockVariableT]],
    valuation: t.Mapping[ClockVariableT, fractions.Fraction],
) -> t.Optional[ta.Bound]:
    upper_bound: t.Optional[ta.Bound] = None
    for constraint in constraints:
        if constraint.right != ta.ZERO_CLOCK:
            continue
        clock = constraint.left
        if isinstance(clock, ta.ZeroClock):
            continue
        difference = constraint.bound.bound - valuation[clock]
        if upper_bound is None or difference < upper_bound.bound:
            upper_bound = ta.Bound(difference, strict=constraint.bound.strict)
        elif difference == upper_bound.bound:
            if not upper_bound.strict and constraint.bound.strict:
                upper_bound = ta.Bound(difference, strict=constraint.bound.strict)
    return upper_bound


_ZERO_BOUND = ta.Bound(fractions.Fraction(0), strict=False)


def _compute_lower_bound(
    constraints: t.Iterable[ta.Constraint[ClockVariableT]],
    valuation: t.Mapping[ClockVariableT, fractions.Fraction],
) -> ta.Bound:
    lower_bound: t.Optional[ta.Bound] = None
    for constraint in constraints:
        bound = constraint.bound
        left, right = constraint.left, constraint.right
        if not isinstance(right, ta.ZeroClock):
            if not isinstance(left, ta.ZeroClock):
                difference = valuation[left] - valuation[right]
                if bound.strict and difference >= bound.bound:
                    break
                elif difference > bound.bound:
                    break
            delta = -(valuation[right] + bound.bound)
            assert delta >= 0
            if lower_bound is None or lower_bound.bound < delta:
                lower_bound = ta.Bound(delta, strict=bound.strict)
            elif lower_bound.bound == delta:
                if bound.strict and not lower_bound.strict:
                    lower_bound = ta.Bound(delta, strict=bound.strict)
    return lower_bound or _ZERO_BOUND


@d.dataclass(frozen=True)
class PTASimulator(t.Generic[StateT, ActionT, ClockVariableT]):
    pta: PTA[StateT, ActionT, ClockVariableT]

    oracle: Oracle[StateT, ActionT, ClockVariableT] = uniform_oracle

    def _compute_option(
        self,
        edge: Edge[StateT, ActionT, ClockVariableT],
        valuation: t.Mapping[ClockVariableT, fractions.Fraction],
    ) -> t.Optional[Option[StateT, ActionT, ClockVariableT]]:
        upper_bound = _compute_upper_bound(
            edge.source.invariant | edge.guard, valuation
        )
        lower_bound = _compute_lower_bound(
            edge.source.invariant | edge.guard, valuation
        )
        # FIXME: care about strict bounds
        if upper_bound is not None and lower_bound.bound > upper_bound.bound:
            return None
        return Option(edge, time_lower_bound=lower_bound, time_upper_bound=upper_bound)

    def run(self) -> t.Iterator[SimulationStep[StateT, ActionT, ClockVariableT]]:
        clock_variables = self.pta.clock_variables
        valuation: t.Dict[ClockVariableT, fractions.Fraction] = {}
        time = fractions.Fraction(0)
        for variable in clock_variables:
            valuation[variable] = fractions.Fraction(0)
        (current_location,) = self.pta.initial_locations
        while True:
            options: t.Set[Option[StateT, ActionT, ClockVariableT]] = set()
            for edge in self.pta.get_edges_from(current_location):
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
