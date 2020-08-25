# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import fractions

from . import base

LocationT = t.TypeVar("LocationT", bound=t.Hashable)
ActionT = t.TypeVar("ActionT", bound=t.Hashable)
ClockVariableT = t.TypeVar("ClockVariableT", bound=t.Hashable)


@d.dataclass(frozen=True)
class ZeroClock:
    pass


ZERO_CLOCK = ZeroClock()


@d.dataclass(frozen=True)
class Bound:
    bound: fractions.Fraction
    strict: bool = False


SomeClock = t.Union[ClockVariableT, ZeroClock]


@d.dataclass(frozen=True)
class Constraint(t.Generic[ClockVariableT]):
    left: t.Union[ClockVariableT, ZeroClock]
    right: t.Union[ClockVariableT, ZeroClock]
    bound: Bound


Constraints = t.AbstractSet[Constraint[ClockVariableT]]


@d.dataclass(frozen=True)
class Location(t.Generic[LocationT, ClockVariableT]):
    invariant: t.FrozenSet[Constraint[ClockVariableT]]
    state: LocationT


@d.dataclass(frozen=True)
class Edge(t.Generic[LocationT, ActionT, ClockVariableT]):
    source: Location[LocationT, ClockVariableT]
    action: ActionT
    guard: t.FrozenSet[Constraint[ClockVariableT]]
    reset: t.FrozenSet[ClockVariableT]
    destination: Location[LocationT, ClockVariableT]


class TA(
    base.TS[
        Location[LocationT, ClockVariableT], Edge[LocationT, ActionT, ClockVariableT]
    ],
    t.Generic[LocationT, ActionT, ClockVariableT],
):
    pass
