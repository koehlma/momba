# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from ..kit import dbm

from . import base


StateT = t.TypeVar("StateT", bound=t.Hashable)
ActionT = t.TypeVar("ActionT", bound=t.Hashable)
ClockT = t.TypeVar("ClockT", bound=t.Hashable)


@d.dataclass(frozen=True)
class Location(t.Generic[StateT, ClockT]):
    invariant: t.FrozenSet[dbm.Constraint[ClockT]]
    state: StateT


@d.dataclass(frozen=True)
class Edge(t.Generic[StateT, ActionT, ClockT]):
    source: Location[StateT, ClockT]
    action: ActionT
    guard: t.FrozenSet[dbm.Constraint[ClockT]]
    reset: t.FrozenSet[ClockT]
    destination: Location[StateT, ClockT]


class TA(
    base.TS[Location[StateT, ClockT], Edge[StateT, ActionT, ClockT]],
    t.Generic[StateT, ActionT, ClockT],
):
    pass
