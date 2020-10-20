# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from . import base


LocationT = t.TypeVar("LocationT", bound=t.Hashable)
ActionT = t.TypeVar("ActionT", bound=t.Hashable)


@d.dataclass(frozen=True)
class Edge(t.Generic[LocationT, ActionT]):
    source: LocationT
    action: ActionT
    destination: LocationT


class LTS(
    base.TS[LocationT, Edge[LocationT, ActionT]], t.Generic[LocationT, ActionT],
):
    pass
