# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses

if t.TYPE_CHECKING:
    from . import context  # noqa: F401


@dataclasses.dataclass(frozen=True)
class Action:
    ctx: context.Context
    name: str
