# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
