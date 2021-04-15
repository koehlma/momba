# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

from . import actions, expressions


@d.dataclass(frozen=True)
class Observation:
    action_type: actions.ActionType
    arguments: t.Tuple[expressions.Expression, ...]
    probability: t.Optional[expressions.Expression]
