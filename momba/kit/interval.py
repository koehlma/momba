# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses
import fractions


NumberType = t.Union[int, float, fractions.Fraction]


@dataclasses.dataclass(frozen=True)
class Interval:
    infimum: NumberType
    supremum: NumberType
    infimum_included: bool = True
    supremum_included: bool = True
