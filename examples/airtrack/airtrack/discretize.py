# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

import math

from . import solve


MIN_AOA = math.radians(-5)  # [rad]
MAX_AOA = math.radians(5)  # [rad]


@d.dataclass(frozen=True)
class StableAoAConfiguration:
    aoa: float
    trim: float


def discretize_aoa(granularity: int = 500) -> t.Sequence[StableAoAConfiguration]:
    step_size = (MAX_AOA - MIN_AOA) / (granularity - 1)
    return [
        StableAoAConfiguration(pitch, solve.solve_trim(pitch))
        for pitch in (MIN_AOA + index * step_size for index in range(granularity))
    ]


MIN_AIRSPEED = 50.0  # [m/s]
MAX_AIRSPEED = 240.0  # [m/s]


def discretize_airspeed(granularity: int = 200) -> t.Sequence[float]:
    step_size = (MAX_AIRSPEED - MIN_AIRSPEED) / (granularity - 1)
    return [MIN_AIRSPEED + index * step_size for index in range(granularity)]
