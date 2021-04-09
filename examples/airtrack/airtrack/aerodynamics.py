# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import math


def aoa_to_lift_coefficient(aoa: float) -> float:
    aoa_in_degrees = math.degrees(aoa)

    linear = 0.18 * aoa_in_degrees + 0.4

    if aoa_in_degrees > 10:
        stall_factor = 0.5 + 0.1 * aoa_in_degrees - 0.005 * aoa_in_degrees ** 2
    elif aoa_in_degrees < -10:
        stall_factor = 0.5 - 0.1 * aoa_in_degrees - 0.005 * aoa_in_degrees ** 2
    else:
        stall_factor = 1.0

    return linear * stall_factor


def lift_coefficient_to_aoa(coefficient: float) -> float:
    aoa_in_degrees = (coefficient - 0.4) / 0.18
    assert -10 < aoa_in_degrees < 10, "AoA outside of reversible range"
    return math.radians(aoa_in_degrees)


def aoa_to_drag_coefficient(aoa: float) -> float:
    aoa_in_degrees = math.degrees(aoa)
    return (
        4e-09 * aoa_in_degrees ** 6
        + 4e-08 * aoa_in_degrees ** 5
        - 7e-07 * aoa_in_degrees ** 4
        - 1e-05 * aoa_in_degrees ** 3
        + 0.0009 * aoa_in_degrees ** 2
        + 0.0033 * aoa_in_degrees
        + 0.0301
    )
