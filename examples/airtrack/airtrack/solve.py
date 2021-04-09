# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from . import aerodynamics, airplane, atmosphere


def solve_trim(aoa: float) -> float:
    wing = airplane.WING_AREA * airplane.ARM_WING
    hstab = airplane.HSTAB_AREA * airplane.ARM_HSTAB
    stable_ratio = -wing / hstab
    wing_coefficient = aerodynamics.aoa_to_lift_coefficient(aoa)
    # this is actually a linear function if we apply all definitions
    return aerodynamics.lift_coefficient_to_aoa(stable_ratio * wing_coefficient)


def solve_altitude(aoa: float, trim: float, speed: float, mass: float) -> float:
    wing_coefficient = aerodynamics.aoa_to_lift_coefficient(aoa)
    hstab_coefficient = aerodynamics.aoa_to_lift_coefficient(aoa + trim)
    wing = wing_coefficient * airplane.WING_AREA * airplane.ARM_WING
    hstab = hstab_coefficient * airplane.HSTAB_AREA * airplane.ARM_HSTAB
    force_gravity = mass * atmosphere.GRAVITY
    density = 2 * force_gravity / (speed ** 2 * (wing + hstab))
    return atmosphere.density_to_altitude(density)
