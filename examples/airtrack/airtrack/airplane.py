# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d

from . import aerodynamics, atmosphere


# http://www.b737.org.uk/techspecsdetailed.htm
# values for Boeing 737-800
WING_AREA = 124.58  # [m²]
HSTAB_AREA = 32.78  # [m²]

MAX_MASS = 79_002  # [kg]

# two engines delivering 130kN each
MAX_THRUST = 130_000 * 2  # [N]

HEIGHT = 4  # [m]
LENGTH = 39  # [m]


CENTER_OF_WING = LENGTH / 2
CENTER_OF_GRAVITY = LENGTH / 3
CENTER_OF_HSTAB = LENGTH - 3

ARM_WING = CENTER_OF_WING - CENTER_OF_GRAVITY  # [m]
ARM_HSTAB = CENTER_OF_HSTAB - CENTER_OF_GRAVITY  # [m]


def compute_wing_lift(
    altitude: float,
    aoa: float,
    speed: float,
) -> float:
    coefficient = aerodynamics.aoa_to_lift_coefficient(aoa)
    density = atmosphere.altitude_to_density(altitude)
    return coefficient * (density * speed ** 2) / 2 * WING_AREA


def compute_hstab_lift(
    altitude: float,
    aoa: float,
    speed: float,
) -> float:
    coefficient = aerodynamics.aoa_to_lift_coefficient(aoa)
    density = atmosphere.altitude_to_density(altitude)
    return coefficient * (density * speed ** 2) / 2 * HSTAB_AREA


def compute_drag(
    altitude: float,
    aoa: float,
    speed: float,
) -> float:
    coefficient = aerodynamics.aoa_to_drag_coefficient(aoa)
    density = atmosphere.altitude_to_density(altitude)
    return coefficient * (density * speed ** 2) / 2 * WING_AREA


@d.dataclass
class State:
    altitude: float
    aoa: float
    trim: float
    airspeed: float
    thrust: float
    mass: float = 0.8 * MAX_MASS

    @property
    def moment_of_intertia(self) -> float:
        return 1 / 5 * self.mass * (LENGTH ** 2 + HEIGHT ** 2)

    @property
    def force_gravity(self) -> float:
        return self.mass * atmosphere.GRAVITY

    @property
    def force_wing_lift(self) -> float:
        return compute_wing_lift(self.altitude, self.aoa, self.airspeed)

    @property
    def force_hstab_lift(self) -> float:
        return compute_hstab_lift(self.altitude, self.aoa + self.trim, self.airspeed)

    @property
    def force_lift(self) -> float:
        return self.force_wing_lift + self.force_hstab_lift

    @property
    def force_thrust(self) -> float:
        return self.thrust * MAX_THRUST

    @property
    def force_drag(self) -> float:
        return compute_drag(self.altitude, self.aoa, self.airspeed)

    @property
    def pitch_acceleration(self) -> float:
        torque = self.force_wing_lift * ARM_WING + self.force_hstab_lift * ARM_HSTAB
        return torque / self.moment_of_intertia

    @property
    def horizontal_acceleration(self) -> float:
        return (self.force_thrust - self.force_drag) / self.mass

    @property
    def vertical_acceleration(self) -> float:
        return (self.force_lift - self.force_gravity) / self.mass
