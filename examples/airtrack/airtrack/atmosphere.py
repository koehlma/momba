# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>


IDEAL_GAS_CONSTANT = 8.31446  # [J/(mol*K)]

SEE_LEVEL_PRESSURE = 101_325  # [Pa]
SEE_LEVEL_TEMPERATURE = 288.15  # [K]

TEMPERATURE_LAPSE_RATE = 0.0065  # [K/m]

GRAVITY = 9.80665  # [m/s^2]
MOLAR_MASS = 0.0289652  # [kg/mol]


def temperature_at(altitude: float) -> float:
    # [K] - [K/m] * [m] = [K]
    return SEE_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * altitude


def pressure_at(altitude: float) -> float:
    # [K/m] * [m] / [K] = []
    base = 1 - (TEMPERATURE_LAPSE_RATE * altitude / SEE_LEVEL_TEMPERATURE)
    #   [m/s^2] * [kg/mol] / ([J/(mol*K)] * [K/m])
    # = [m/s^2] * [kg/mol] / [J/(mol*m)]
    # = [m/s^2] * [kg/mol] * [mol*m/J]
    # = [m/s^2] * [kg] * [m/J]
    # = [m/s^2] * [kg] * [m*s^2/(kg*m^2)]
    # = []
    exponent = GRAVITY * MOLAR_MASS / (IDEAL_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
    coefficient = base ** exponent
    return coefficient * SEE_LEVEL_PRESSURE


def altitude_to_density(altitude: float) -> float:
    #   [Pa] * [kg/mol] / ([J/(mol*K)] * [K])
    # = [Pa] * [kg/mol] / [J/mol]
    # = [Pa] * [kg/mol] * [mol/J]
    # = [Pa] * [kg/J]
    # = [kg/(m*s^2)] * [kg*s^2/(kg*m^2)]
    # = [kg/m^3]
    return (
        pressure_at(altitude)
        * MOLAR_MASS
        / (IDEAL_GAS_CONSTANT * temperature_at(altitude))
    )


def density_to_altitude(density: float) -> float:
    from scipy.optimize import fsolve

    # let's just solve this numerically
    return fsolve(lambda altitude: altitude_to_density(altitude) - density, 10_000.0)[0]
