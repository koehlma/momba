# -*- coding: utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

import dataclasses as d
import typing as t

from momba import model
from momba.moml import expr


@d.dataclass(frozen=True)
class Sensor:
    position: int

    tick_lower_bound: int
    tick_upper_bound: int

    fault_sporadic: bool


@d.dataclass(frozen=True)
class Scenario:
    length: int
    sensors: t.Tuple[Sensor, ...]

    running_tick_lower_bound: int
    running_tick_upper_bound: int

    fault_tick_lower_bound: int
    fault_tick_upper_bound: int

    fault_friction: bool


@d.dataclass(frozen=True)
class _ActionTypes:
    trigger: model.ActionType

    fault_friction: model.ActionType
    fault_sporadic: model.ActionType


def _build_action_types(ctx: model.Context) -> _ActionTypes:
    return _ActionTypes(
        trigger=ctx.create_action_type(
            "trigger",
            parameters=[
                model.actions.ActionParameter(
                    model.types.INT, comment="the position of the sensor"
                )
            ],
        ),
        # the conveyor slows down as a result of increased friction
        fault_friction=ctx.create_action_type("fault_friction"),
        # the sensor sporadically triggers without an item being present
        fault_sporadic=ctx.create_action_type(
            "fault_sporadic",
            parameters=[
                model.actions.ActionParameter(
                    model.types.INT, comment="the position of the sensor"
                )
            ],
        ),
    )


def _build_conveyor_automaton(
    ctx: model.Context, scenario: Scenario, action_types: _ActionTypes
) -> model.Automaton:
    automaton = ctx.create_automaton(name="Conveyor")

    automaton.scope.declare_variable("t", typ=model.types.CLOCK, initial_value=0)

    location_running = automaton.create_location(
        "running",
        initial=True,
        progress_invariant=expr("t <= $bound", bound=scenario.running_tick_upper_bound),
    )
    location_fault = automaton.create_location(
        "fault",
        progress_invariant=expr("t <= $bound", bound=scenario.fault_tick_upper_bound),
    )

    automaton.create_edge(
        location_running,
        destinations={
            model.create_destination(
                location_running,
                assignments={
                    "position": expr(
                        "(position + 1) % $length", length=scenario.length
                    ),
                    "t": model.ensure_expr(0),
                },
            )
        },
        guard=expr("t >= $bound", bound=scenario.running_tick_lower_bound),
    )
    automaton.create_edge(
        location_running,
        destinations={model.create_destination(location_fault)},
        action_pattern=action_types.fault_friction.create_pattern(),
    )
    automaton.create_edge(
        location_fault,
        destinations={
            model.create_destination(
                location_fault,
                assignments={
                    "position": expr(
                        "(position + 1) % $length", length=scenario.length
                    ),
                    "t": model.ensure_expr(0),
                },
            )
        },
        guard=expr("t >= $bound", bound=scenario.fault_tick_lower_bound),
    )

    return automaton


def _build_sensor_automaton(
    ctx: model.Context, action_types: _ActionTypes
) -> model.Automaton:
    automaton = ctx.create_automaton(name="Sensor")

    automaton.declare_parameter("SENSOR_POSITION", typ=model.types.INT)
    automaton.declare_parameter("TICK_LOWER_BOUND", typ=model.types.INT)
    automaton.declare_parameter("TICK_UPPER_BOUND", typ=model.types.INT)

    automaton.scope.declare_variable(
        "active", typ=model.types.BOOL, initial_value=False
    )
    automaton.scope.declare_variable(
        "sporadic",
        typ=model.types.BOOL,
        initial_value=False,
    )
    automaton.scope.declare_variable("t", typ=model.types.CLOCK, initial_value=0)

    location_sensing = automaton.create_location(
        "sensing", initial=True, progress_invariant=expr("t <= TICK_UPPER_BOUND")
    )

    automaton.create_edge(
        location_sensing,
        destinations={
            model.create_destination(
                location_sensing,
                assignments={
                    "t": model.ensure_expr(0),
                    "active": model.ensure_expr(True),
                },
            )
        },
        guard=expr(
            "t >= TICK_LOWER_BOUND and ((position == SENSOR_POSITION and not active) or sporadic)"
        ),
        action_pattern=action_types.trigger.create_pattern(
            model.actions.WriteArgument(expr("SENSOR_POSITION"))
        ),
    )
    automaton.create_edge(
        location_sensing,
        destinations={
            model.create_destination(
                location_sensing,
                assignments={
                    "t": model.ensure_expr(0),
                },
            )
        },
        guard=expr("t >= TICK_LOWER_BOUND and position == SENSOR_POSITION and active"),
    )
    automaton.create_edge(
        location_sensing,
        destinations={
            model.create_destination(
                location_sensing,
                assignments={
                    "t": model.ensure_expr(0),
                    "active": model.ensure_expr(False),
                },
            )
        },
        guard=expr("t >= TICK_LOWER_BOUND and position != SENSOR_POSITION"),
    )
    automaton.create_edge(
        location_sensing,
        destinations={
            model.create_destination(
                location_sensing,
                assignments={"sporadic": model.ensure_expr(True)},
            )
        },
        guard=expr("not sporadic"),
        action_pattern=action_types.fault_sporadic.create_pattern(
            model.actions.WriteArgument(expr("SENSOR_POSITION"))
        ),
    )

    return automaton


def build_model(scenario: Scenario) -> model.Network:
    ctx = model.Context(model.ModelType.TA)

    ctx.global_scope.declare_variable("position", typ=model.types.INT, initial_value=0)

    action_types = _build_action_types(ctx)

    conveyor_automaton = _build_conveyor_automaton(ctx, scenario, action_types)
    sensor_automaton = _build_sensor_automaton(ctx, action_types)

    conveyor_instance = conveyor_automaton.create_instance()

    sensor_instances = [
        (
            sensor,
            sensor_automaton.create_instance(
                arguments=(
                    sensor.position,
                    sensor.tick_lower_bound,
                    sensor.tick_upper_bound,
                )
            ),
        )
        for sensor in scenario.sensors
    ]

    network = ctx.create_network()

    network.add_instance(conveyor_instance)
    if scenario.fault_friction:
        network.create_link(
            {
                conveyor_instance: action_types.fault_friction.create_pattern(),
            },
            result=action_types.fault_friction.create_pattern(),
        )

    for sensor, instance in sensor_instances:
        network.create_link(
            {
                instance: action_types.trigger.create_pattern(
                    model.actions.GuardArgument("x")
                )
            },
            result=action_types.trigger.create_pattern(
                model.actions.GuardArgument("x")
            ),
        )
        if sensor.fault_sporadic:
            network.create_link(
                {
                    instance: action_types.fault_sporadic.create_pattern(
                        model.actions.GuardArgument("x")
                    )
                },
                result=action_types.fault_sporadic.create_pattern(
                    model.actions.GuardArgument("x")
                ),
            )

    return network
