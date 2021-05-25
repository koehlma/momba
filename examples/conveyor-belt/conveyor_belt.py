#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
A family of industrial automation conveyor belt models.
"""

import dataclasses as d
import typing as t

import pathlib

import click
import tomlkit

from momba import engine, model
from momba.engine import translator
from momba.moml import expr


@d.dataclass(frozen=True)
class Sensor:
    position: int

    tick_lower_bound: int
    tick_upper_bound: int

    fault_sporadic: bool


@d.dataclass(frozen=True)
class Conveyor:
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
    ctx: model.Context, conveyor: Conveyor, action_types: _ActionTypes
) -> model.Automaton:
    automaton = ctx.create_automaton(name="Conveyor")

    automaton.scope.declare_variable("t", typ=model.types.CLOCK, initial_value=0)

    location_running = automaton.create_location(
        "running",
        initial=True,
        progress_invariant=expr("t <= $bound", bound=conveyor.running_tick_upper_bound),
    )
    location_fault = automaton.create_location(
        "fault",
        progress_invariant=expr("t <= $bound", bound=conveyor.fault_tick_upper_bound),
    )

    automaton.create_edge(
        location_running,
        destinations={
            model.create_destination(
                location_running,
                assignments={
                    "position": expr(
                        "(position + 1) % $length", length=conveyor.length
                    ),
                    "t": model.ensure_expr(0),
                },
            )
        },
        guard=expr("t >= $bound", bound=conveyor.running_tick_lower_bound),
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
                        "(position + 1) % $length", length=conveyor.length
                    ),
                    "t": model.ensure_expr(0),
                },
            )
        },
        guard=expr("t >= $bound", bound=conveyor.fault_tick_lower_bound),
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


def build_model(conveyor: Conveyor) -> model.Network:
    ctx = model.Context(model.ModelType.TA)

    ctx.global_scope.declare_variable("position", typ=model.types.INT, initial_value=0)

    action_types = _build_action_types(ctx)

    conveyor_automaton = _build_conveyor_automaton(ctx, conveyor, action_types)
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
        for sensor in conveyor.sensors
    ]

    network = ctx.create_network()

    network.add_instance(conveyor_instance)
    if conveyor.fault_friction:
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


def load_scenario(scenario: pathlib.Path) -> Conveyor:
    data = tomlkit.loads(scenario.read_text("utf-8"))
    return Conveyor(
        length=data["length"],
        sensors=tuple(
            Sensor(
                position=sensor["position"],
                tick_lower_bound=sensor.get("tick_lower_bound", 100),
                tick_upper_bound=sensor.get("tick_upper_bound", 100),
                fault_sporadic=sensor.get("fault_sporadic", False),
            )
            for sensor in data.get("sensors", ())
        ),
        running_tick_lower_bound=data.get("running_tick_lower_bound", 500),
        running_tick_upper_bound=data.get("running_tick_upper_bound", 550),
        fault_tick_lower_bound=data.get("fault_tick_lower_bound", 600),
        fault_tick_upper_bound=data.get("fault_tick_upper_bound", 750),
        fault_friction=data.get("fault_friction", True),
    )


@click.group()
def main() -> None:
    """
    A family of industrial automation conveyor belt models.
    """


@main.command()
@click.argument("scenario", type=pathlib.Path)
@click.option(
    "-o", "--output", type=pathlib.Path, default=pathlib.Path("conveyor_belt.json")
)
def build(scenario: pathlib.Path, output: pathlib.Path) -> None:
    """
    Builds and outputs the model in MombaIR.
    """
    conveyor = load_scenario(scenario)
    network = build_model(conveyor)
    output.write_text(
        translator.translate_network(network).json_network, encoding="utf-8"
    )


@main.command()
@click.argument("scenario", type=pathlib.Path)
def count(scenario: pathlib.Path) -> None:
    """
    Counts the number of zones and transitions.
    """
    conveyor = load_scenario(scenario)
    network = build_model(conveyor)
    explorer = engine.Explorer(network, time_type=engine.time.ZoneF64)
    print(f"States: {explorer.count_states()}")
    print(f"Transitions: {explorer.count_transitions()}")


if __name__ == "__main__":
    main()
