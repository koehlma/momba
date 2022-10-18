# -*- coding: utf-8 -*-
#
# Copyright (C) 2022, Saarland University
# Copyright (C) 2022, Maximilian Köhl <koehl@cs.uni-saarland.de>


import pathlib

import click
import tomlkit

from momba import engine
from momba.engine import translator

from .builder import Scenario, Sensor, build_model


def load_scenario(scenario: pathlib.Path) -> Scenario:
    data = tomlkit.loads(scenario.read_text("utf-8"))
    return Scenario(
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
    network = build_model(load_scenario(scenario))
    output.write_text(
        translator.translate_network(network).json_network, encoding="utf-8"
    )


@main.command()
@click.argument("scenario", type=pathlib.Path)
def count(scenario: pathlib.Path) -> None:
    """
    Counts the number of zones and transitions.
    """
    network = build_model(load_scenario(scenario))
    explorer = engine.Explorer(network, time_type=engine.time.ZoneF64)
    print(f"States: {explorer.count_states()}")
    print(f"Transitions: {explorer.count_transitions()}")


if __name__ == "__main__":
    main()
