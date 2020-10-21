# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib
import time

import click

import racetrack

from momba.analysis import checkers
from momba.tools import modest, storm
from momba.tools.errors import ToolError


FUEL_MODELS = {
    "LINEAR": racetrack.fuel_model_linear,
    "QUADRATIC": racetrack.fuel_model_quadratic,
    "REGULAR": racetrack.fuel_model_regular,
}

CHECKERS = {
    "MODEST": modest.checker,
    "STORM_DD": storm.checker_dd,
    "STORM_SPARSE": storm.checker_sparse,
    "CROSS": checkers.CrossChecker((modest.checker, storm.checker_dd)),
}


def print_results(result: checkers.Result) -> None:
    for name, value in result.items():
        print(f"  {name}: {float(value)}")


@click.command()
@click.argument("track_file", type=pathlib.Path)
@click.argument("start_cell", type=int)
@click.option(
    "--underground",
    "underground_name",
    type=click.Choice(tuple(underground.name for underground in racetrack.Underground)),
    default=racetrack.Underground.TARMAC.name,
)
@click.option(
    "--tank-type",
    "tank_type_name",
    type=click.Choice(tuple(tank_type.name for tank_type in racetrack.TankType)),
    default=racetrack.TankType.LARGE.name,
)
@click.option("--max-speed", type=int, default=2)
@click.option("--max-acceleration", type=int, default=2)
@click.option(
    "--fuel-model",
    "fuel_model_name",
    type=click.Choice(FUEL_MODELS.keys()),
    default="REGULAR",
)
@click.option(
    "--checker",
    "checker_names",
    type=click.Choice(CHECKERS.keys()),
    default=("CROSS",),
    multiple=True,
)
def check(
    track_file: pathlib.Path,
    start_cell: int,
    underground_name: str,
    tank_type_name: str,
    max_speed: int,
    max_acceleration: int,
    fuel_model_name: str,
    checker_names: t.Sequence[str],
) -> None:
    """
    Performs model checking on a model for the given scenario.
    """
    track = racetrack.Track.from_source(track_file.read_text(encoding="utf-8"))
    if start_cell not in track.start_cells:
        print(
            f"Invalid start cell {start_cell}. Valid start cells {track.start_cells!r}."
        )
        return
    underground = racetrack.Underground[underground_name]
    tank_type = racetrack.TankType[tank_type_name]
    fuel_model = FUEL_MODELS[fuel_model_name]

    # construct a Scenario from the given parameters
    scenario = racetrack.Scenario(
        track,
        start_cell,
        tank_type=tank_type,
        underground=underground,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        fuel_model=fuel_model,
    )

    print("Building model...")
    network = racetrack.construct_model(scenario)

    try:
        for checker_name in checker_names:
            checker = CHECKERS[checker_name]
            print(f"Invoking checker '{checker_name}'...")
            start_time = time.monotonic()
            result = checker.check(network)
            end_time = time.monotonic()
            print(f"Results after {end_time - start_time:.2f}s:")
            print_results(result)
    except ToolError as error:
        print("Error while invoking model checker:")
        print(" ".join(map(str, error.command)))
        if error.stdout is not None:
            print("Stdout:")
            print(error.stdout.decode())
        if error.stderr is not None:
            print("Stderr:")
            print(error.stderr.decode())


if __name__ == "__main__":
    check()
