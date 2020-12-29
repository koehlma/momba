# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

import click

import racetrack

from momba import engine


FUEL_MODELS = {
    "LINEAR": racetrack.fuel_model_linear,
    "QUADRATIC": racetrack.fuel_model_quadratic,
    "REGULAR": racetrack.fuel_model_regular,
}


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
def main(
    track_file: pathlib.Path,
    start_cell: int,
    underground_name: str,
    tank_type_name: str,
    max_speed: int,
    max_acceleration: int,
    fuel_model_name: str,
) -> None:
    """
    Compiles the Racetrack model to MombaCR.
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

    print("Compiling model...")
    compiled = engine.compile_mdp(network)

    initial_states = compiled.initial_states

    print("Initial States:", len(initial_states))

    for state in initial_states:
        print(state.global_env)
        print(state.locations)
        for transition in state.transitions:
            print(transition)
            for destination in transition.destinations:
                print(destination, destination.probability)


if __name__ == "__main__":
    main()
