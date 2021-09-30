# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib
import random

import click

from momba import engine, jani

from . import console, model


@click.group()
def main() -> None:
    """
    A formal model of the pen-and-paper game *Racetrack*.
    """


@main.command()
@click.argument("track_file", type=pathlib.Path)
@click.argument("output_directory", type=pathlib.Path)
@click.option(
    "--speed-bound", type=int, default=3, help="Maximal allowed speed of the car."
)
@click.option(
    "--acceleration-bound",
    type=int,
    default=2,
    help="Maximal allowed acceleration of the car.",
)
@click.option(
    "--allow-momba-operators",
    default=False,
    is_flag=True,
    help="Use JANI extension `x-momba-operators`.",
)
@click.option(
    "--indent", type=int, default=None, help="Indentation for the JANI files."
)
def generate(
    track_file: pathlib.Path,
    output_directory: pathlib.Path,
    speed_bound: int,
    acceleration_bound: int,
    allow_momba_operators: bool,
    indent: t.Optional[int],
) -> None:
    """
    Generates a family of JANI models from the provided track file.

    TRACK_FILE A Racetrack track in ASCII format.
    OUTPUT_DIRECTORY A directory to write the JANI models to.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    track = model.Track.from_source(track_file.read_text(encoding="utf-8"))

    print("Generate scenarios...")
    scenarios = tuple(model.generate_scenarios(track, speed_bound, acceleration_bound))

    print(f"Generating {len(scenarios)} models...")

    with click.progressbar(scenarios) as progressbar:
        for scenario in progressbar:
            network = model.construct_model(scenario)
            filename = (
                "car"
                f"_{scenario.max_speed}"
                f"_{scenario.max_acceleration}"
                f"_{scenario.underground.name}"
                f"_{scenario.tank_type.name}"
                f"_{scenario.start_cell}.jani"
            )
            (output_directory / filename).write_text(
                jani.dump_model(
                    network,
                    indent=indent,
                    allow_momba_operators=allow_momba_operators,
                ),
                encoding="utf-8",
            )


@main.command()
@click.argument("track_file", type=pathlib.Path)
@click.option(
    "--crazy-driver",
    "crazy_driver",
    is_flag=True,
    default=False,
    help="Drive randomly instead of interactively.",
)
@click.option("--max-speed", type=int, default=None, help="Top speed of the car.")
@click.option(
    "--max-acceleration",
    type=int,
    default=2,
    help="Top acceleration of the car.",
)
@click.option(
    "--underground",
    "underground_name",
    type=click.Choice(tuple(underground.name for underground in model.Underground)),
    default=model.Underground.TARMAC.name,
    help="The underground to drive on.",
)
@click.option(
    "--tank-type",
    "tank_type_name",
    type=click.Choice(tuple(tank_type.name for tank_type in model.TankType)),
    default=model.TankType.LARGE.name,
    help="The tank type to drive with.",
)
def race(
    track_file: pathlib.Path,
    crazy_driver: bool,
    max_speed: t.Optional[int],
    max_acceleration: int,
    underground_name: str,
    tank_type_name: str,
) -> None:
    """
    Runs an interactive simulation where you can steer the car.
    """
    track = model.Track.from_source(track_file.read_text(encoding="utf-8"))
    print("Input Track".center(track.width))

    print("=" * track.width)
    print(console.format_track(track))
    print("=" * track.width, end="\n\n")

    start_cell = model.Coordinate(
        *map(
            int,
            click.prompt(
                "Please select a start cell",
                type=click.Choice(
                    tuple(
                        f"{coordinate.x} {coordinate.y}"
                        for coordinate in sorted(track.start_cells)
                    )
                ),
                show_choices=True,
            ).split(),
        )
    )

    start_cell = random.choice(list(track.start_cells))

    scenario = model.Scenario(
        track,
        start_cell=start_cell,
        max_acceleration=max_acceleration,
        max_speed=max_speed,
        underground=model.Underground[underground_name],
        tank_type=model.TankType[tank_type_name],
    )

    print("\nBuilding model...")
    network = model.construct_model(scenario)

    # retrieve the instance of the car automaton we are going to control
    car_instance = next(
        instance for instance in network.instances if instance.automaton.name == "car"
    )

    explorer = engine.Explorer.new_discrete_time(network)

    (state,) = explorer.initial_states

    running = True

    while running:
        while not all(
            car_instance in transition.edge_vector for transition in state.transitions
        ):
            state = random.choice(state.transitions).destinations.pick().state

        car_x = state.global_env["car_x"].as_int
        car_y = state.global_env["car_y"].as_int
        car_cell = model.Coordinate(car_x, car_y)
        dx = state.global_env["car_dx"].as_int
        dy = state.global_env["car_dy"].as_int
        fuel = state.global_env["fuel"].as_int

        print(f"\ndx: {dx}, dy: {dy}, fuel: {fuel}\n")
        print(console.format_track(track, car_cell))

        if car_cell in scenario.track.goal_cells:
            print("\nGame won!")
            break

        options: t.Dict[t.Tuple[int, int], engine.Transition[engine.DiscreteTime]] = {}
        transitions = state.transitions

        if not transitions:
            print("\nGame over!")
            break

        if crazy_driver:
            # shortcut decision by letting the (random) crazy driver take the decision
            state = random.choice(transitions).destinations.pick().state
            continue

        for option in transitions:
            annotation = option.edge_vector[car_instance].annotation
            assert annotation is not None
            ax = annotation["ax"]
            ay = annotation["ay"]
            assert isinstance(ax, int) and isinstance(ay, int)
            options[(ax, ay)] = option

        print()
        chosen_ax: int = click.prompt(
            "Please choose an acceleration in x direction in range"
            f" [-{scenario.max_acceleration}, {scenario.max_acceleration}]",
            type=click.IntRange(-scenario.max_acceleration, scenario.max_acceleration),
        )
        chosen_ay: int = click.prompt(
            "Please choose an acceleration in y direction in range"
            f" [-{scenario.max_acceleration}, {scenario.max_acceleration}]",
            type=click.IntRange(-scenario.max_acceleration, scenario.max_acceleration),
        )
        decision = options[(chosen_ax, chosen_ay)]
        state = decision.destinations.pick().state


if __name__ == "__main__":
    main()
