# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>
# Copyright (C) 2020, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import pathlib
import random

import click

import colorama

from momba.explore import engine

from racetrack import Scenario, Coordinate, Track, CellType, construct_model


colorama.init()


_BACKGROUND_COLORS = {
    CellType.BLANK: colorama.Back.BLACK,
    CellType.BLOCKED: colorama.Back.RED,
    CellType.GOAL: colorama.Back.GREEN,
    CellType.START: colorama.Back.BLUE,
}


def format_cell(track: Track, cell: int, car: t.Optional[int] = None) -> str:
    typ = track.get_cell_type(cell)
    background = _BACKGROUND_COLORS[typ]
    car_cell = cell == car
    if car_cell:
        foreground = colorama.Fore.RED if typ is CellType.GOAL else colorama.Fore.YELLOW
    else:
        foreground = (
            colorama.Fore.WHITE if typ is CellType.BLANK else colorama.Fore.BLACK
        )
    symbol = "*" if car_cell else "."
    return f"{background}{foreground}{symbol}{colorama.Style.RESET_ALL}"


def format_track(track: Track, car: t.Optional[int] = None) -> str:
    lines: t.List[str] = []
    for y in range(track.height):
        lines.append(
            "".join(
                format_cell(track, track.coordinate_to_cell(Coordinate(x, y)), car=car)
                for x in range(track.width)
            )
        )
    return "\n".join(lines)


@click.command()
@click.argument("track_file", type=pathlib.Path)
@click.option(
    "--crazy-driver",
    "crazy_driver",
    is_flag=True,
    default=False,
    help="Drive randomly instead of interactively.",
)
def race(track_file: pathlib.Path, crazy_driver: bool) -> None:
    track = Track.from_source(track_file.read_text(encoding="utf-8"))
    print("Input Track".center(track.width))

    print("=" * track.width)
    print(format_track(track))
    print("=" * track.width, end="\n\n")

    start_cell = int(
        click.prompt(
            "Please select a start cell",
            type=click.Choice(tuple(map(str, track.start_cells))),
            show_choices=True,
        )
    )

    scenario = Scenario(track, start_cell=start_cell, max_acceleration=2, max_speed=2)

    print("\nBuilding model...")
    network = construct_model(scenario)

    # retrieve the instance of the car automaton we are going to control
    car_instance = next(
        instance for instance in network.instances if instance.automaton.name == "car"
    )

    mdp = engine.MombaMDP(network)

    (state,) = mdp.initial_locations

    while True:
        car_pos = state.binding["car_pos"].as_int
        dx = state.binding["car_dx"].as_int
        dy = state.binding["car_dy"].as_int
        fuel = state.binding["fuel"].as_int

        print(f"\ndx: {dx}, dy: {dy}, fuel: {fuel}\n")
        print(format_track(track, car_pos))

        if car_pos in scenario.track.goal_cells:
            print("\nGame won!")
            break

        options: t.Dict[t.Tuple[int, int], engine.MDPEdge] = {}
        edges = mdp.get_edges_from(state)

        if not edges:
            print("\nGame over!")
            break

        if crazy_driver:
            # shortcut decision by letting the (random) crazy driver take the decision
            state = random.choice(tuple(edges)).destinations.pick()
            continue

        for option in edges:
            assert option.action is not None
            annotation = option.action.vector[car_instance].annotation
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
        state = decision.destinations.pick()


if __name__ == "__main__":
    race()
