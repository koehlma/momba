# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import argparse
import itertools
import pathlib
import re

from momba import model
from momba.ext import jani
from momba.model import expressions, types
from momba.model.expressions import minimum, maximum


parser = argparse.ArgumentParser(description='Reads a track file.')
parser.add_argument('track', type=pathlib.Path, help='the map description in ASCII track format')
parser.add_argument('output', type=pathlib.Path, help='JANI output file')
parser.add_argument('--indent', type=int, default=2, help='indentation for JANI file')
parser.add_argument(
    '--allow-momba-operators',
    default=False,
    action='store_true',
    help='use JANI extension x-momba-operators'
)


def main(arguments: t.Optional[t.Sequence[str]] = None) -> None:
    namespace = parser.parse_args(arguments)

    network = build_model(namespace.track)

    namespace.output.write_bytes(
        jani.dump_model(
            network,
            indent=namespace.indent,
            allow_momba_operators=namespace.allow_momba_operators
        )
    )


def build_model(track_path: pathlib.Path) -> model.Network:
    with track_path.open('r', encoding='utf-8') as track_file:
        firstline = track_file.readline()
        track = track_file.read().replace('\n', '').strip()

    dimension = re.match(r'dim: (?P<width>\d+) (?P<height>\d+)', firstline)
    assert dimension is not None, 'invalid format: dimension missing'

    width, height = int(dimension['width']), int(dimension['height'])

    track_blank = [match.start() for match in re.finditer(r'\.', track)]
    track_blocked = [match.start() for match in re.finditer(r'x', track)]
    track_start = [match.start() for match in re.finditer(r's', track)]
    track_goal = [match.start() for match in re.finditer(r'g', track)]

    track_available = (track_start + track_goal + track_blank)
    track_available.sort()

    network = model.Network(model.ModelType.MDP)

    network.declare_constant('DIM_X', types.INT, width)
    network.declare_constant('DIM_Y', types.INT, height)
    network.declare_constant('TRACK_SIZE', types.INT, width * height)

    DIM_X = expressions.identifier('DIM_X')
    DIM_Y = expressions.identifier('DIM_Y')
    TRACK_SIZE = expressions.identifier('TRACK_SIZE')

    network.declare_variable('car_dx', types.INT)
    network.declare_variable('car_dy', types.INT)
    network.declare_variable('car_pos', types.INT[0, TRACK_SIZE - 1])

    car_dx = expressions.identifier('car_dx')
    car_dy = expressions.identifier('car_dy')
    car_pos = expressions.identifier('car_pos')

    network.restrict_initial = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_start)
    )

    car = network.create_automaton(name='car')
    location = car.create_location(initial=True)

    for dx, dy in itertools.product((-1, 0, 1), repeat=2):
        car.create_edge(
            location,
            destinations={
                model.create_destination(location, assignments={
                    'car_dx': maximum(minimum(car_dx + dx, 1), -1),
                    'car_dy': maximum(minimum(car_dy + dy, 1), -1)
                })
            },
            action='step'
        )

    controller = network.create_automaton(name='controller')
    location = controller.create_location(initial=True)

    x_coord = car_pos % DIM_X
    y_coord = car_pos // DIM_X

    controller.create_edge(
            location,
            destinations={
                model.create_destination(location, assignments={
                    'car_pos':  maximum(
                        minimum(
                            maximum(minimum(x_coord + car_dx, DIM_X - 1), 0)
                            + DIM_X * maximum(minimum(y_coord + car_dy, DIM_Y - 1), 0),
                            TRACK_SIZE - 1
                        ), 0
                    )
                })
            },
            action='step'
        )

    terminator = network.create_automaton(name='terminator')
    location = terminator.create_location(initial=True)

    def out_of_bounds_x(x: model.Expression) -> model.Expression:
        return (x >= DIM_X) | (x < 0)

    def out_of_bounds_y(y: model.Expression) -> model.Expression:
        return (y >= DIM_Y) | (y < 0)

    def out_of_bounds(x: model.Expression, y: model.Expression) -> model.Expression:
        return out_of_bounds_x(x) | out_of_bounds_y(y)

    offtrack = out_of_bounds(x_coord + car_dx, y_coord + car_dy)

    crashed = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_blocked)
    )

    goal = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_goal)
    )

    has_terminated = ~goal & ~crashed & ~offtrack

    terminator.create_edge(
        location,
        guard=has_terminated,
        action='step',
        destinations={model.create_destination(location)}
    )

    car_instance = car.create_instance()
    controller_instance = controller.create_instance()
    terminator_instance = terminator.create_instance()

    composition = network.create_composition(
        {car_instance, controller_instance, terminator_instance}
    )
    composition.create_synchronization(
        {
            car_instance: 'step',
            controller_instance: 'step',
            terminator_instance: 'step'
        }, result='step'
    )

    return network


if __name__ == '__main__':
    main()
