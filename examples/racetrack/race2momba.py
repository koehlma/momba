#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import argparse
import enum
import itertools
import pathlib
import re

from momba import model
from momba.ext import jani
from momba.model import expressions, properties, types
from momba.model.expressions import minimum, maximum


class Underground(enum.Enum):
    TARMAC = "tarmac"
    ICE = "ice"
    SAND = "sand"


parser = argparse.ArgumentParser(description="Reads a track file.")
parser.add_argument(
    "track", type=pathlib.Path, help="the map description in ASCII track format"
)
parser.add_argument("output", type=pathlib.Path, help="JANI output directory")
parser.add_argument("--max_speed", type=int, default=3, help="maximal speed of the car")
parser.add_argument(
    "--max_acc", type=int, default=3, help="maximal acceleration in one step"
)
parser.add_argument("--indent", type=int, default=2, help="indentation for JANI file")
parser.add_argument(
    "--allow-momba-operators",
    default=False,
    action="store_true",
    help="use JANI extension x-momba-operators",
)


def main(arguments: t.Optional[t.Sequence[str]] = None) -> None:
    namespace = parser.parse_args(arguments)

    namespace.output.mkdir(parents=True, exist_ok=True)

    for car_max_speed in range(1, namespace.max_speed + 1):
        for car_max_acc in range(1, namespace.max_acc + 1):
            for underground in Underground:
                network = build_model(
                    namespace.track, car_max_speed, car_max_acc, underground
                )

                out = (
                    namespace.output
                    / f"car_{car_max_speed}_{car_max_acc}_{underground.value}.jani"
                )

                out.write_bytes(
                    jani.dump_model(
                        network,
                        indent=namespace.indent,
                        allow_momba_operators=namespace.allow_momba_operators,
                    )
                )
    print("done")


def build_model(
    track_path: pathlib.Path, max_speed: int, max_acc: int, underground: Underground
) -> model.Network:
    with track_path.open("r", encoding="utf-8") as track_file:
        firstline = track_file.readline()
        track = track_file.read().replace("\n", "").strip()

    dimension = re.match(r"dim: (?P<width>\d+) (?P<height>\d+)", firstline)
    assert dimension is not None, "invalid format: dimension missing"

    width, height = int(dimension["width"]), int(dimension["height"])
    assert (
        len(track) == width * height
    ), "given track dimensions do not match actual track size"

    track_blank = [match.start() for match in re.finditer(r"\.", track)]
    track_blocked = [match.start() for match in re.finditer(r"x", track)]
    track_start = [match.start() for match in re.finditer(r"s", track)]
    track_goal = [match.start() for match in re.finditer(r"g", track)]
    assert len(track_start) > 0, "no start field specified"
    assert len(track_goal) > 0, "no goal field specified"

    def acc_prob(ground: Underground) -> model.Expression:
        if ground is Underground.TARMAC:
            return expressions.real_div(7, 8)
        elif ground is Underground.SAND:
            return expressions.real_div(1, 3)
        elif ground is Underground.ICE:
            return expressions.real_div(1, 6)
        else:
            raise NotImplementedError("This underground is not specified")

    def acc_underground(ground: Underground, x: model.Expression) -> model.Expression:
        if ground is Underground.TARMAC:
            return x
        if ground is Underground.SAND:
            return expressions.ite(
                expressions.gt(x, 0), expressions.sub(x, 1), expressions.add(x, 1)
            )
        if ground is Underground.ICE:
            return expressions.convert(0)

    ctx = model.Context(model.ModelType.MDP)
    network = ctx.create_network()

    ctx.global_scope.declare_constant("DIM_X", types.INT, width)
    ctx.global_scope.declare_constant("DIM_Y", types.INT, height)
    ctx.global_scope.declare_constant("TRACK_SIZE", types.INT, width * height)

    DIM_X = expressions.identifier("DIM_X")
    DIM_Y = expressions.identifier("DIM_Y")
    TRACK_SIZE = expressions.identifier("TRACK_SIZE")

    ctx.global_scope.declare_variable("car_dx", types.INT)
    ctx.global_scope.declare_variable("car_dy", types.INT)
    ctx.global_scope.declare_variable(
        "car_pos", types.INT[0, expressions.sub(TRACK_SIZE, 1)]
    )

    car_dx = expressions.identifier("car_dx")
    car_dy = expressions.identifier("car_dy")
    car_pos = expressions.identifier("car_pos")

    in_goal = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(g)) for g in track_goal)
    )
    prop = properties.maxProb(in_goal)
    ctx.define_property(prop, name="goalProb")

    network.restrict_initial = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_start)
    )

    car = network.create_automaton(name="car")
    location = car.create_location(initial=True)

    def new_speed(
        current: model.Expression, change: model.Expression
    ) -> model.Expression:
        return expressions.maximum(
            expressions.minimum(expressions.add(current, change), max_speed), -max_speed
        )

    for ax, ay in itertools.product(range(-max_acc, max_acc + 1), repeat=2):
        car.create_edge(
            location,
            destinations={
                model.create_destination(
                    location,
                    assignments={
                        "car_dx": new_speed(car_dx, ax),
                        "car_dy": new_speed(car_dy, ay),
                    },
                    probability=acc_prob(underground),
                ),
                model.create_destination(
                    location,
                    assignments={
                        "car_dx": new_speed(car_dx, acc_underground(underground, ax)),
                        "car_dy": new_speed(car_dy, acc_underground(underground, ay)),
                    },
                    probability=expressions.sub(1, acc_prob(underground)),
                ),
            },
            action="step",
        )

    controller = network.create_automaton(name="controller")
    location = controller.create_location(initial=True)

    x_coord = expressions.mod(car_pos, DIM_X)
    y_coord = expressions.floor_div(car_pos, DIM_X)

    def out_of_bounds_x(x: model.Expression) -> model.Expression:
        return expressions.lor(expressions.ge(x, DIM_X), (expressions.lt(x, 0)))

    def out_of_bounds_y(y: model.Expression) -> model.Expression:
        return expressions.lor(expressions.ge(y, DIM_Y), (expressions.lt(y, 0)))

    def out_of_bounds(x: model.Expression, y: model.Expression) -> model.Expression:
        return expressions.lor(out_of_bounds_x(x), out_of_bounds_y(y))

    offtrack = out_of_bounds(
        expressions.add(x_coord, car_dx), expressions.add(y_coord, car_dy)
    )

    goal = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_goal)
    )

    blocked = expressions.lor(
        *(expressions.eq(car_pos, expressions.convert(pos)) for pos in track_blocked)
    )

    # disjuncts = []

    # for x in range(0, DIM_X):
    #     for y in range(0, DIM_Y):
    #         for dx in range(-max_speed, max_speed+1):
    #             for dy in range(-max_speed, max_speed+1):
    #                 if ((x+dx >= 0) & (x+dx < DIM_X) & (y+dy >= 0) & (y+dy < DIM_Y)):
    #                     if(checkcollision(x, y, dx, dy, track, DIM_Y)):
    #                         disj = expressions.land(
    #                             expressions.eq(car_pos, expressions.convert(x+DIM_Y*y)),
    #                             expressions.eq(car_dx, expressions.convert(dx)),
    #                             expressions.eq(car_dy, expressions.convert(dy)))
    #                         disjuncts.append(disj)
    # crashed = expressions.lor(disjuncts)

    def is_blocked_at(pos: model.Expression) -> model.Expression:
        return expressions.lor(
            *(
                expressions.eq(pos, expressions.convert(val))
                for i, val in enumerate(track_blocked)
            )
        )

    # used floor instead of round sometimes because JANI only knows floor and ceil
    car_will_crash = expressions.lor(
        *(
            is_blocked_at(
                expressions.add(
                    expressions.floor(
                        expressions.add(
                            expressions.mod(car_pos, DIM_Y),
                            expressions.mul(
                                (
                                    expressions.real_div(
                                        expressions.convert(i),
                                        expressions.convert(max_speed),
                                    )
                                ),
                                car_dx,
                            ),
                        )
                    ),
                    expressions.mul(
                        DIM_Y,
                        expressions.floor(
                            expressions.add(
                                expressions.floor(expressions.real_div(car_pos, DIM_Y)),
                                expressions.mul(
                                    (expressions.real_div(i, max_speed)), car_dy
                                ),
                            )
                        ),
                    ),
                )
            )
            for i in range(max_speed + 1)
        )
    )

    not_terminated = expressions.land(
        expressions.lnot(goal),
        expressions.lnot(offtrack),
        expressions.lnot(car_will_crash),
    )

    controller.create_edge(
        location,
        destinations={
            model.create_destination(
                location,
                assignments={
                    "car_pos": maximum(
                        minimum(
                            expressions.add(
                                maximum(
                                    minimum(
                                        expressions.add(x_coord, car_dx),
                                        expressions.sub(DIM_X, 1),
                                    ),
                                    0,
                                ),
                                expressions.mul(
                                    DIM_X,
                                    maximum(
                                        minimum(
                                            expressions.add(y_coord, car_dy),
                                            expressions.sub(DIM_Y, 1),
                                        ),
                                        0,
                                    ),
                                ),
                            ),
                            expressions.sub(TRACK_SIZE, 1),
                        ),
                        0,
                    )
                },
            )
        },
        action="step",
        guard=not_terminated,
    )

    car_instance = car.create_instance()
    controller_instance = controller.create_instance()

    composition = network.create_composition({car_instance, controller_instance})
    composition.create_synchronization(
        {car_instance: "step", controller_instance: "step"}, result="step"
    )

    return network


if __name__ == "__main__":
    main()
