# -*- coding:utf-8 -*-
#
# Copyright (C) 2020-2021, Saarland University
# Copyright (C) 2020-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>
# Copyright (C) 2020-2021, Michaela Klauck <klauck@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum
import itertools
import math
import re


from momba import model
from momba.model import expressions, types
from momba.moml import expr, prop


class TankType(enum.Enum):
    """
    An enumeration of different *tank types*.

    The actual tank size is calculate based on the size of the
    track and *capacity factor*.

    Attributes
    ----------
    capacity_factor:
        The capacity factor associated with the tank size.
    """

    SMALL = 0.5
    """ A small tank. """

    MEDIUM = 0.75
    """ A medium-sized tank. """

    LARGE = 1
    """ A large tank. """

    capacity_factor: float

    def __init__(self, capacity_factor: float) -> None:
        self.capacity_factor = capacity_factor


class AccelerationModel(t.Protocol):
    def __call__(self, acceleration: model.Expression) -> model.Expression:
        pass


class Underground(enum.Enum):
    """
    An enumeration of different *undergrounds*.

    Undergrounds introduce probabilistic noise modeling
    slippery road conditions.

    Attributes
    ----------
    acceleration_probability:
        An expression for the probability that the acceleration succeeds.
    acceleration_model:
        A function for computing the *abnormal* acceleration.
    """

    TARMAC = expr("9 / 10"), lambda a: a
    """
    A very solid non-slippery underground introducing no noise.
    """

    SAND = (
        expr("5 / 10"),
        lambda a: expr("$a > 0 ? $a - 1 : ($a < 0 ? $a + 1 : 0)", a=a),
    )
    """
    A sandy underground introducing some noise, be cautious!
    """

    ICE = expr("3 / 10"), lambda a: expr("0")
    """
    A very slippy underground.
    """

    acceleration_probability: model.Expression
    acceleration_model: AccelerationModel

    def __init__(
        self,
        acceleration_probability: model.Expression,
        acceleration_model: AccelerationModel,
    ) -> None:
        self.acceleration_probability = acceleration_probability
        self.acceleration_model = acceleration_model


@d.dataclass(frozen=True)
class Coordinate:
    """
    Represents a coordinate on the track.
    """

    x: int
    """ The :math:`x` coordinate. """

    y: int
    """ The :math:`y` coordinate. """


class CellType(enum.Enum):
    """
    An enumeration of *cell types*.
    """

    BLANK = "."
    """
    A *blank cell* where one can drive.
    """

    BLOCKED = "x"
    """
    A cell *blocked* by an obstacle.
    """

    START = "s"
    """
    A start cell.
    """

    GOAL = "g"
    """
    A goal cell.
    """


@d.dataclass(frozen=True)
class Track:
    """
    Represents a *track*.

    Attributes
    ----------
    width:
        The width of the track.
    height:
        The height of the track.
    blank_cells:
        The set of blank cells.
    blocked_cells:
        The set of blocked cells.
    start_cells:
        The set of start cells.
    goal_cells:
        The set of goal cells.
    """

    width: int
    height: int

    blank_cells: t.Set[int]
    blocked_cells: t.Set[int]
    start_cells: t.Set[int]
    goal_cells: t.Set[int]

    @property
    def size(self) -> int:
        """
        The total size, i.e., number of cells, of the track.
        """
        return self.width * self.height

    def get_cell_type(self, cell: int) -> CellType:
        """
        Retrives the type of the given *cell*.
        """
        if cell in self.blank_cells:
            return CellType.BLANK
        elif cell in self.blocked_cells:
            return CellType.BLOCKED
        elif cell in self.start_cells:
            return CellType.START
        else:
            assert cell in self.goal_cells
            return CellType.GOAL

    def cell_to_coordinate(self, cell: int) -> Coordinate:
        """
        Computes the *coordinate* of the given *cell*.
        """
        return Coordinate(cell % self.width, cell // self.width)

    def coordinate_to_cell(self, coordinate: Coordinate) -> int:
        """
        Computes the *cell* at the given *coordinate*.
        """
        return coordinate.y * self.width + coordinate.x

    @property
    def cells(self) -> t.Iterable[int]:
        """
        An iterable of the *cells* of the track.
        """
        return tuple(range(0, self.width * self.height))

    @property
    def textual_description(self) -> str:
        """
        Converts the track into is textual description.
        """
        lines = [f"dim: {self.width} {self.height}"]
        for y in range(self.height):
            lines.append(
                "".join(
                    self.get_cell_type(self.coordinate_to_cell(Coordinate(x, y))).value
                    for x in range(self.width)
                )
            )
        return "\n".join(lines)

    @classmethod
    def from_source(cls, source: str) -> Track:
        """
        Converts a textual specification of a track into a :class:`Track`.
        """
        firstline, _, remainder = source.partition("\n")
        track = "".join(line.strip() for line in remainder.splitlines())

        dimension = re.match(r"dim: (?P<height>\d+) (?P<width>\d+)", firstline)
        assert dimension is not None, "invalid format: dimension missing"

        width, height = int(dimension["width"]), int(dimension["height"])
        assert (
            len(track) == width * height
        ), "given track dimensions do not match actual track size"

        blank_cells = {match.start() for match in re.finditer(r"\.", track)}
        blocked_cells = {match.start() for match in re.finditer(r"x", track)}
        start_cells = {match.start() for match in re.finditer(r"s", track)}
        goal_cells = {match.start() for match in re.finditer(r"g", track)}
        assert len(start_cells) > 0, "no start cell specified"
        assert len(goal_cells) > 0, "no goal cell specified"

        return cls(width, height, blank_cells, blocked_cells, start_cells, goal_cells)


class FuelModel(t.Protocol):
    def __call__(
        self, scenario: Scenario, dx: model.Expression, dy: model.Expression
    ) -> model.Expression:
        pass


def fuel_model_linear(
    scenario: Scenario, dx: model.Expression, dy: model.Expression
) -> model.Expression:
    return expr("abs($dx) + abs($dy)", dx=dx, dy=dy)


def fuel_model_quadratic(
    scenario: Scenario, dx: model.Expression, dy: model.Expression
) -> model.Expression:
    return expr("$linear ** 2", linear=fuel_model_linear(scenario, dx, dy))


def fuel_model_regular(
    scenario: Scenario, dx: model.Expression, dy: model.Expression
) -> model.Expression:
    return expr(
        "(1 + $max_acceleration) + $quadratic",
        max_acceleration=scenario.max_acceleration,
        quadratic=fuel_model_quadratic(scenario, dx, dy),
    )


@d.dataclass(frozen=True)
class Scenario:
    """
    A scenario description comprising a track, start cell, tank type, underground,
    maximal speed and acceleration values, and a fuel model.
    """

    track: Track

    start_cell: int

    tank_type: TankType = TankType.LARGE
    underground: Underground = Underground.TARMAC

    max_speed: int = 1
    max_acceleration: int = 1

    fuel_model: FuelModel = fuel_model_regular

    def __post_init__(self) -> None:
        assert (
            self.start_cell in self.track.start_cells
        ), f"invalid start cell {self.start_cell}"

    @property
    def tank_size(self) -> int:
        return math.floor(
            self.tank_type.capacity_factor * 3 * len(self.track.blank_cells)
        )

    @property
    def possible_accelerations(self) -> t.Iterable[int]:
        return tuple(range(-self.max_acceleration, self.max_acceleration + 1))

    def compute_consumption(
        self, dx: model.Expression, dy: model.Expression
    ) -> model.Expression:
        return self.fuel_model(self, dx, dy)


def construct_model(scenario: Scenario) -> model.Network:
    """
    Constructs an MDP network from the provided scenario description.
    """

    ctx = model.Context(model.ModelType.MDP)
    network = ctx.create_network(name="Featured Racetrack")

    track = scenario.track

    ctx.global_scope.declare_constant("WIDTH", types.INT, value=track.width)
    ctx.global_scope.declare_constant("HEIGHT", types.INT, value=track.height)
    ctx.global_scope.declare_constant("TRACK_SIZE", types.INT, value=track.size)
    ctx.global_scope.declare_constant(
        "TANK_SIZE",
        types.INT,
        value=scenario.tank_size,
    )

    ctx.global_scope.declare_variable(
        "car_dx",
        types.INT.bound(-scenario.max_speed, scenario.max_speed),
        initial_value=0,
    )
    ctx.global_scope.declare_variable(
        "car_dy",
        types.INT.bound(-scenario.max_speed, scenario.max_speed),
        initial_value=0,
    )

    ctx.global_scope.declare_variable(
        "car_pos",
        types.INT.bound(0, track.size - 1),
        initial_value=scenario.start_cell,
    )
    ctx.global_scope.declare_variable(
        "fuel",
        types.INT.bound(0, scenario.tank_size),
        initial_value=scenario.tank_size,
    )

    step = ctx.create_action_type("step").create_pattern()

    goal_cells = track.goal_cells
    on_goal = expressions.logic_any(*(expr("car_pos == $g", g=g) for g in goal_cells))

    ctx.define_property(
        "goalProbability", prop("min({ Pmax(F($on_goal)) | initial })", on_goal=on_goal)
    )
    ctx.define_property(
        "goalProbabilityFuel",
        prop("min({ Pmax(F($on_goal and fuel > 0)) | initial })", on_goal=on_goal),
    )

    def construct_car_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="car")
        initial = automaton.create_location(initial=True)

        def update_speed(
            current: model.Expression, acceleration: expressions.ValueOrExpression
        ) -> model.Expression:
            return expr(
                "max(min($current + $acceleration, $max_speed), -$max_speed)",
                current=current,
                acceleration=acceleration,
                max_speed=scenario.max_speed,
            )

        for ax, ay in itertools.product(scenario.possible_accelerations, repeat=2):
            automaton.create_edge(
                source=initial,
                destinations={
                    model.create_destination(
                        location=initial,
                        assignments={
                            "car_dx": update_speed(expr("car_dx"), ax),
                            "car_dy": update_speed(expr("car_dy"), ay),
                        },
                        probability=scenario.underground.acceleration_probability,
                    ),
                    model.create_destination(
                        location=initial,
                        assignments={
                            "car_dx": update_speed(
                                expr("car_dx"),
                                scenario.underground.acceleration_model(ax),
                            ),
                            "car_dy": update_speed(
                                expr("car_dy"),
                                scenario.underground.acceleration_model(ay),
                            ),
                        },
                        probability=expr(
                            "1 - $p",
                            p=scenario.underground.acceleration_probability,
                        ),
                    ),
                },
                action_pattern=step,
                annotation={"ax": ax, "ay": ay},
            )

        return automaton

    def construct_tank_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="tank")
        initial = automaton.create_location(initial=True)

        car_dx, car_dy = expr("car_dx"), expr("car_dy")
        fuel_model = scenario.compute_consumption
        automaton.create_edge(
            source=initial,
            destinations={
                model.create_destination(
                    initial,
                    assignments={
                        "fuel": expr(
                            "min(TANK_SIZE, max(0, fuel - floor($consumption)))",
                            consumption=fuel_model(car_dx, car_dy),
                        )
                    },
                )
            },
            action_pattern=step,
            guard=expr(
                "fuel >= $consumption",
                consumption=scenario.compute_consumption(
                    expr("car_dx"), expr("car_dy")
                ),
            ),
        )

        return automaton

    def construct_controller_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="controller")
        initial = automaton.create_location(initial=True)

        car_x = expr("car_pos % WIDTH")
        car_y = expr("car_pos // WIDTH")

        offtrack = expr(
            "$x >= WIDTH or $x < 0 or $y >= HEIGHT or $y < 0",
            x=expr("$car_x + car_dx", car_x=car_x),
            y=expr("$car_y + car_dy", car_y=car_y),
        )

        def is_blocked_at(pos: model.Expression) -> model.Expression:
            return expressions.logic_any(
                *(
                    expr("$pos == $cell", pos=pos, cell=cell)
                    for cell in track.blocked_cells
                )
            )

        will_crash = expressions.logic_any(
            *(
                is_blocked_at(
                    expr(
                        "floor($car_x + ($speed / $max_speed) * car_dx)"
                        " + WIDTH * floor($car_y + ($speed / $max_speed) * car_dy)",
                        car_x=car_x,
                        car_y=car_y,
                        speed=speed,
                        max_speed=scenario.max_speed,
                    )
                )
                for speed in range(scenario.max_speed + 1)
            )
        )

        not_terminated = expr(
            "not ($on_goal and car_dx == 0 and car_dy == 0 and fuel == 0)"
            " and not $offtrack"
            " and not $will_crash",
            on_goal=on_goal,
            offtrack=offtrack,
            will_crash=will_crash,
        )

        next_car_pos = expr(
            """
            floor(
                max(min($car_x + car_dx, WIDTH - 1), 0)
                + (WIDTH * max(min($car_y + car_dy, HEIGHT - 1), 0))
            )
            """,
            car_x=car_x,
            car_y=car_y,
        )

        automaton.create_edge(
            source=initial,
            destinations={
                model.create_destination(
                    initial,
                    assignments={"car_pos": next_car_pos},
                )
            },
            action_pattern=step,
            guard=not_terminated,
        )

        return automaton

    car = construct_car_automaton().create_instance()
    tank = construct_tank_automaton().create_instance()
    controller = construct_controller_automaton().create_instance()

    network.create_link(
        {car: step, tank: step, controller: step},
        result=step,
    )

    return network


def generate_scenarios(
    track: Track, speed_bound: int, acceleration_bound: int
) -> t.Iterator[Scenario]:
    for start_cell in track.start_cells:
        for max_speed in range(1, speed_bound + 1):
            for max_acceleration in range(1, acceleration_bound + 1):
                for underground in Underground:
                    for tank_type in TankType:
                        yield Scenario(
                            track,
                            start_cell,
                            tank_type,
                            underground,
                            max_speed,
                            max_acceleration,
                        )
