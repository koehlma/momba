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

    SLIPPERY_TARMAC = expr("9 / 10"), lambda a: expr("0")
    """
    A solid but somewhat slippery underground.
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


@d.dataclass(frozen=True, order=True)
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

    BLANK = ".", 0
    """
    A *blank cell* where one can drive.
    """

    BLOCKED = "x", 1
    """
    A cell *blocked* by an obstacle.
    """

    START = "s", 2
    """
    A start cell.
    """

    GOAL = "g", 3
    """
    A goal cell.
    """

    symbol: str
    number: int

    def __init__(self, symbol: str, number: int) -> None:
        self.symbol = symbol
        self.number = number


class Direction(enum.Enum):
    NORTH = (0, -1)
    NORTH_EAST = (1, -1)
    EAST = (1, 0)
    SOUTH_EAST = (1, 1)
    SOUTH = (0, 1)
    SOUTH_WEST = (-1, 1)
    WEST = (-1, 0)
    NORTH_WEST = (-1, -1)

    delta: Coordinate

    def __init__(self, delta_x: int, delta_y: int) -> None:
        self.delta = Coordinate(delta_x, delta_y)

    @property
    def distance_variable(self) -> str:
        return f"dist_{self.name.lower()}"


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

    blank_cells: t.FrozenSet[Coordinate]
    blocked_cells: t.FrozenSet[Coordinate]
    start_cells: t.FrozenSet[Coordinate]
    goal_cells: t.FrozenSet[Coordinate]

    def get_cell_type(self, cell: Coordinate) -> CellType:
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

    @property
    def textual_description(self) -> str:
        """
        Converts the track into its textual description.
        """
        lines = [f"dim: {self.width} {self.height}"]
        for y in range(self.height):
            lines.append(
                "".join(
                    self.get_cell_type(Coordinate(x, y)).value
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
        dimension = re.match(r"dim: (?P<height>\d+) (?P<width>\d+)", firstline)
        assert dimension is not None, "invalid format: dimension missing"
        width, height = int(dimension["width"]), int(dimension["height"])

        track = [
            list(line.strip())
            for line in remainder.splitlines(keepends=False)
            if line.strip()
        ]

        assert (
            len(track) == height
        ), "given track height does not match actual track height"
        assert all(
            len(row) == width for row in track
        ), "given track width does not match actual track width"

        def get_coordinates(expected_cell_char: str) -> t.FrozenSet[Coordinate]:
            return frozenset(
                Coordinate(x, y)
                for y, row in enumerate(track)
                for x, cell_char in enumerate(row)
                if cell_char == expected_cell_char
            )

        blank_cells = get_coordinates(".")
        blocked_cells = get_coordinates("x")
        start_cells = get_coordinates("s")
        goal_cells = get_coordinates("g")

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

    start_cell: t.Optional[Coordinate]

    tank_type: TankType = TankType.LARGE
    underground: Underground = Underground.TARMAC

    max_speed: t.Optional[int] = None
    max_acceleration: int = 1

    fuel_model: t.Optional[FuelModel] = fuel_model_regular

    compute_distances: bool = False

    random_start: bool = False

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
        assert self.fuel_model is not None, "no fuel model has been defined"
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

    speed_bound = (
        max(scenario.track.width, scenario.track.height) + scenario.max_acceleration
    )

    ctx.global_scope.declare_variable(
        "car_dx",
        types.INT.bound(-speed_bound, speed_bound),
        initial_value=0,
    )
    ctx.global_scope.declare_variable(
        "car_dy",
        types.INT.bound(-speed_bound, speed_bound),
        initial_value=0,
    )

    ctx.global_scope.declare_variable(
        "car_x",
        types.INT.bound(-1, track.width),
        initial_value=scenario.start_cell.x if scenario.start_cell is not None else 0,
    )
    ctx.global_scope.declare_variable(
        "car_y",
        types.INT.bound(-1, track.height),
        initial_value=scenario.start_cell.y if scenario.start_cell is not None else 0,
    )

    ctx.global_scope.declare_variable(
        "map",
        typ=types.array_of(types.array_of(types.INT.bound(0, 3))),
        is_transient=True,
        initial_value=model.expressions.ArrayValue(
            tuple(
                model.expressions.ArrayValue(
                    tuple(
                        model.ensure_expr(track.get_cell_type(Coordinate(x, y)).number)
                        for x in range(track.width)
                    )
                )
                for y in range(track.height)
            )
        ),
    )

    if scenario.compute_distances:
        ctx.global_scope.declare_variable(
            "goal_dist_x",
            typ=types.INT.bound(-track.width - 1, track.width + 1),
            initial_value=0,
        )
        ctx.global_scope.declare_variable(
            "goal_dist_y",
            typ=types.INT.bound(-track.height - 1, track.height + 1),
            initial_value=0,
        )
        ctx.global_scope.declare_variable(
            "goal_dist",
            typ=types.INT.bound(0, track.width + track.height + 2),
            initial_value=0,
        )
        for direction in Direction:
            ctx.global_scope.declare_variable(
                direction.distance_variable,
                typ=types.INT.bound(
                    0, math.floor(math.sqrt(track.width ** 2 + track.height ** 2)) + 1
                ),
                initial_value=0,
            )

    if scenario.fuel_model is not None:
        ctx.global_scope.declare_variable(
            "fuel",
            types.INT.bound(0, scenario.tank_size),
            initial_value=scenario.tank_size,
        )

    accelerate = ctx.create_action_type("accelerate").create_pattern()
    # The environment is about to move the car.
    move_tick = ctx.create_action_type("move_tick").create_pattern()
    # The environment is about to check the state of the car.
    check_tick = ctx.create_action_type("check_tick").create_pattern()
    # The environment is about to delegate the decision back to the car.
    delegate = ctx.create_action_type("delegate").create_pattern()

    def is_off_track(
        car_x: model.Expression = expr("car_x"),
        car_y: model.Expression = expr("car_y"),
    ):
        return expr(
            "$car_x >= WIDTH or $car_x < 0 or $car_y >= HEIGHT or $car_y < 0",
            car_x=car_x,
            car_y=car_y,
        )

    def is_at_cell(
        typ: CellType,
        car_x: model.Expression = expr("car_x"),
        car_y: model.Expression = expr("car_y"),
    ):
        return model.expressions.ite(
            is_off_track(car_x, car_y),
            model.ensure_expr(typ is CellType.BLOCKED),
            expr(
                f"$cell_number == {typ.number}",
                cell_number=model.expressions.ArrayAccess(
                    model.expressions.ArrayAccess(expr("map"), car_y), car_x
                ),
            ),
        )

    def is_at_goal(
        car_x: model.Expression = expr("car_x"),
        car_y: model.Expression = expr("car_y"),
    ) -> model.Expression:
        return is_at_cell(CellType.GOAL, car_x, car_y)

    def is_at_blocked(
        car_x: model.Expression = expr("car_x"),
        car_y: model.Expression = expr("car_y"),
    ) -> model.Expression:
        return is_at_cell(CellType.BLOCKED, car_x, car_y)

    # In case the fuel is empty before reaching the goal, the model goes
    # into a dead state without transitions. Hence, this property also
    # covers the consumption of fuel.
    ctx.define_property(
        "goalProbability",
        prop("min({ Pmax(F($is_at_goal)) | initial })", is_at_goal=is_at_goal()),
    )

    def construct_car_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="car")
        initial = automaton.create_location(initial=True)

        def compute_speed(
            current: model.Expression, acceleration: expressions.ValueOrExpression
        ) -> model.Expression:
            if scenario.max_speed is None:
                return expr(
                    "$current + $acceleration",
                    current=current,
                    acceleration=acceleration,
                )
            else:
                return expr(
                    "max(min($current + $acceleration, $max_speed), -$max_speed)",
                    current=current,
                    acceleration=acceleration,
                    max_speed=scenario.max_speed,
                )

        for ax, ay in itertools.product(scenario.possible_accelerations, repeat=2):
            automaton.create_edge(
                source=initial,
                destinations=[
                    model.create_destination(
                        location=initial,
                        assignments={
                            "car_dx": compute_speed(expr("car_dx"), ax),
                            "car_dy": compute_speed(expr("car_dy"), ay),
                        },
                        probability=scenario.underground.acceleration_probability,
                    ),
                    model.create_destination(
                        location=initial,
                        assignments={
                            "car_dx": compute_speed(
                                expr("car_dx"),
                                scenario.underground.acceleration_model(ax),
                            ),
                            "car_dy": compute_speed(
                                expr("car_dy"),
                                scenario.underground.acceleration_model(ay),
                            ),
                        },
                        probability=expr(
                            "1 - $p",
                            p=scenario.underground.acceleration_probability,
                        ),
                    ),
                ],
                action_pattern=accelerate,
                annotation={"ax": ax, "ay": ay},
            )

        return automaton

    def construct_tank_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="tank")
        initial = automaton.create_location(initial=True)

        consumption = scenario.compute_consumption(expr("car_dx"), expr("car_dy"))
        automaton.create_edge(
            source=initial,
            destinations=[
                model.create_destination(
                    initial,
                    assignments={
                        "fuel": expr(
                            "fuel - floor($consumption)", consumption=consumption
                        )
                    },
                )
            ],
            action_pattern=check_tick,
            guard=expr(
                "fuel >= $consumption",
                consumption=consumption,
            ),
        )

        return automaton

    def construct_environment_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="environment")

        automaton.scope.declare_variable(
            "start_x", typ=types.INT.bound(-1, track.width), initial_value=0
        )
        automaton.scope.declare_variable(
            "start_y", typ=types.INT.bound(-1, track.height), initial_value=0
        )
        automaton.scope.declare_variable(
            "counter",
            typ=types.INT.bound(0, max(track.width, track.height) + 1),
            initial_value=0,
        )

        initial = automaton.create_location("initial", initial=True)
        position_set = automaton.create_location("position_set")
        wait_for_car = automaton.create_location("wait_for_car")
        move_car = automaton.create_location("move_car")
        env_check = automaton.create_location("env_check")

        move_ticks = expr("max(abs(car_dx), abs(car_dy))")

        if scenario.start_cell is None:
            options = set(track.start_cells)
            if scenario.random_start:
                options.update(track.blank_cells)
            automaton.create_edge(
                initial,
                destinations=[
                    model.create_destination(
                        position_set,
                        assignments={
                            "car_x": model.ensure_expr(start_cell.x),
                            "car_y": model.ensure_expr(start_cell.y),
                        },
                        probability=expr(f"1 / {len(options)}"),
                    )
                    for start_cell in options
                ],
            )
        else:
            automaton.create_edge(
                initial,
                destinations=[
                    model.create_destination(
                        position_set,
                        assignments={
                            "car_x": model.ensure_expr(scenario.start_cell.x),
                            "car_y": model.ensure_expr(scenario.start_cell.x),
                        },
                    )
                ],
            )

        automaton.create_edge(
            source=position_set,
            destinations=[model.create_destination(wait_for_car)],
            action_pattern=delegate,
        )

        # Wait for the decision of the car.
        automaton.create_edge(
            source=wait_for_car,
            destinations=[
                model.create_destination(
                    location=move_car,
                    assignments={
                        "counter": expr("0"),
                        "start_x": expr("car_x"),
                        "start_y": expr("car_y"),
                    },
                )
            ],
            action_pattern=accelerate,
        )

        # Move the car or delegate the decision back to the car.
        automaton.create_edge(
            source=move_car,
            destinations=[
                model.create_destination(
                    env_check,
                    assignments={
                        "counter": expr("counter + 1"),
                        "car_x": expr(
                            "start_x + floor((counter + 1) * (car_dx / $move_ticks) + 0.5)",
                            move_ticks=move_ticks,
                        ),
                        "car_y": expr(
                            "start_y + floor((counter + 1) * (car_dy / $move_ticks) + 0.5)",
                            move_ticks=move_ticks,
                        ),
                    },
                )
            ],
            guard=expr("counter < $move_ticks", move_ticks=move_ticks),
            action_pattern=move_tick,
        )
        automaton.create_edge(
            source=move_car,
            destinations=[model.create_destination(wait_for_car)],
            guard=expr("counter >= $move_ticks", move_ticks=move_ticks),
            action_pattern=delegate,
        )

        # Checker whether we should terminate or continue moving the car.
        should_terminate = expr(
            "$is_off_track or $is_at_goal or $is_at_blocked",
            is_off_track=is_off_track(),
            is_at_goal=is_at_goal(),
            is_at_blocked=is_at_blocked(),
        )
        automaton.create_edge(
            source=env_check,
            destinations=[model.create_destination(move_car)],
            guard=expr("not $should_terminate", should_terminate=should_terminate),
            action_pattern=check_tick,
        )

        return automaton

    def construct_distance_automaton() -> model.Automaton:
        automaton = ctx.create_automaton(name="Distance")

        compute = automaton.create_location("compute")
        done = automaton.create_location("done")
        wait = automaton.create_location("wait", initial=True)

        def get_goal_dist_x(goal: Coordinate) -> model.Expression:
            return expr(f"{goal.x} - car_x")

        def get_goal_dist_y(goal: Coordinate) -> model.Expression:
            return expr(f"{goal.y} - car_y")

        def get_goal_dist(goal: Coordinate) -> model.Expression:
            return expr(
                "abs($dist_x) + abs($dist_y)",
                dist_x=get_goal_dist_x(goal),
                dist_y=get_goal_dist_y(goal),
            )

        automaton.create_edge(
            done,
            destinations=[model.create_destination(wait)],
            action_pattern=accelerate,
        )

        assignments = {
            direction.distance_variable: model.ensure_expr(0) for direction in Direction
        }
        assignments["goal_dist"] = model.ensure_expr(track.width + track.height + 2)
        for goal in track.goal_cells:
            assignments["goal_dist"] = model.expressions.minimum(
                get_goal_dist(goal), assignments["goal_dist"]
            )
        automaton.create_edge(
            wait,
            destinations=[
                model.create_destination(
                    compute,
                    assignments=assignments,
                )
            ],
            action_pattern=delegate,
        )

        def get_current_x(direction: Direction):
            return expr(
                "car_x + $factor * $distance",
                factor=direction.delta.x,
                distance=expr(direction.distance_variable),
            )

        def get_current_y(direction: Direction):
            return expr(
                "car_y + $factor * $distance",
                factor=direction.delta.y,
                distance=expr(direction.distance_variable),
            )

        def is_done(direction: Direction):
            return model.expressions.logic_or(
                is_at_blocked(get_current_x(direction), get_current_y(direction)),
                is_off_track(get_current_x(direction), get_current_y(direction)),
            )

        is_done_all = model.expressions.logic_all(
            *(is_done(direction) for direction in Direction)
        )

        assignments = {"goal_dist_x": track.width + 1, "goal_dist_y": track.height + 1}
        for goal in track.goal_cells:
            do_apply = expr(
                "goal_dist == $to_this_goal", to_this_goal=get_goal_dist(goal)
            )
            assignments["goal_dist_x"] = model.expressions.ite(
                do_apply, get_goal_dist_x(goal), assignments["goal_dist_x"]
            )
            assignments["goal_dist_y"] = model.expressions.ite(
                do_apply, get_goal_dist_y(goal), assignments["goal_dist_y"]
            )
        automaton.create_edge(
            compute,
            destinations=[model.create_destination(done, assignments=assignments)],
            guard=is_done_all,
        )

        automaton.create_edge(
            compute,
            destinations=[
                model.create_destination(
                    compute,
                    assignments={
                        direction.distance_variable: expr(
                            f"{direction.distance_variable} + $delta",
                            delta=model.expressions.ite(is_done(direction), 0, 1),
                        )
                        for direction in Direction
                    },
                )
            ],
            guard=expr("not $is_done", is_done=is_done_all),
        )

        return automaton

    car = construct_car_automaton().create_instance()
    environment = construct_environment_automaton().create_instance()

    accelerate_vector = {car: accelerate, environment: accelerate}
    check_tick_vector = {environment: check_tick}
    delegate_vector = {environment: delegate}

    if scenario.fuel_model:
        tank = construct_tank_automaton().create_instance()
        check_tick_vector[tank] = check_tick

    if scenario.compute_distances:
        instance = construct_distance_automaton().create_instance()
        delegate_vector[instance] = delegate
        accelerate_vector[instance] = accelerate

    network.create_link(accelerate_vector, result=accelerate)
    network.create_link({environment: move_tick}, result=move_tick)
    network.create_link(check_tick_vector, result=check_tick)
    network.create_link(delegate_vector, result=delegate)

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
                            compute_distances=True,
                        )
