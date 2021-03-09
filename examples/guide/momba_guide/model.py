# flake8: noqa


TRACK = (
    "       x         xxxxx     x      ",
    "           xxx          x      xx ",
    " xxxxxxx          xx              ",
)


import dataclasses as d
import typing as t


@d.dataclass(frozen=True)
class Cell:
    x: int
    y: int


@d.dataclass(frozen=True)
class Track:
    width: int
    height: int
    obstacles: t.Set[Cell]

    @classmethod
    def from_ascii(cls, ascii: t.Tuple[str, ...]) -> "Track":
        width = len(ascii[0])
        height = len(ascii)
        obstacles = set()
        for y, line in enumerate(ascii):
            for x, symbol in enumerate(line):
                if symbol == "x":
                    obstacles.add(Cell(x, y))
        return cls(width, height, obstacles)


track = Track.from_ascii(TRACK)
track


from momba import model

# creates a *Markov Decision Process* (MDP) modeling context
ctx = model.Context(model.ModelType.MDP)
ctx


ctx.global_scope.declare_variable("pos_x", typ=model.types.INT, initial_value=0)
ctx.global_scope.declare_variable("pos_y", typ=model.types.INT, initial_value=0)


left_action = ctx.create_action_type("left")
right_action = ctx.create_action_type("right")
stay_action = ctx.create_action_type("stay")


moves = {left_action: -1, right_action: 1, stay_action: 0}


environment_automaton = ctx.create_automaton(name="Environment")


ready_location = environment_automaton.create_location("ready", initial=True)


environment_automaton.scope.declare_variable(
    "is_finished", typ=model.types.BOOL, initial_value=False
)
environment_automaton.scope.declare_variable(
    "has_crashed", typ=model.types.BOOL, initial_value=False
)


from momba.moml import expr


def has_finished(x: model.Expression, track: Track) -> model.Expression:
    return expr("$x >= $width", x=x, width=track.width)


def has_crashed(
    x: model.Expression, y: model.Expression, track: Track
) -> model.Expression:
    out_of_bounds = expr("$y >= $height or $y < 0", y=y, height=track.height)
    on_obstacle = model.expressions.logic_any(
        *(
            expr(
                "$x == $obstacle_x and $y == $obstacle_y",
                x=x,
                y=y,
                obstacle_x=obstacle.x,
                obstacle_y=obstacle.y,
            )
            for obstacle in track.obstacles
        )
    )
    return model.expressions.logic_or(out_of_bounds, on_obstacle)


can_move = expr("not is_finished and not has_crashed")


for action_type, delta in moves.items():
    new_pos_x = expr("pos_x + 1")
    new_pos_y = expr("pos_y + $delta", delta=delta)

    environment_automaton.create_edge(
        ready_location,
        destinations={
            model.create_destination(
                ready_location,
                probability=expr("0.6"),
                assignments={
                    "pos_x": new_pos_x,
                    "pos_y": new_pos_y,
                    "is_finished": has_finished(new_pos_x, track),
                    "has_crashed": has_crashed(new_pos_x, new_pos_y, track),
                },
            ),
            model.create_destination(
                ready_location,
                probability=expr("1 - 0.6"),
                assignments={
                    "pos_x": new_pos_x,
                    "is_finished": has_finished(new_pos_x, track),
                    "has_crashed": has_crashed(new_pos_x, expr("pos_y"), track),
                },
            ),
        },
        action_pattern=action_type.create_pattern(),
        guard=can_move,
    )


environment = environment_automaton.create_instance()


network = ctx.create_network()
network.add_instance(environment)
for action_type in moves.keys():
    network.create_link(
        {environment: action_type.create_pattern()},
        result=action_type.create_pattern(),
    )
