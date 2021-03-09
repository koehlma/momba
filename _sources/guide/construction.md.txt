# Model Construction
(model_construction)=

Momba provides *append-only* APIs for incremental model construction, i.e., one can define a model incrementally but one cannot change already defined parts of a model.
For instance, it is possible to add a location to an already defined automaton but it is not possible to remove a location from the automaton.
Thereby, the provided APIs ensure that the model is *valid* at all times.

Here, we construct a model of a simple [jump 'n' run](https://en.wikipedia.org/wiki/Platform_game) game where the player has to avoid obstacles while moving forward at a constant speed.
To avoid obstacles the player can move left or right.
The goal is to reach the end of the *track* without crashing into an obstacle.

With the help of a little ASCII art, a track may be represented as follows:

```{jupyter-execute}
TRACK = (
    "       x         xxxxx     x      ",
    "           xxx          x      xx ",
    " xxxxxxx          xx              ",
)
```

Every `x` represents an obstacle and the player starts on the left and moves forward to the right.
Now, given such a track, we would like to create a model of the game.
Note that this goes beyond what is possible with mere parametrization of a model.

First, we create a more formal representation of the scenario to be modeled.
To this end, we define a class `Track` holding all the information of the track:

```{jupyter-execute}
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
    def from_ascii(cls, ascii: t.Tuple[str, ...]) -> "Map":
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
```

With this in place, we are now ready to construct the model.


## Modeling with Momba

The package [`momba.model`](momba_models) provides the core modeling API for model construction.
At the heart of every model is a *modeling context* ({class}`~momba.model.Context`) of a specific *model type* ({class}`~momba.model.ModelType`).
Every modeling context can hold several automata and automata networks comprised of these automata.
Also, global variables, constants, functions, and properties are declared on the modeling context.
To construct a model, we first have to create a modeling context of the respective type:

```{jupyter-execute}
from momba import model

# creates a *Markov Decision Process* (MDP) modeling context
ctx = model.Context(model.ModelType.MDP)
ctx
```

In our case, the model will be a *Markov Decision Process* (MDP).

To represent the position of the car, we define two global variables, `pos_x` and `pos_y`:

```{jupyter-execute}
ctx.global_scope.declare_variable("pos_x", typ=model.types.INT, initial_value=0)
ctx.global_scope.declare_variable("pos_y", typ=model.types.INT, initial_value=0)
```

To control the game, we create three *action types*:

```{jupyter-execute}
left_action = ctx.create_action_type("left")
right_action = ctx.create_action_type("right")
stay_action = ctx.create_action_type("stay")
```

Each of these actions corresponds to a distance moved on the {math}`y`-axis:

```{jupyter-execute}
moves = {left_action: -1, right_action: 1, stay_action: 0}
```

Next, we define an automaton modeling the *environment*:

```{jupyter-execute}
environment_automaton = ctx.create_automaton(name="Environment")
```

This automaton will have one location:

```{jupyter-execute}
ready_location = environment_automaton.create_location("ready", initial=True)
```

To keep track of whether the player reached the goal or crashed into an obstacle or a wall, we declare two local variables `is_finished` and `has_crashed` for the environment automaton:

```{jupyter-execute}
environment_automaton.scope.declare_variable(
    "is_finished", typ=model.types.BOOL, initial_value=False
)
environment_automaton.scope.declare_variable(
    "has_crashed", typ=model.types.BOOL, initial_value=False
)
```

With the help of Momba's [*syntax-aware macros*](moml_macros) we now define two functions computing whether the player *has crashed* or *has finished* depending on the position:

```{jupyter-execute}
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
```

Momba makes it easy to define functions operating on [JANI-model](https://jani-spec.org) expressions.
In particular, the function {func}`~momba.moml.expr` can be used to build expressions with a succinct and intuitive syntax.
We use the same approach to define an expression indicating whether a move can be performed:

```{jupyter-execute}
can_move = expr("not is_finished and not has_crashed")
```

Now, with these definitions in place, we create the edges of the automaton:

```{jupyter-execute}
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
                probability=expr("1 - 0.4"),
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
```

For each of the previously defined action types, an edge is created.
In this case, the model introduces probabilistic noise and a move might fail, i.e., not change the {math}`y` coordinate, with a probability of {math}`0.4`.
When taking an edge, the variables get updated accordingly.


Finally, we create an *instance* of the automaton:

```{jupyter-execute}
environment = environment_automaton.create_instance()
```

The instance is then added to an automaton network and a *synchronization link* for each action type is created which allows the resulting composition to perform the respective action:

```{jupyter-execute}
network = ctx.create_network()
network.add_instance(environment)
for action_type in moves.keys():
    network.create_link(
        {environment: action_type.create_pattern()},
        result=action_type.create_pattern(),
    ) 
```

You can find the full source code of this example [here](https://github.com/koehlma/momba/blob/master/examples/guide/momba_guide/model.py).

