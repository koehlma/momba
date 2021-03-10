# Interactive Game

The `racetrack` packages installs a command line tool:

```plain
Usage: racetrack [OPTIONS] COMMAND [ARGS]...

  A formal model of the pen-and-paper game *Racetrack*.

Options:
  --help  Show this message and exit.

Commands:
  generate  Generates a family of JANI models from the provided track file.
  race      Runs an interactive simulation where you can steer the car.
```

Running `racetrack race` with a track file allows exploring the model interactively.
After choosing a start cell, the user can enter an acceleration for the {math}`x` and {math}`y` axes after each step:

```{figure} game.png
:width: 85%
:alt: Screenshot of the interactive simulation.

Screenshot of the interactive simulation.
```

This allows to drive the car and explore the model interactively.
