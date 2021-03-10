# Quickstart

If you would like to use the model for your own research, here is how you get started.
To use the model from within Python, we first import the necessary packages:

```{jupyter-execute}
import random

from momba import engine

from racetrack import model, tracks, svg
```

The `racetrack` package allows constructing models based on *scenario descriptions*.
In the case of Racetrack, a scenario description specifies the track and other parameters such as the tank size or the underground (see {class}`~racetrack.model.Scenario`).
Here, we use the *Barto-Big* track:

```{jupyter-execute}
tracks.BARTO_BIG
```

We construct a scenario by randomly selecting a start cell of that track:

```{jupyter-execute}
start_cell = random.choice(list(tracks.BARTO_BIG.start_cells))
scenario = model.Scenario(tracks.BARTO_BIG, start_cell, underground=model.Underground.SAND)
scenario
```

Based on the scenario description, we then build the actual model:
```{jupyter-execute}
mdp = model.construct_model(scenario)
mdp
```

As you can see, this gave us an automaton {class}`~momba.model.Network`.

We can now export the network to JANI, run a model checker on it, or explore it with Momba's state exploration engine—just to mention a few possibilities.
To explore the model with Momba's state space exploration engine, we construct an {class}`~momba.engine.Explorer`:

```{jupyter-execute}
explorer = engine.Explorer.new_discrete_time(mdp)
explorer
```

We can now inspect the initial state and visualize it:
```{jupyter-execute}
initial_state, = explorer.initial_states
initial_state.global_env
```

```{jupyter-execute}
svg.format_track(tracks.BARTO_BIG, car=initial_state.global_env["car_pos"].as_int)
```

For further details on what to do with the model, we refer to the [user guide](../../guide/index).
