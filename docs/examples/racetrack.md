# Racetrack

An example showcasing almost all nifty features of Momba is centered around [Racetrack](https://racetrack.perspicuous-computing.science/).
Originally, Racetrack has been a pen and paper game where a car has to be steered on a two-dimensional grid from a start position to a goal position.
We developed a formal model of this game using Momba.
This page documents how you may use this model for your own research.
It also serves as a paradigmatic example how to leverage Python's vast ecosystem for documentation and model exploration.
This documentation has been generated with [Sphinx](https://www.sphinx-doc.org/en/master/) using [Jupyter](https://jupyter.org/).


## Installation

The Racetrack model is not bundled with Momba but can easily be installed as follows:

```bash
pip install racetrack
```

Check out the source of the Racetrack model [here](https://github.com/koehlma/momba/tree/master/examples/racetrack).


## Quickstart

First, we import the necessary packages:

```{jupyter-execute}
import random

from momba import engine

from racetrack import model, tracks, svg
```

Let's select a start cell from the Barto-Big track and construct a {class}`~racetrack.model.Scenario`:
```{jupyter-execute}
start_cell = random.choice(list(tracks.BARTO_BIG.start_cells))
scenario = model.Scenario(tracks.BARTO_BIG, start_cell)
scenario
```

Based on the scenario description, we then build the actual model:
```{jupyter-execute}
mdp = model.construct_model(scenario)
mdp
```

As you can see, this gave us an automaton {class}`~momba.model.Network`.

Next, we construct an {class}`~momba.engine.Explorer` for model exploration:
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



## The Model

You can easily use the model from within a [Jupyter](https://jupyter.org/) notebook:

```{jupyter-execute}
from racetrack import model, tracks
```


### Tracks

Let's have a look at the *Barto-Big* track:
```{jupyter-execute}
tracks.BARTO_BIG
```

When used is a Jupyter notebook, a track gets rendered in a human-readable form.
The start cells are colored blue, goal cells green, and blocked cells red.
Internally, a track is represented by its width, height, and a sets for each type of cell.
The *cells* of a track are enumerated from top-left to bottom-right.

```{jupyter-execute}
print(repr(tracks.BARTO_BIG))
```

Let's also include some auto-generated documentation for the {class}`~racetrack.model.Track` class.


```{eval-rst}
.. autoclass:: racetrack.model.Track
    :members:
```


```{eval-rst}
.. autoclass:: racetrack.model.Scenario
    :members:
```



### Features

```{eval-rst}
.. autoclass:: racetrack.model.Underground
    :members:
```
