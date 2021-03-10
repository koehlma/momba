# Racetrack
(example_racetrack)=

An example showcasing almost all nifty features of Momba is centered around [Racetrack](https://racetrack.perspicuous-computing.science/).
Originally, Racetrack has been a pen and paper game where a car has to be steered on a two-dimensional grid from a start position to a goal position.
We developed a formal model of this game using Momba.
This page documents how you may use this model for your own research.
It also serves as a paradigmatic example how to leverage Python's vast ecosystem for documentation and model exploration.
This documentation has been generated using [Sphinx](https://www.sphinx-doc.org/en/master/) with embedded [Jupyter Notebook](https://jupyter.org/) cells.
Check out the [Sphinx source](https://github.com/koehlma/momba/tree/master/docs/examples/racetrack) of this page for further details.


## Installation

The Racetrack model is not bundled with Momba but can easily be installed as follows:

```bash
pip install racetrack
```

The full source code of this package is available [here](https://github.com/koehlma/momba/tree/master/examples/racetrack).

The [`racetrack`](https://pypi.org/project/racetrack) package also comes with an [interactive game mode](game) where you can steer the car through the track and thereby explore the behavior of the formal model.


```{toctree}
:maxdepth: 2
:hidden:

quickstart
game
model
```