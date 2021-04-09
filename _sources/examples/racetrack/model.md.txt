# The Model

```{warning}
This part of the documentation is still incomplete.
```

This part of the documentation demonstrates how to utilize [Sphinx](https://www.sphinx-doc.org/en/master/) with embedded [Jupyter Notebook](https://jupyter.org/) cells to document a model.
To this end, we first import the model:

```{jupyter-execute}
from racetrack import model, tracks
```


## Tracks

Let's have a look at the *Barto-Big* track:
```{jupyter-execute}
tracks.BARTO_BIG
```

When used is a Jupyter notebook, a track is rendered as an SVG.
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

.. autoclass:: racetrack.model.Coordinate
    :members:

.. autoclass:: racetrack.model.CellType
    :members:
```







### Scenarios


```{eval-rst}
.. autoclass:: racetrack.model.Scenario
    :members:
```

```{eval-rst}
.. autoclass:: racetrack.model.Underground
    :members:
    :member-order: bysource
```

```{eval-rst}
.. autoclass:: racetrack.model.TankType
    :members:
    :member-order: bysource
```