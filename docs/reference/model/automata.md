# Automata

Automata are a core concept of the [JANI modeling formalism](https://jani-spec.org).
An automaton is defined by a set of *locations* connected via *edges*.
Creating an automaton requires a {class}`~momba.model.Context`.
We recommend constructing automata via the method {meth}`~momba.model.Context.create_automaton` of the respective modeling context:

```{jupyter-execute}
from momba import model

# creates a *Probabilistic Timed Automata* (PTA) modeling context
ctx = model.Context(model.ModelType.PTA)
# let's create an automaton named *environment*
environment = ctx.create_automaton(name="environment")
environment
```

```{warning}
Automata in Momba support *parameters*.
This feature, however, is not part of the official JANI specification.
In case you want your model to work with a broad variety of tools do not add any parameters to your automata.
```

```{eval-rst}
.. autoclass:: momba.model.Automaton
    :members:
    :member-order: bysource
```

## Locations

```{eval-rst}
.. autoclass:: momba.model.Location
    :members:
```


## Edges

```{eval-rst}
.. autoclass:: momba.model.Edge
    :members:

.. autoclass:: momba.model.Destination
    :members:

.. autofunction:: momba.model.create_destination

.. autoclass:: momba.model.Assignment
    :members:

.. autofunction:: momba.model.automata.are_compatible
```


## Instances

```{eval-rst}
.. autoclass:: momba.model.Instance
    :members:
```
