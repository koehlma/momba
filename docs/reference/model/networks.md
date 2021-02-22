# Networks

Automata can be composed to an *automaton network*.
We recommend constructing networks via the method {meth}`~momba.model.Context.create_network` of the respective modeling context:

```{jupyter-execute}
from momba import model

# creates a *Probabilistic Timed Automata* (PTA) modeling context
ctx = model.Context(model.ModelType.PTA)
# let's create an automaton network named *network*
network = ctx.create_network(name="network")
network
```

A network comprises a set of automaton instances connected via *links*.

```{eval-rst}
.. autoclass:: momba.model.Network
    :members:
    :member-order: bysource

.. autoclass:: momba.model.Link
    :members:
    :member-order: bysource
```
