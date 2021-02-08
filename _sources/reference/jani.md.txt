# JANI Models

Momba exposes two main functions to work with [JANI-model](https://jani-spec.org) files, {func}`~momba.jani.dump_model` and {func}`~momba.jani.load_model`.
As the names suggest, {func}`~momba.jani.dump_model` exports a Momba automaton {class}`~momba.model.Network` to JANI-model while the function {func}`~momba.jani.load_model` imports an automaton {class}`~momba.model.Network` from a JANI-model file.


## Examples 

### Loading a JANI-Model

```{jupyter-execute}
import pathlib

from momba import jani

path = pathlib.Path("tests/resources/QVBS2019/benchmarks/mdp/firewire/firewire.true.jani")
jani.load_model(path.read_text("utf-8"))
```

### Exporting a JANI-Model

```{jupyter-execute}
from momba import jani, model

ctx = model.Context(model.ModelType.MDP)
network = ctx.create_network()

# ... build the automaton network

jani.dump_model(network)
```


## Reference

```{eval-rst}
.. autofunction:: momba.jani.load_model

.. autofunction:: momba.jani.dump_model

.. autoclass:: momba.jani.ModelFeature
    :members:
```

### Exceptions

```{eval-rst}
.. autoclass:: momba.jani.JANIError

.. autoclass:: momba.jani.InvalidJANIError

.. autoclass:: momba.jani.UnsupportedJANIError
```
