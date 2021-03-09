# Exploration Engine
(momba_engine)=

Momba comes with its own state space exploration engine.
To use this engine, you have to install the optional dependency [`momba_engine`](https://pypi.org/project/momba_engine/) (or install Momba with the `all` feature flag).
The API of the package `momba_engine` itself is private.
Here, we document the public API exposed as part of Momba.


## Example

First, an {class}`~momba.engine.Explorer` has to be created for the network:

```{jupyter-execute}
import pathlib

from momba import engine, jani

# let's load a model from the QVBS
path = pathlib.Path("tests/resources/QVBS2019/benchmarks/mdp/firewire/firewire.true.jani")
network = jani.load_model(path.read_text("utf-8"))

# create a discrete time explorer for the loaded network
explorer = engine.Explorer.new_discrete_time(
    network,
    parameters={
        "delay": 3,
        "deadline": 200,
    }
)
explorer
```

We can then use the created explorer to explore the state space of the model.

Let us start by querying the initial states of the model:

```{jupyter-execute}
initial_states = explorer.initial_states
len(initial_states)
```

There is just one initial state, let's have a closer look at it:

```{jupyter-execute}
(initial_state,) = initial_states
```

We can easily inspect the global environment

```{jupyter-execute}
initial_state.global_env
```

the locations the different automata instances are in

```{jupyter-execute}
for instance, location in initial_state.locations.items():
    print(f"{instance.automaton.name}: {location.name}")
```

and also the local environment of each instance (in this case there are no local variables):
```{jupyter-execute}
for instance in network.instances:
    print(instance.automaton.name)
    print(initial_state.get_local_env(instance))
```

So, let's explore the successors of the initial state `initial_state`.
To this end, we query the outgoing transitions of the initial state and their respective destinations:

```{jupyter-execute}
for transition in initial_state.transitions:
    if transition.action is None:
        # the action of the transition is internal
        print("Action: τ")
    else:
        print(f"Action: {transition.action.action_type.label}")
    for destination in transition.destinations.support:
        print(f"  With p={destination.probability} to {destination}.")
```

Let's chose a transition uniformly and then pick a successor state at random:
```{jupyter-execute}
import random

transition = random.choice(initial_state.transitions)
print(f"Action: {transition.action.action_type.label}")

successor = transition.destinations.pick().state

successor.global_env
```


## Reference

```{eval-rst}
.. autoclass:: momba.engine.Value
    :members:

.. autoclass:: momba.engine.Explorer
    :members:

.. autoclass:: momba.engine.State
    :members:

.. autoclass:: momba.engine.Transition
    :members:

.. autoclass:: momba.engine.Action
    :members:

.. autoclass:: momba.engine.Destination
    :members:
```


### Time Representations

The state space exploration engines supports different time representations.
For discrete time models (MDP, DTMC, and LTS) {class}`~momba.engine.DiscreteTime` should be used.
Creating an explorer with {meth}`~momba.engine.Explorer.new_discrete_time` will use {class}`~momba.engine.DiscreteTime`.
For continuous-time models (TA and PTA) different time representations are available.

```{eval-rst}
.. autoclass:: momba.engine.TimeType
    :members:

.. autoclass:: momba.engine.DiscreteTime
    :members:
```
