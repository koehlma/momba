# Model Exploration

In the previous section, we defined a formal model of a simple jump 'n' run game.
Here, we demonstrate how to use Momba's state space exploration engine to explore the state space of this model.

First, we import the model:

```{jupyter-execute}
from momba_guide import model

model.network
```

The package [`momba.engine`](momba_engine) exposes the API for state space exploration.
We import the package and create a discrete time explorer for our MDP automaton network:

```{jupyter-execute}
from momba import engine

explorer = engine.Explorer.new_discrete_time(model.network)
```

The model has one initial state which can be obtained as follows:

```{jupyter-execute}
(state,) = explorer.initial_states

state.global_env
```

The `global_env` attributes contains a mapping from global variable names to the values of these variables.
As defined in the model, the player starts at the position {math}`(0, 0)` on the track.

Having a state, we can ask for the outgoing transitions of this state:

```{jupyter-execute}
for transition in state.transitions:
    assert transition.action is not None
    print(f"Move: {transition.action.action_type.label}")
    for destination in transition.destinations.support:
        print(f"  Probability: {destination.probability}")
        print(f"    Globals: {destination.state.global_env}")
        print(f"    Locals: {destination.state.get_local_env(model.environment)}")
```

Based on this API, it is straightforward to develop a domain specific tool for interactive model exploration or test various aspects of the model.