# Model Exploration

In the previous section, we defined a formal model of a simple jump 'n' run game.
Here, we demonstrate how to use Momba's state space exploration engine to explore the state space of this model.

For the following examples, we have to import the model defined in [Model Construction](construction):

```{jupyter-execute}
from momba_guide import model
```

The package [`momba.engine`](../reference/engine) exposes the API for state space exploration.
Momba's state space exploration engine supports a variety of different model types including *Probabilistic Timed Automata* (PTAs) and *Markov Decision Processes* (MDPs).
It is written in [Rust](https://www.rust-lang.org).

We import the package and create a *discrete time explorer* for our MDP automaton network:

```{jupyter-execute}
from momba import engine

explorer = engine.Explorer.new_discrete_time(model.network)
```

The model has one initial state which can be obtained as follows:

```{jupyter-execute}
(state,) = explorer.initial_states
```

Remember that we declared two global variables, `pos_x` and `pos_y`, which hold the position of the player on the track.
The values of these variables can be accessed as follows:

```{jupyter-execute}
state.global_env
```

The `global_env` attributes contains a mapping from variable names to the values of these variables.
As defined in the model, the player starts at the position {math}`(0, 0)` on the track.

Having a state, we can also ask for the outgoing transitions of this state:

```{jupyter-execute}
for transition in state.transitions:
    assert transition.action is not None
    print(f"Move: {transition.action.action_type.label}")
    for destination in transition.destinations.support:
        print(f"  Probability: {destination.probability}")
        print(f"    Globals: {destination.state.global_env}")
        print(f"    Locals: {destination.state.get_local_env(model.environment)}")
```

As specified in the model, we can keep our current {math}`y` position or move left or right.
As the model is probabilistic, there are multiple destinations with different probabilities for each transition.
As the player starts on a left-most position on the track, successfully moving left leads to a crash into the wall.
This is also reflected in the successor state where `has_crashed` becomes `True`.

Based on this API, it is straightforward to develop a domain specific tool for interactive model exploration or test various aspects of the model.
For a more elaborate example where we programmed an interactive racing game including some visualizations based on a model using Momba's state space exploration engine, check out the [Racetrack](../examples/racetrack/index) example.
