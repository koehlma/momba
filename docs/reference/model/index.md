# Momba Models

The package `momba.model` contains the core data structures for the representation of quantitative models.
Momba's internal model representation closely follows the [JANI specification](https://jani-spec.org).
A model is represented as a network of interacting automata.
At the heart of every model is a *modeling context* represented by a {class}`~momba.model.Context` object.
A modeling context specifies a model type (MDP, PTA, et cetera) and contains declarations for global variables.
A modeling context allows creating automata ({class}`~momba.model.Automaton`) of the respective model type as well as composing those automata to networks ({class}`~momba.model.Network`).

```{note}
The data structures are *append only*, i.e., one can define a model incrementally but one cannot change already defined parts of a model.
For instance, it is possible to add a location to an already defined automaton but it is not possible to remove a location from the automaton.
Thereby, the provided API ensures that the model is valid at all times.
```


```{toctree}
:hidden:

context
automata
networks
actions
expressions
properties
types
functions
exceptions
```

