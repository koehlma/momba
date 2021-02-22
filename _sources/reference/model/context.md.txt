# Context

At the heart of every model is a *modeling context* of a specific *model type*.
Every modeling context can hold several automata and automata networks comprised of these automata.

A modeling context is created as follows proving a model type:

```{jupyter-execute}
from momba import model

# creates a *Probabilistic Timed Automata* (PTA) modeling context
ctx = model.Context(model.ModelType.PTA)
ctx
```

```{eval-rst}
.. autoclass:: momba.model.Context
    :members:
    :member-order: bysource
```

The different model types are represented by an enum:

```{eval-rst}
.. autoclass:: momba.model.ModelType
    :members:
```


## Scope

*Variables* and *constants* are declared in a *scope*.
There is one *global scope* associated with each modeling context (see attribute {attr}`~momba.model.Context.global_scope`).
In addition, each automaton has its own *local scope*.
Scopes can be *nested* enabling access to identifies declared in a *parent scope* within a *child scope*.
For instance, the local scopes of automata defined on a modeling context are children of the global scope of the respective context enabling access to both local and global variables and constants.


```{eval-rst}
.. autoclass:: momba.model.Scope
    :members:
    :member-order: bysource
```


## Declarations

There are two kinds of *identifier declarations*, *variable declarations* and *constant declarations*.
Every identifier declaration declares a particular identifier with a specified {class}`~momba.model.Type`.

```{eval-rst}
.. autoclass:: momba.model.IdentifierDeclaration
    :members:

.. autoclass:: momba.model.VariableDeclaration
    :members:

.. autoclass:: momba.model.ConstantDeclaration
    :members:
```


## Properties

Every modeling context may have several *property definitions* attached to it.

```{eval-rst}
.. autoclass:: momba.model.PropertyDefinition
    :members:
```
