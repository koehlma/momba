# Actions

```{warning}
Momba support value passing via actions.
This feature is not part of the official JANI specification.
In case you want your model to work with a broad variety of tools do not add any parameters to your action types.
Action types without any parameters correspond to standard labeled actions as per the JANI specification.
```

```{eval-rst}
.. autoclass:: momba.model.ActionType
    :members:

.. autoclass:: momba.model.ActionParameter
    :members:
```


## Patterns

When synchronizing with other automata, *action patterns* are used.
Again, action patterns without any *arguments* correspond to standard labeled actions as per the JANI specification.

```{eval-rst}
.. autoclass:: momba.model.ActionPattern
    :members:
```


## Arguments

Arguments are used for value passing and are not yet fully documented.

```{eval-rst}
.. autoclass:: momba.model.ActionArgument
    :members:

.. autoclass:: momba.model.WriteArgument
    :members:

.. autoclass:: momba.model.ReadArgument
    :members:

.. autoclass:: momba.model.GuardArgument
    :members:
```