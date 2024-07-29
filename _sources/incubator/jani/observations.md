# Observations

This document specifies the JANI extension `x-observations` extending the *JANI model interchange format* with support for *observations*.
When enabled, edges can be annotated with a field `observation` containing a non-empty array inducing a probability distribution over actions for each transition.
When taken, any of the actions will be *observed* with the specified probability.

```js
{
    [...],
    "?observation": Array.of({
        "label": Identifier,
        "?arguments": Array.of(Expression),
        "?probability": Expression
    })
}
```

```{note}
Add support for observations when entering or leaving a state.
```