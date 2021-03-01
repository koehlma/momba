# Properties

In case you create expressions using Python code you probably want to use the convenience function {func}`~momba.moml.prop` instead of working with the classes provided here directly.
Note that this function is still provisional, however, it provides a much more concise way of defining expressions.
In any case, instead of constructing objects directly it is recommended to use the constructor functions documented bellow.


## Constructor Functions

```{eval-rst}
.. autofunction:: momba.model.properties.aggregate

.. autofunction:: momba.model.properties.min_prob

.. autofunction:: momba.model.properties.max_prob

.. autofunction:: momba.model.properties.forall_paths

.. autofunction:: momba.model.properties.exists_path

.. autofunction:: momba.model.properties.min_expected_reward

.. autofunction:: momba.model.properties.max_expected_reward

.. autofunction:: momba.model.properties.min_steady_state

.. autofunction:: momba.model.properties.max_steady_state

.. autofunction:: momba.model.properties.until

.. autofunction:: momba.model.properties.weak_until

.. autofunction:: momba.model.properties.release

.. autofunction:: momba.model.properties.eventually

.. autofunction:: momba.model.properties.globally
```


## Class Reference

```{eval-rst}
.. autoclass:: momba.model.properties.Aggregate
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.StatePredicate
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.StateSelector
    :members:
    :member-order: bysource

.. autoattribute:: momba.model.properties.INITIAL_STATES

.. autoattribute:: momba.model.properties.DEADLOCK_STATES

.. autoattribute:: momba.model.properties.TIMELOCK_STATES

.. autoclass:: momba.model.properties.Probability
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.PathQuantifier
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.AccumulationInstant
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.ExpectedReward
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.RewardInstant
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.SteadyState
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.BinaryPathFormula
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.UnaryPathFormula
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.Interval
    :members:
    :member-order: bysource

.. autoclass:: momba.model.properties.RewardBound
    :members:
    :member-order: bysource
```