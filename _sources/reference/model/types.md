# Types

[JANI-model](https://jani-spec.org) defines a type system for expressions.


```{eval-rst}
.. autoclass:: momba.model.Type
    :members:
    :member-order: bysource

.. autoattribute:: momba.model.types.INT
.. autoattribute:: momba.model.types.REAL
.. autoattribute:: momba.model.types.BOOL
.. autoattribute:: momba.model.types.CLOCK
.. autoattribute:: momba.model.types.CONTINUOUS

.. autofunction:: momba.model.types.array_of
.. autofunction:: momba.model.types.set_of
```


## Class Hierarchy

```{eval-rst}
.. autoclass:: momba.model.types.NumericType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.IntegerType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.RealType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.BoolType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.ClockType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.ContinuousType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.BoundedType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.ArrayType
    :members:
    :member-order: bysource

.. autoclass:: momba.model.types.SetType
    :members:
    :member-order: bysource
```