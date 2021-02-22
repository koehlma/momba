# Expressions

In case you create expressions using Python code you probably want to use the convenience function {func}`~momba.moml.expr` instead of working with the classes provided here directly.
Note that this function is still provisional, however, it provides a much more concise way of defining expressions.

```{eval-rst}
.. autoclass:: momba.model.Expression
    :members:
    :member-order: bysource

.. autofunction:: momba.model.ensure_expr
```


## Reference

```{eval-rst}
.. autoclass:: momba.model.expressions.BooleanConstant
    :members:
    :member-order: bysource

.. autoattribute:: momba.model.expressions.TRUE

.. autoattribute:: momba.model.expressions.FALSE

.. autoclass:: momba.model.expressions.NumericConstant
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.IntegerConstant
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.NamedReal
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.RealConstant
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Name
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.BinaryExpression
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Boolean
    :members:
    :member-order: bysource
```


## Operators

```{eval-rst}
.. autoclass:: momba.model.operators.Operator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.BinaryOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.BooleanOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.ArithmeticBinaryOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.EqualityOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.ComparisonOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.UnaryOperator
    :members:
    :member-order: bysource
```