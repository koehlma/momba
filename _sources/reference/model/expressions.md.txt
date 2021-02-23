# Expressions

In case you create expressions using Python code you probably want to use the convenience function {func}`~momba.moml.expr` instead of working with the classes provided here directly.
Note that this function is still provisional, however, it provides a much more concise way of defining expressions.
In any case, instead of constructing objects directly it is recommended to use the constructor functions documented bellow.

```{eval-rst}
.. autoclass:: momba.model.Expression
    :members:
    :member-order: bysource

.. autofunction:: momba.model.ensure_expr

.. autoclass:: momba.model.expressions.ConversionError
```


## Constructor Functions

```{eval-rst}
.. autofunction:: momba.model.expressions.ite

.. autofunction:: momba.model.expressions.logic_not

.. autofunction:: momba.model.expressions.logic_any

.. autofunction:: momba.model.expressions.logic_or

.. autofunction:: momba.model.expressions.logic_all

.. autofunction:: momba.model.expressions.logic_and

.. autofunction:: momba.model.expressions.logic_xor

.. autofunction:: momba.model.expressions.logic_implies

.. autofunction:: momba.model.expressions.logic_equiv

.. autofunction:: momba.model.expressions.equals

.. autofunction:: momba.model.expressions.not_equals

.. autofunction:: momba.model.expressions.less

.. autofunction:: momba.model.expressions.less_or_equal

.. autofunction:: momba.model.expressions.greater

.. autofunction:: momba.model.expressions.greater_or_equal

.. autofunction:: momba.model.expressions.add

.. autofunction:: momba.model.expressions.sub

.. autofunction:: momba.model.expressions.mul

.. autofunction:: momba.model.expressions.mod

.. autofunction:: momba.model.expressions.real_div

.. autofunction:: momba.model.expressions.log

.. autofunction:: momba.model.expressions.power

.. autofunction:: momba.model.expressions.minimum

.. autofunction:: momba.model.expressions.maximum

.. autofunction:: momba.model.expressions.floor_div

.. autofunction:: momba.model.expressions.floor

.. autofunction:: momba.model.expressions.ceil

.. autofunction:: momba.model.expressions.absolute

.. autofunction:: momba.model.expressions.sgn

.. autofunction:: momba.model.expressions.trunc

.. autofunction:: momba.model.expressions.name
```


## Class Hierarchy

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

.. autoclass:: momba.model.expressions.ArithmeticBinary
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Equality
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Comparison
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Conditional
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.UnaryExpression
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.ArithmeticUnary
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Not
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Sample
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Selection
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Derivative
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.ArrayAccess
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.ArrayValue
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.ArrayConstructor
    :members:
    :member-order: bysource

.. autoclass:: momba.model.expressions.Trigonometric
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

.. autoclass:: momba.model.operators.NotOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.ArithmeticUnaryOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.MinMax
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.Quantifier
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.BinaryPathOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.UnaryPathOperator
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.AggregationFunction
    :members:
    :member-order: bysource

.. autoclass:: momba.model.operators.TrigonometricFunction
    :members:
    :member-order: bysource
```

## Distributions

```{eval-rst}
.. autoclass:: momba.model.distributions.DistributionType
    :members:
    :member-order: bysource
```
