import dataclasses as d
import typing as t

import functools

from momba import model


@d.dataclass(frozen=True)
class Value:
    value: t.Union[int, float, bool]


@functools.singledispatch
def fold_constants(expr: model.Expression) -> t.Union[model.Expression, Value]:
    # TODO: it would be nice to have a general API to
    # recursively fold over the AST of expressions
    return expr


@fold_constants.register
def _fold_boolean(
    expr: model.expressions.BooleanConstant,
) -> t.Union[model.Expression, Value]:
    return Value(expr.boolean)


@fold_constants.register
def _fold_integer(
    expr: model.expressions.IntegerConstant,
) -> t.Union[model.Expression, Value]:
    return Value(expr.integer)


@fold_constants.register
def _fold_real(
    expr: model.expressions.RealConstant,
) -> t.Union[model.Expression, Value]:
    return Value(expr.integer)


@fold_constants.register
def _fold_comparison(
    expr: model.expressions.Comparison,
) -> t.Union[model.Expression, Value]:
    left = fold_constants(expr.left)
    right = fold_constants(expr.right)
    if isinstance(left, Value) and isinstance(right, Value):
        # TODO: this should throw a proper exception
        assert not isinstance(left.value, bool) and not isinstance(
            right.value, bool
        ), "comparison is not defined for boolean values"
        return Value(expr.operator.native_function(left.value, right.value))
    return model.expressions.Comparison(expr.operator, left, right)


def _shortcircuit_commutative_ops(
    expr: model.expressions.Boolean, value: bool, other: model.Expression
) -> t.Union[model.Expression, Value]:
    if expr.operator is model.operators.BooleanOperator.AND:
        if value:
            return other
        else:
            return Value(False)
    elif expr.operator is model.operators.BooleanOperator.OR:
        if value:
            return Value(True)
        else:
            return other
    else:
        return expr


@fold_constants.register
def _fold_boolean(
    expr: model.expressions.Boolean,
) -> t.Union[model.Expression, Value]:
    left = fold_constants(expr.left)
    right = fold_constants(expr.right)
    if isinstance(left, Value) and isinstance(right, Value):
        # TODO: this should throw a proper exception
        assert isinstance(left.value, bool) and isinstance(
            right.value, bool
        ), "boolean operator is not defined for non-boolean values"
        return Value(expr.operator.native_function(left.value, right.value))
    elif isinstance(left, Value):
        # TODO: this should throw a proper exception
        assert isinstance(
            left.value, bool
        ), "boolean operator is not defined for non-boolean values"
        return _shortcircuit_commutative_ops(expr, left.value, right)
    elif isinstance(right, Value):
        # TODO: this should throw a proper exception
        assert isinstance(
            right.value, bool
        ), "boolean operator is not defined for non-boolean values"
        return _shortcircuit_commutative_ops(expr, right.value, left)
    return model.expressions.Boolean(expr.operator, left, right)


def main():
    from momba.moml import expr

    expr1 = expr("1 < 2 and x")
    expr2 = expr("2 < 4 and x")

    print(fold_constants(expr1))
    print(fold_constants(expr2))

    print(fold_constants(expr1) == fold_constants(expr2))


if __name__ == "__main__":
    main()
