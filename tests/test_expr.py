from momba import model
from momba.moml import expr


def test_expr_array_idx():
    assert expr("x[3][4]") == model.expressions.ArrayAccess(
        model.expressions.ArrayAccess(
            model.expressions.Name("x"), model.expressions.IntegerConstant(3)
        ),
        model.expressions.IntegerConstant(4),
    )
