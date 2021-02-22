# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import enum
import fractions
import math

from . import types


class Operator:
    """
    The base class of all operators.

    Attributes
    ----------
    symbol:
        Symbol associated with the operator.
    """

    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class BinaryOperator(Operator):
    """
    Base class for all binary operators.
    """


class NativeBooleanFunction(t.Protocol):
    def __call__(self, left: bool, right: bool) -> bool:
        pass


class BooleanOperator(BinaryOperator, enum.Enum):
    """
    An enum of boolean operators.
    """

    AND = "∧", lambda left, right: left and right
    """ Logical conjunction. """

    OR = "∨", lambda left, right: left or right
    """ Logical disjunction. """

    # requires JANI extension `derived-operators`
    IMPLY = "⇒", lambda left, right: not left or right
    """ Logical implication. """

    # requires JANI extension `x-momba-operators`
    XOR = "⊕", lambda left, right: (left or right) and not (left and right)
    """ Logical exclusive disjunction. """

    # requires JANI extension `x-momba-operators`
    EQUIV = "⇔", lambda left, right: left is right
    """ Logical equivalence. """

    native_function: NativeBooleanFunction

    def __init__(self, symbol: str, native_function: NativeBooleanFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


Number = t.Union[int, float, fractions.Fraction]


class NativeBinaryArithmeticFunction(t.Protocol):
    def __call__(self, left: Number, right: Number) -> Number:
        pass


class ArithmeticBinaryOperator(BinaryOperator, enum.Enum):
    """
    An enum of arithmetic binary operators.
    """

    ADD = "+", lambda left, right: left + right
    """ Addition. """

    SUB = "-", lambda left, right: left - right
    """ Substraction. """

    MUL = "*", lambda left, right: left * right
    """ Multiplication. """

    MOD = "%", lambda left, right: left % right
    """ Euclidean remainder. """

    REAL_DIV = (
        "/",
        lambda left, right: fractions.Fraction(left) / fractions.Fraction(right),
    )
    """ Real devision. """

    LOG = "log", lambda left, right: math.log(left, right)
    """ Logarithm. """

    POW = "pow", lambda left, right: pow(left, right)
    """ Power. """

    # requires JANI extension `derived-operators`
    MIN = "min", lambda left, right: min(left, right)
    """ Minimum. """

    # requires JANI extension `derived-operators`
    MAX = "max", lambda left, right: max(left, right)
    """ Maximum. """

    # requires JANI extension `x-momba-operators`
    FLOOR_DIV = "//", lambda left, right: left // right
    """ Euclidean division. """

    native_function: NativeBinaryArithmeticFunction

    def __init__(
        self, symbol: str, native_function: NativeBinaryArithmeticFunction
    ) -> None:
        super().__init__(symbol)
        self.native_function = native_function


class NativeEqualityFunction(t.Protocol):
    def __call__(self, left: t.Any, right: t.Any) -> bool:
        pass


class EqualityOperator(BinaryOperator, enum.Enum):
    """
    An enum of equality operators.
    """

    EQ = "=", lambda left, right: left == right
    """ Is equal. """

    NEQ = "≠", lambda left, right: left != right
    """ Is not equal. """

    native_function: NativeEqualityFunction

    def __init__(self, symbol: str, native_function: NativeEqualityFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


class NativeComparisonFunction(t.Protocol):
    def __call__(self, left: t.Any, right: t.Any) -> bool:
        pass


class ComparisonOperator(BinaryOperator, enum.Enum):
    """
    An enum of comparison operators.
    """

    LT = "<", True, lambda left, right: left < right
    """ Is less than. """

    LE = "≤", False, lambda left, right: left <= right
    """ Is less than or equal to. """

    # requires JANI extension `derived-operators`
    GE = "≥", False, lambda left, right: left >= right
    """ Is greater than or equal to. """

    # requires JANI extension `derived-operators`
    GT = ">", True, lambda left, right: left > right
    """ Is greater than. """

    is_strict: bool
    native_function: NativeComparisonFunction

    def __init__(
        self, symbol: str, is_strict: bool, native_function: NativeComparisonFunction
    ) -> None:
        super().__init__(symbol)
        self.is_strict = is_strict
        self.native_function = native_function

    def swap(self) -> ComparisonOperator:
        return _COMPARISON_SWAP_TABLE[self]

    @property
    def is_less(self) -> bool:
        return self in {ComparisonOperator.LT, ComparisonOperator.LE}

    @property
    def is_greater(self) -> bool:
        return self in {ComparisonOperator.GE, ComparisonOperator.GT}


_COMPARISON_SWAP_TABLE = {
    ComparisonOperator.LT: ComparisonOperator.GT,
    ComparisonOperator.LE: ComparisonOperator.GE,
    ComparisonOperator.GE: ComparisonOperator.LE,
    ComparisonOperator.GT: ComparisonOperator.LT,
}


class UnaryOperator(Operator):
    """
    Base class for all unary operators.
    """


class NotOperator(UnaryOperator, enum.Enum):
    NOT = "¬"


class NativeUnaryArithmeticFunction(t.Protocol):
    def __call__(self, operand: Number) -> int:
        pass


class ArithmeticUnaryOperator(UnaryOperator, enum.Enum):
    CEIL = "ceil", lambda _: types.INT, lambda operand: math.ceil(operand)
    FLOOR = "floor", lambda _: types.INT, lambda operand: math.floor(operand)

    # requires JANI extension `derived-operators`
    ABS = "abs", lambda typ: typ, lambda operand: abs(operand)
    SGN = (
        "sgn",
        lambda _: types.INT,
        lambda operand: -1 if operand < 0 else (1 if operand > 0 else 0),
    )
    TRC = "trc", lambda _: types.INT, lambda operand: math.trunc(operand)

    infer_result_type: types.InferTypeUnary
    native_function: NativeUnaryArithmeticFunction

    def __init__(
        self,
        symbol: str,
        infer_result_type: types.InferTypeUnary,
        native_function: NativeUnaryArithmeticFunction,
    ) -> None:
        super().__init__(symbol)
        self.infer_result_type = infer_result_type
        self.native_function = native_function


class MinMax(enum.Enum):
    MIN = "min"
    MAX = "max"


class Quantifier(enum.Enum):
    FORALL = "∀"
    EXISTS = "∃"


class BinaryPathOperator(enum.Enum):
    UNTIL = "U"
    WEAK_UNTIL = "W"

    # requires JANI extension `derived-operators`
    RELEASE = "R"


class UnaryPathOperator(enum.Enum):
    # requires JANI extension `derived-operators`
    EVENTUALLY = "F"
    GLOBALLY = "G"


class AggregationFunction(enum.Enum):
    MIN = "min", {types.REAL}, lambda _: types.REAL
    MAX = "max", {types.REAL}, lambda _: types.REAL
    SUM = "sum", {types.REAL}, lambda _: types.REAL
    AVG = "avg", {types.REAL}, lambda _: types.REAL
    COUNT = "count", {types.BOOL}, lambda _: types.INT
    EXISTS = "∃", {types.BOOL}, lambda _: types.BOOL
    FORALL = "∀", {types.BOOL}, lambda _: types.BOOL
    ARGMIN = "argmin", {types.REAL}, lambda _: types.set_of(types.STATE)
    ARGMAX = "argmax", {types.REAL}, lambda _: types.set_of(types.STATE)
    VALUES = "values", {types.REAL, types.BOOL}, lambda typ: types.set_of(typ)

    symbol: str

    allowed_values_type: t.Set[types.Type]
    infer_result_type: types.InferTypeUnary

    def __init__(
        self,
        symbol: str,
        allowed_values_type: t.Set[types.Type],
        infer_result_type: types.InferTypeUnary,
    ) -> None:
        self.symbol = symbol
        self.allowed_values_type = allowed_values_type
        self.infer_result_type = infer_result_type


class TrigonometricFunction(UnaryOperator, enum.Enum):
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    COT = "cot"
    SEC = "sec"
    CSC = "csc"

    ARC_SIN = "asin"
    ARC_COS = "acos"
    ARC_TAN = "atan"
    ARC_COT = "acot"
    ARC_SEC = "asec"
    ARC_CSC = "acsc"
