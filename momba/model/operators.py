# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import enum
import fractions
import math


class Operator:
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class BinaryOperator(Operator):
    pass


class NativeBooleanFunction(t.Protocol):
    def __call__(self, left: bool, right: bool) -> bool:
        pass


class BooleanOperator(BinaryOperator, enum.Enum):
    AND = "∧", lambda left, right: left and right
    OR = "∨", lambda left, right: left or right
    XOR = "⊕", lambda left, right: (left or right) and not (left and right)
    IMPLY = "⇒", lambda left, right: not left or right
    EQUIV = "⇔", lambda left, right: left is right

    native_function: NativeBooleanFunction

    def __init__(self, symbol: str, native_function: NativeBooleanFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


Number = t.Union[int, float, fractions.Fraction]


class NativeArithmeticFunction(t.Protocol):
    def __call__(self, left: Number, right: Number) -> Number:
        pass


class ArithmeticOperator(BinaryOperator, enum.Enum):
    ADD = "+", lambda left, right: left + right
    SUB = "-", lambda left, right: left - right
    MUL = "*", lambda left, right: left * right
    MOD = "%", lambda left, right: left % right  # TODO: is this correct?

    MIN = "min", lambda left, right: min(left, right)
    MAX = "max", lambda left, right: max(left, right)

    FLOOR_DIV = "//", lambda left, right: left // right
    REAL_DIV = "/", lambda left, right: left / right

    LOG = "log", lambda left, right: math.log(left, right)
    POW = "pow", lambda left, right: pow(left, right)

    native_function: NativeArithmeticFunction

    def __init__(self, symbol: str, native_function: NativeArithmeticFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


class NativeEqualityFunction(t.Protocol):
    def __call__(self, left: t.Any, right: t.Any) -> bool:
        pass


class EqualityOperator(BinaryOperator, enum.Enum):
    EQ = "=", lambda left, right: left == right
    NEQ = "≠", lambda left, right: left != right

    native_function: NativeEqualityFunction

    def __init__(self, symbol: str, native_function: NativeEqualityFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


class NativeComparisonFunction(t.Protocol):
    def __call__(self, left: t.Any, right: t.Any) -> bool:
        pass


class ComparisonOperator(BinaryOperator, enum.Enum):
    LT = "<", True, lambda left, right: left < right
    LE = "≤", False, lambda left, right: left <= right
    GE = "≥", False, lambda left, right: left >= right
    GT = ">", True, lambda left, right: left > right

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
    pass


class NotOperator(UnaryOperator, enum.Enum):
    NOT = "¬"


class NativeRoundFunction(t.Protocol):
    def __call__(self, operand: Number) -> int:
        pass


class RoundOperator(UnaryOperator, enum.Enum):
    CEIL = "ceil", lambda operand: math.ceil(operand)
    FLOOR = "floor", lambda operand: math.floor(operand)

    native_function: NativeRoundFunction

    def __init__(self, symbol: str, native_function: NativeRoundFunction) -> None:
        super().__init__(symbol)
        self.native_function = native_function


# TODO for MX: review the code for properties Michaela wrote


class Expected(enum.Enum):
    EMAX = "Emax"
    EMIN = "Emin"


class Probability(enum.Enum):
    PMIN = "Pmin"
    PMAX = "Pmax"


class Steady(enum.Enum):
    SMIN = "Smin"
    SMAX = "Smax"


class PathOperator(enum.Enum):
    FORALL = "∀"
    EXISTS = "∃"


class TimeOperator(enum.Enum):
    UNTIL = "U"
    WEAKU = "W"


class FilterFunction(enum.Enum):
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    ARGMIN = "argmin"
    ARGMAX = "argmax"
    EXISTS = "∃"
    FORALL = "∀"
    VALUES = "values"
