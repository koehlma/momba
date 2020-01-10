# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import enum


class BinaryOperator:
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class BooleanOperator(BinaryOperator, enum.Enum):
    AND = "∧"
    OR = "∨"
    XOR = "⊕"
    IMPLY = "⇒"
    EQUIV = "⇔"


class ArithmeticOperator(BinaryOperator, enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    MOD = "%"

    MIN = "min"
    MAX = "max"

    FLOOR_DIV = "//"
    REAL_DIV = "/"

    LOG = "log"
    POW = "pow"


class EqualityOperator(BinaryOperator, enum.Enum):
    EQ = "="
    NEQ = "≠"


class ComparisonOperator(BinaryOperator, enum.Enum):
    LT = "<"
    LE = "≤"
    GE = "≥"
    GT = ">"

    def swap(self) -> ComparisonOperator:
        return _COMPARISON_SWAP_TABLE[self]


_COMPARISON_SWAP_TABLE = {
    ComparisonOperator.LT: ComparisonOperator.GT,
    ComparisonOperator.LE: ComparisonOperator.GE,
    ComparisonOperator.GE: ComparisonOperator.LE,
    ComparisonOperator.GT: ComparisonOperator.LT,
}


class UnaryOperator:
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class Not(UnaryOperator, enum.Enum):
    NOT = "¬"


class Round(UnaryOperator, enum.Enum):
    CEIL = "ceil"
    FLOOR = "floor"


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
