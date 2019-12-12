# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import enum


class BinaryOperator:
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class Boolean(BinaryOperator, enum.Enum):
    AND = '∧'
    OR = '∨'
    XOR = '⊕'
    IMPLY = '⇒'
    EQUIV = '⇔'


class ArithmeticOperator(BinaryOperator, enum.Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    MOD = '%'

    MIN = 'min'
    MAX = 'max'

    FLOOR_DIV = '//'
    REAL_DIV = '/'

    LOG = 'log'
    POW = 'pow'


class EqualityOperator(BinaryOperator, enum.Enum):
    EQ = '='
    NEQ = '≠'


class Comparison(BinaryOperator, enum.Enum):
    LT = '<'
    LE = '≤'
    GE = '≥'
    GT = '>'


class UnaryOperator:
    symbol: str

    def __init__(self, symbol: str):
        self.symbol = symbol


class Not(UnaryOperator, enum.Enum):
    NOT = '¬'


class Round(UnaryOperator, enum.Enum):
    CEIL = 'ceil'
    FLOOR = 'floor'
