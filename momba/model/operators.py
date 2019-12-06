# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import enum


class BooleanOperator(enum.Enum):
    AND = '∧'
    OR = '∨'
    XOR = '⊕'
    IMPLY = '⇒'
    EQUIV = '⇔'


class ArithmeticOperator(enum.Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    MOD = '%'


class EqualityOperator(enum.Enum):
    EQ = '='
    NEQ = '≠'


class ComparisonOperator(enum.Enum):
    LT = '<'
    LE = '≤'
    GE = '≥'
    GT = '>'
