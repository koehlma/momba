# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations


class ExplorationError(Exception):
    pass


class EvaluationError(ExplorationError):
    pass


class UnboundIdentifierError(EvaluationError):
    """ An identifier is used but not bound to a value. """


class UnsupportedExpressionError(EvaluationError):
    """ An expression is unsupported and cannot be evaluated. """
