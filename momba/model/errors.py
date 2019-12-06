# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations


class ModelingError(Exception):
    pass


class NotAConstantError(ModelingError):
    pass


class NotAVariableError(ModelingError):
    pass


class InvalidDeclarationError(ModelingError):
    pass


class UnboundIdentifierError(ModelingError):
    pass


class InvalidTypeError(ModelingError):
    pass


class IncompatibleAssignmentsError(ModelingError):
    pass


class InvalidOperationError(ModelingError):
    pass
