# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

from .. import errors


class ModelingError(errors.MombaError):
    pass


class InvalidTypeError(ModelingError):
    pass


class NotAConstantError(ModelingError):
    pass


class NotAVariableError(ModelingError):
    pass


class InvalidOperationError(ModelingError):
    pass


class IncompatibleAssignmentsError(ModelingError):
    pass


class TypeConstructionError(ModelingError):
    pass


class InvalidDeclarationError(ModelingError):
    pass


class UnboundIdentifierError(ModelingError):
    pass
