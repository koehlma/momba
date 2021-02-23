# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations


class ModelingError(Exception):
    """ A modeling error."""


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
    """
    Declaration is invalid.
    """


class UnboundIdentifierError(ModelingError):
    pass


class NotFoundError(ModelingError):
    """Entity not found. """
