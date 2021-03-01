# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from . import errors, expressions, types

if t.TYPE_CHECKING:
    from . import context


@d.dataclass(frozen=True)
class FunctionParameter:
    """
    A *function parameter*.

    Attributes
    ----------
    name:
        The name of the parameter.
    typ:
        The type of the parameter.
    """

    name: str
    typ: types.Type


@d.dataclass(frozen=True)
class FunctionDefinition:
    """
    A *function definition*.

    Attributes
    ----------
    name:
        The name of the defined function.
    parameters:
        The parameters of the function.
    returns:
        The return type of the function.
    body:
        The body of the function.
    """

    name: str
    parameters: t.Tuple[FunctionParameter, ...]
    returns: types.Type
    body: expressions.Expression


@d.dataclass(frozen=True)
class CallExpression(expressions.Expression):
    """
    A *call expression*.

    Attributes
    ----------
    function:
        The name of the function to be called.
    arguments:
        The arguments to be passed to the function.
    """

    function: str
    arguments: t.Tuple[expressions.Expression, ...]

    def infer_type(self, scope: context.Scope) -> types.Type:
        definition = scope.get_function(self.function)
        if len(self.arguments) != len(definition.parameters):
            raise errors.InvalidTypeError(
                f"function {self.function} expects {len(definition.parameters)}"
                f" arguments but {len(self.arguments)} are given"
            )
        for argument, parameter in zip(self.arguments, definition.parameters):
            argument_type = argument.infer_type(scope)
            if not parameter.typ.is_assignable_from(argument_type):
                raise errors.InvalidTypeError(
                    f"parameter {parameter.name} of function {self.function}"
                    f" has type {parameter.typ} but argument of type"
                    f" {argument_type} has been provided"
                )
        return definition.returns

    @property
    def children(self) -> t.Sequence[expressions.Expression]:
        return self.arguments
