# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc

from . import errors

if t.TYPE_CHECKING:
    from . import context, expressions, types


@d.dataclass(frozen=True)
class ActionParameter:
    """
    Represents an action parameter.

    Attributes
    ----------
    typ:
        The type of the parameter.
    comment:
        An optional comment for the parameter.
    """

    typ: types.Type
    comment: t.Optional[str] = None


@d.dataclass(frozen=True)
class ActionType:
    """
    Represents an action type.

    Attributes
    ----------
    label:
        The label of the action type.
    parameters:
        The parameters of the action type.
    comment:
        An optional comment for the action type.
    """

    label: str
    parameters: t.Tuple[ActionParameter, ...] = ()
    comment: t.Optional[str] = None

    @property
    def has_parameters(self) -> bool:
        """
        Returns :obj:`True` if and only if the action type has parameters.
        """
        return bool(self.parameters)

    @property
    def arity(self) -> int:
        """
        The arity, i.e., number of parameters, of the action type.
        """
        return len(self.parameters)

    def create_pattern(self, *arguments: ActionArgument) -> ActionPattern:
        return ActionPattern(self, arguments=arguments)


class ActionArgument(abc.ABC):
    """
    Represents an argument for an action pattern.
    """


@d.dataclass(frozen=True)
class WriteArgument(ActionArgument):
    """
    Writes the value of the expression for the respective parameter.
    """

    # is evaluated in the automata's scope, not in the edge scope
    expression: expressions.Expression


@d.dataclass(frozen=True)
class ReadArgument(ActionArgument):
    """
    Reads a value from the respective parameter.
    """

    # gets declared in the edge scope
    identifier: str


@d.dataclass(frozen=True)
class GuardArgument(ActionArgument):
    """
    Represents an argument to be used in a guard.
    """

    identifier: str


@d.dataclass(frozen=True)
class ActionPattern:
    """
    Represents an action pattern.

    Attributes
    ----------
    action_type:
        The type of the action pattern.
    arguments:
        The arguments of the action pattern.
    """

    action_type: ActionType
    arguments: t.Tuple[ActionArgument, ...] = ()

    def __post_init__(self) -> None:
        if len(self.action_type.parameters) != len(self.arguments):
            raise errors.InvalidTypeError(
                f"action type {self.action_type} requires {self.action_type.arity} "
                f"arguments but {len(self.arguments)} were given"
            )

    def declare_in(self, scope: context.Scope) -> None:
        """
        Declares the identifiers of the pattern in the given scope.

        .. warning::
            Experimental feature used for value passing only.
        """
        for argument, parameter in zip(self.arguments, self.action_type.parameters):
            if isinstance(argument, GuardArgument):
                if scope.is_declared(argument.identifier):
                    assert scope.lookup(argument.identifier).typ == parameter.typ
                else:
                    scope.declare_variable(
                        argument.identifier, parameter.typ, is_transient=True
                    )
            elif isinstance(argument, ReadArgument):
                scope.declare_variable(
                    argument.identifier, parameter.typ, is_transient=True
                )
