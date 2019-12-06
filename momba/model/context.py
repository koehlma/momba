# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses
import typing

from . import errors, expressions, types

if typing.TYPE_CHECKING:
    # XXX: stupid stuff to make mypy and the linter happy
    from . import assignments  # noqa: F401


Identifier = str

Typed = typing.Union['expressions.Expression', 'assignments.Target']


@dataclasses.dataclass(frozen=True)
class Declaration:
    identifier: Identifier

    def validate(self, scope: Scope) -> None:
        pass

    def is_constant_in(self, scope: Scope) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class VariableDeclaration(Declaration):
    typ: types.Type


@dataclasses.dataclass(frozen=True)
class ConstantDeclaration(Declaration):
    value: expressions.Expression

    def validate(self, scope: Scope) -> None:
        if not self.value.is_constant_in(scope):
            raise errors.NotAConstantError(
                f'value {self.value} of constant declaration is not a constant'
            )

    def is_constant_in(self, scope: Scope) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class ParameterDeclaration(Declaration):
    typ: types.Type

    def is_constant_in(self, scope: Scope) -> bool:
        return True


class Scope:
    context: Context
    parent: typing.Optional[Scope]

    _declarations: typing.Dict[Identifier, Declaration]
    _types: typing.Dict[Typed, types.Type]

    def __init__(self, context: Context, parent: typing.Optional[Scope] = None):
        self.context = context
        self.parent = parent
        self._declarations = {}
        self._types = {}

    def get_type(self, typed: Typed) -> types.Type:
        if typed not in self._types:
            inferred_type = typed.infer_type(self)
            inferred_type.validate_in(self)
            self._types[typed] = inferred_type
        return self._types[typed]

    def is_constant(self, expression: expressions.Expression) -> bool:
        return expression.is_constant_in(self)

    def lookup(self, identifier: Identifier) -> Declaration:
        try:
            return self._declarations[identifier]
        except KeyError:
            if self.parent is None:
                raise errors.UnboundIdentifierError(
                    f'identifier {identifier} is unbound in scope {self}'
                )
            return self.parent.lookup(identifier)

    def declare(self, declaration: Declaration) -> None:
        if declaration.identifier in self._declarations:
            raise errors.InvalidDeclarationError(
                f'identifier `{declaration.identifier} has already been declared'
            )
        declaration.validate(self)
        self._declarations[declaration.identifier] = declaration

    def declare_variable(self, identifier: Identifier, typ: types.Type) -> None:
        self.declare(VariableDeclaration(identifier, typ))

    def declare_constant(self, identifier: Identifier, value: expressions.Expression) -> None:
        self.declare(ConstantDeclaration(identifier, value))

    def declare_parameter(self, identifier: Identifier, typ: types.Type) -> None:
        self.declare(ParameterDeclaration(identifier, typ))


class Context:
    global_scope: Scope

    def __init__(self) -> None:
        self.global_scope = Scope(self)

    def new_scope(self) -> Scope:
        return Scope(self, parent=self.global_scope)
