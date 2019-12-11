# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import collections
import dataclasses

from . import context, errors, expressions, types


class Target(abc.ABC):
    @abc.abstractmethod
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class Identifier(Target):
    identifier: context.Identifier

    def infer_type(self, scope: context.Scope) -> types.Type:
        declaration = scope.lookup(self.identifier)
        if not isinstance(declaration, context.VariableDeclaration):
            raise errors.NotAVariableError(
                f'invalid assignment to non-variable identifier {self.identifier}'
            )
        return declaration.typ


@dataclasses.dataclass(frozen=True)
class Assignment:
    target: Target
    value: expressions.Expression
    index: int = 0

    def validate(self, scope: context.Scope) -> None:
        target_type = self.target.infer_type(scope)
        value_type = scope.get_type(self.value)
        if not target_type.is_assignable_from(value_type):
            raise errors.InvalidTypeError(
                f'cannot assign {value_type} to {target_type}'
            )


def are_compatible(assignments: t.Iterable[Assignment]) -> bool:
    groups: t.DefaultDict[int, t.AbstractSet[Target]] = collections.defaultdict(set)
    for assignment in assignments:
        target = assignment.target
        if target in groups[assignment.index]:
            return False
    return True
