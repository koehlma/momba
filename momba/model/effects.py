# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import collections
import dataclasses

from . import context, errors

if t.TYPE_CHECKING:
    from . import expressions, types


class Target(abc.ABC):
    @abc.abstractmethod
    def infer_type(self, scope: context.Scope) -> types.Type:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_local_in(self, scope: context.Scope) -> bool:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class Name(Target):
    identifier: str

    def infer_type(self, scope: context.Scope) -> types.Type:
        declaration = scope.lookup(self.identifier)
        if not isinstance(declaration, context.VariableDeclaration):
            raise errors.NotAVariableError(
                f"invalid assignment to non-variable identifier {self.identifier}"
            )
        return declaration.typ

    def is_local_in(self, scope: context.Scope) -> bool:
        return scope.is_local(self.identifier)


@d.dataclass(frozen=True)
class Assignment:
    target: Target
    value: expressions.Expression
    index: int = 0

    def validate(self, scope: context.Scope) -> None:
        target_type = self.target.infer_type(scope)
        value_type = scope.get_type(self.value)
        if not target_type.is_assignable_from(value_type):
            raise errors.InvalidTypeError(
                f"cannot assign {value_type} to {target_type}"
            )


def are_compatible(assignments: t.Iterable[Assignment]) -> bool:
    groups: t.DefaultDict[int, t.Set[Target]] = collections.defaultdict(set)
    for assignment in assignments:
        target = assignment.target
        if target in groups[assignment.index]:
            return False
        groups[assignment.index].add(target)
    return True
