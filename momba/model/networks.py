# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

from . import errors, types
from .actions import ActionPattern
from .automata import Instance
from .expressions import Expression

if t.TYPE_CHECKING:
    from . import context


@d.dataclass(frozen=True, eq=False)
class Link:
    """
    Represents a link of an automaton network.

    Attributes
    ----------
    vector:
        The synchronization vector.
    result:
        The resulting action pattern.
    """

    vector: t.Mapping[Instance, ActionPattern]
    result: t.Optional[ActionPattern] = None
    condition: t.Optional[Expression] = None

    def construct_scope(self, ctx: context.Context) -> context.Scope:
        scope = ctx.global_scope.create_child_scope()
        for pattern in self.vector.values():
            pattern.declare_in(scope)
        return scope


class Network:
    """
    Represents an automaton network.

    Attributes
    ----------
    ctx:
        The :class:`Context` associated with the network.
    name:
        The optional name of the network.
    """

    ctx: context.Context

    name: t.Optional[str]

    _initial_restriction: t.Optional[Expression]

    _instances: t.List[Instance]
    _links: t.List[Link]

    def __init__(self, ctx: context.Context, *, name: t.Optional[str] = None) -> None:
        self.ctx = ctx
        self.name = name
        self._initial_restriction = None
        self._instances = []
        self._links = []

    @property
    def links(self) -> t.List[Link]:
        """
        The set of links of the network.
        """
        return self._links

    @property
    def instances(self) -> t.List[Instance]:
        """
        The set of instances of the network.
        """
        return self._instances

    @property
    def initial_restriction(self) -> t.Optional[Expression]:
        """
        An optional restriction to be satisfied by initial states.

        This property can be set *only once* on a network. The expression
        has to be boolean.

        Raises :class:`~errors.ModelingError` when set twice or to a
        non-boolean expression.
        """
        return self._initial_restriction

    @initial_restriction.setter
    def initial_restriction(self, initial_restriction: Expression) -> None:
        if self._initial_restriction is not None:
            raise errors.ModelingError(
                "restriction of initial valuations has already been set"
            )
        if self.ctx.global_scope.get_type(initial_restriction) != types.BOOL:
            raise errors.InvalidTypeError(
                "restriction of initial valuations must have type `types.BOOL`"
            )
        self._initial_restriction = initial_restriction

    def add_instance(self, instance: Instance) -> None:
        """
        Adds an instance to the network.

        The instance and network are required to be in the same
        modeling context.
        """
        assert instance.automaton.ctx is self.ctx
        if instance not in self._instances:
            self._instances.append(instance)

    def create_link(
        self,
        vector: t.Mapping[Instance, ActionPattern],
        *,
        result: t.Optional[ActionPattern] = None,
        condition: t.Optional[Expression] = None
    ) -> Link:
        """
        Creates a synchronization link between automata instances.

        The parameter `vector` is a mapping from the instances
        participating in the synchronization to action patterns with
        which they participate.

        The parameter `result` is the action pattern resulting
        from synchronizing.
        """
        link = Link(vector, result=result, condition=condition)
        for instance in vector.keys():
            self.add_instance(instance)
        self._links.append(link)
        return link
