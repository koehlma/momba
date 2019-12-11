# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

from . import context, errors, types
from .automata import Automaton

if t.TYPE_CHECKING:
    from .expressions import Expression


class Network:
    """
    The core class representing a network of interacting SHAs.
    """
    ctx: context.Context

    _restrict_initial: t.Optional[Expression]
    _automata: t.Set[Automaton]

    def __init__(self, model_type: context.ModelType = context.ModelType.SHA) -> None:
        self.ctx = context.Context(model_type)
        self._restrict_initial = None
        self._automata = set()

    @property
    def restrict_initial(self) -> t.Optional[Expression]:
        return self._restrict_initial

    @restrict_initial.setter
    def restrict_initial(self, restrict_initial: Expression) -> None:
        if self._restrict_initial is not None:
            raise errors.InvalidOperationError(
                f'restriction of initial valuations has already been set'
            )
        if self.ctx.global_scope.get_type(restrict_initial) != types.BOOL:
            raise errors.InvalidTypeError(
                f'restriction of initial valuations must have type `types.BOOL`'
            )
        self._restrict_initial = restrict_initial

    @property
    def automata(self) -> t.AbstractSet[Automaton]:
        """
        The set of :py_class:`momba.Automaton` making up the model.
        """
        return self._automata

    def new_automaton(self) -> Automaton:
        automaton = Automaton(self.ctx)
        self._automata.add(automaton)
        return automaton
