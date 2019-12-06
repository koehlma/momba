# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing

from . import context
from .automata import Automaton


class Model:
    ctx: context.Context

    _automata: typing.Set[Automaton]

    def __init__(self) -> None:
        self.ctx = context.Context()
        self._automata = set()

    @property
    def automata(self) -> typing.AbstractSet[Automaton]:
        return self._automata

    def new_automaton(self) -> Automaton:
        automaton = Automaton(self.ctx)
        self._automata.add(automaton)
        return automaton
