# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Interfaces to `momba_engine` hiding away complexity.
"""

from __future__ import annotations

import dataclasses as d
import typing as t

import functools

from . import translator

from ._engine import engine

from .. import model


@d.dataclass(frozen=True)
class State:
    _state: t.Any

    @functools.cached_property
    def _as_json(self) -> str:
        return self._state.as_json()

    @functools.cached_property
    def successors(self) -> t.Sequence[State]:
        return tuple(map(State, self._state.successors()))


@d.dataclass(frozen=True)
class CompiledNetwork:
    translation: translator.Translation
    _compiled: t.Any

    def count_states(self) -> int:
        """
        Counts the number of states of the network.

        Warning: This method constructs the entire state space. Calling it
        might consume significant amounts of memory and computational
        resources.
        """
        return self._compiled.count_states()

    @functools.cached_property
    def initial_states(self) -> t.Sequence[State]:
        return tuple(map(State, self._compiled.initial_states()))


def compile_network(network: model.Network) -> CompiledNetwork:
    translation = translator.translate_network(network)
    return CompiledNetwork(translation, engine.CompiledModel(translation.json_network))
