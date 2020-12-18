# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Interfaces to `momba_engine` hiding away complexity.
"""

import dataclasses as d
import typing as t

from . import compiler

from ._engine import engine

from .. import model


@d.dataclass(frozen=True)
class CompiledNetwork:
    _compiled: t.Any

    def count_states(self) -> int:
        return self._compiled.count_states()


def compile_network(network: model.Network) -> CompiledNetwork:
    return CompiledNetwork(engine.CompiledModel(compiler.compile_network(network)))
