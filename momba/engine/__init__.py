# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Interfaces to `momba_engine` hiding away complexity.
"""

import dataclasses as d
import typing as t

import momba_engine

from .. import model
from ..explore import compiler


@d.dataclass(frozen=True)
class CompiledNetwork:
    _compiled: t.Any

    def count_states(self) -> int:
        return self._compiled.count_states()


def compile_network(network: model.Network) -> CompiledNetwork:
    return CompiledNetwork(
        momba_engine.CompiledModel(compiler.compile_network(network))
    )
