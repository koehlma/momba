# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

from momba.ext import jani
from momba.model import types


DIE_MODEL = pathlib.Path(__file__).parent / "resources" / "die.jani"


def test_basic_jani_import() -> None:
    network = jani.load_model(DIE_MODEL.read_text(encoding="utf-8"))

    assert len(network.ctx.automata) == 1
    assert network.ctx.global_scope.lookup("s").typ == types.INT.bound(0, 7)

    (automaton,) = network.ctx.automata

    assert len(automaton.locations) == 1


def test_basic_import_export() -> None:
    network = jani.load_model(DIE_MODEL.read_text(encoding="utf-8"))
    network = jani.load_model(jani.dump_model(network))

    assert len(network.ctx.automata) == 1
    assert network.ctx.global_scope.lookup("s").typ == types.INT.bound(0, 7)

    (automaton,) = network.ctx.automata

    assert len(automaton.locations) == 1
