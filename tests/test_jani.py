# -*- coding:utf-8 -*-
#
# Copyright (C) 2019–2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

import pytest

from momba import jani
from momba.model import types


DIE_MODEL = pathlib.Path(__file__).parent / "resources" / "die.jani"

QVBS_MODELS = list(
    (pathlib.Path(__file__).parent / "resources" / "QVBS2019").glob("**/*.jani")
)


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


@pytest.mark.xfail(reason="JANI support is not fully implemented yet")
def test_load_qvbs_models() -> None:
    for model in QVBS_MODELS:
        try:
            jani.load_model(model.read_text("utf-8"))
        except jani.UnsupportedJANIError:
            pass
