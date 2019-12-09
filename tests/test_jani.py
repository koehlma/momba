# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

from momba.ext import jani_model
from momba.model import types

import pytest


DIE_MODEL = pathlib.Path(__file__).parent / 'resources' / 'die.jani'


def test_basic_jani_import():
    with pytest.warns(UserWarning):
        network = jani_model.load(DIE_MODEL.read_text(encoding='utf-8'))

    assert len(network.automata) == 1
    assert network.ctx.global_scope.lookup('s').typ == types.INT[0, 7]

    automaton, = network.automata

    assert len(automaton.locations) == 1
