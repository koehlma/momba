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


@pytest.mark.parametrize(
    "model", QVBS_MODELS, ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}"
)
def test_load_qvbs_model_without_properties(model: pathlib.Path) -> None:
    if model.stem == "csma-pta":
        pytest.skip("https://github.com/ahartmanns/qcomp/issues/103")
        return
    try:
        jani.load_model(model.read_text("utf-8-sig"), ignore_properties=True)
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")


@pytest.mark.parametrize(
    "model", QVBS_MODELS, ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}"
)
@pytest.mark.xfail(
    reason="our distinction between properties and expressions is broken"
)
def test_load_qvbs_model(model: pathlib.Path) -> None:
    if model.stem == "csma-pta":
        pytest.skip("https://github.com/ahartmanns/qcomp/issues/103")
        return
    try:
        jani.load_model(model.read_text("utf-8-sig"))
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")
