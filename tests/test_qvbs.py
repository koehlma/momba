# -*- coding:utf-8 -*-
#
# Copyright (C) 2019–2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

import pytest

from momba import jani
from momba.engine import translator

from momba.explicit.translator import translate_model as translate_model_v2


QVBS_MODELS = list(
    (pathlib.Path(__file__).parent / "resources" / "QVBS2019").glob("**/*.jani")
)


@pytest.mark.parametrize(
    "model", QVBS_MODELS, ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}"
)
def test_load_model(model: pathlib.Path) -> None:
    if model.stem == "csma-pta":
        pytest.skip(
            "Model is invalid, see https://github.com/ahartmanns/qcomp/issues/103."
        )
        return
    try:
        jani.load_model(model.read_text("utf-8-sig"))
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")


@pytest.mark.parametrize(
    "model",
    [model for model in QVBS_MODELS if model.parts[-3] in {"mdp"}],
    ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}",
)
@pytest.mark.skip(reason="new simulation engine has not been fully implemented yet")
def test_translate_model(model: pathlib.Path) -> None:
    try:
        network = jani.load_model(model.read_text("utf-8-sig"), ignore_properties=True)
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")
    else:
        translator.translate_network(network)


@pytest.mark.parametrize(
    "model",
    [
        model
        for model in QVBS_MODELS
        if model.parts[-3] in {"mdp", "dtmc", "ctmc", "ta", "pta", "ma"}
    ],
    ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}",
)
@pytest.mark.skip(reason="new simulation engine has not been fully implemented yet")
def test_translate_v2_model(model: pathlib.Path) -> None:
    try:
        network = jani.load_model(model.read_text("utf-8-sig"), ignore_properties=True)
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")
    else:
        translate_model_v2(network)
