# -*- coding:utf-8 -*-
#
# Copyright (C) 2019–2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from momba import jani
from momba.explicit.translator import translate_model as translate_model_v2


MOMBA_EXPLICIT_PATH = (
    pathlib.Path(__file__).parent / ".." / "target" / "debug" / "momba-explicit"
)

assert MOMBA_EXPLICIT_PATH.exists()

QVBS_ROOT = pathlib.Path(__file__).parent / "resources" / "QVBS2019" / "benchmarks"

QVBS_MODELS = list(QVBS_ROOT.glob("**/*.jani"))

MOMBA_MODEL_PATH = pathlib.Path(__file__).parent / "temp"


@pytest.mark.parametrize(
    "model",
    [model for model in QVBS_MODELS if model.parts[-3] in {"mdp", "dtmc"}],
    ids=lambda model: f"{model.parts[-3].upper()}-{model.stem}",
)
def test_translate_v2_model(model: pathlib.Path) -> None:
    try:
        network = jani.load_model(model.read_text("utf-8-sig"), ignore_properties=True)
    except jani.UnsupportedJANIError as error:
        pytest.skip(f"uses unsupported JANI features {error.unsupported_features!r}")
    else:
        translated = translate_model_v2(network)
        momba_model_path = (
            MOMBA_MODEL_PATH / model.relative_to(QVBS_ROOT)
        ).with_suffix(".json")
        momba_model_path.parent.mkdir(parents=True, exist_ok=True)
        momba_model_path.write_text(json.dumps(translated))
        subprocess.check_call([MOMBA_EXPLICIT_PATH, momba_model_path])
