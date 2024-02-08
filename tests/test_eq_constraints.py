# -*- coding:utf-8 -*-
#
# Copyright (C) 2024, Raffael Senn <raffael.senn@uni-konstanz.de>

import pathlib


from momba import engine, jani
from momba.engine import time

EQ_CONSTRAINTS_MODEL = (
    pathlib.Path(__file__).parent / "resources" / "eq_constraints.jani"
)


def test_eq_constraints_exploration() -> None:
    network = jani.load_model(EQ_CONSTRAINTS_MODEL.read_text(encoding="utf-8"))
    e = engine.Explorer(network, time.GlobalTime)
    print("Done!")
