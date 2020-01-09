# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import pathlib

from momba.ext import jani


network = jani.load_model(
    (pathlib.Path(__file__).parent / "die.jani").read_text(encoding="utf-8")
)
