# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

try:
    import momba_engine as engine

    zones = engine.zones
except ImportError:
    raise ImportError(
        "Missing optional dependency `momba_engine`.\n"
        "Using Momba's engine requires installing `momba_engine`."
    )


__all__ = ["engine", "zones"]
