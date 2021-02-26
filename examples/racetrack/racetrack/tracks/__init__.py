# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

from importlib import resources

from ..model import Track


BARTO_BIG = Track.from_source(resources.read_text(__package__, "barto-big.track"))
BARTO_SMALL = Track.from_source(resources.read_text(__package__, "barto-small.track"))
RING = Track.from_source(resources.read_text(__package__, "ring.track"))
TINY = Track.from_source(resources.read_text(__package__, "tiny.track"))
