# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
A collection of data-structures and algorithms useful in the context of quantitative models.
"""

from __future__ import annotations

from . import dbm

from .dbm import DBM
from .intervals import Interval


__all__ = ["dbm", "DBM", "Interval"]
