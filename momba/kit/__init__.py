# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

"""
A collection of data-structures and algorithms useful in the context of quantitative models.
"""

from __future__ import annotations

from . import dbm

from .interval import Interval


__all__ = ['dbm', 'Interval']
