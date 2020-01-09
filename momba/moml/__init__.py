# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

"""
The *Momba-Model Language* (MOML) is a model description language similar to JANI.

In comparison to JANI, MOML is less verbose and thus easier to read and write. The structure of
a MOML file closely resembles JANI. MOML is intended both, as a language to manually describe
quantitative models in and as an alternative representation of JANI models.

In comparison to other modeling languages like MODEST it is more low level.

This package contains a MOML parser, exporter, and converter to convert between MOML and JANI.
"""

from __future__ import annotations

from . import lexer


__all__ = ["lexer"]
