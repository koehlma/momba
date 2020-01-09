# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses


@dataclasses.dataclass(frozen=True)
class Metadata:
    version: t.Optional[str] = None
    author: t.Optional[str] = None
    description: t.Optional[str] = None
    doi: t.Optional[str] = None
    url: t.Optional[str] = None
