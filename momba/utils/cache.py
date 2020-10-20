# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import functools


T = t.TypeVar("T")


# XXX: this is here just to make `mypy` happy
def lru_cache(func: T) -> T:
    return functools.lru_cache()(func)  # type: ignore
