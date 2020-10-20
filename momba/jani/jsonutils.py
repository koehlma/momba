# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses


# XXX: ignore this type definition, mypy does not support recursive types
JSON = t.Union[None, int, float, str, t.Sequence["JSON"], t.Mapping[str, "JSON"]]  # type: ignore

_JANIMap = t.Dict[str, JSON]  # type: ignore


@dataclasses.dataclass(frozen=True)
class JSONObject:
    parent: t.Optional[JSONObject]
    key: t.Union[int, str, None]


@dataclasses.dataclass(frozen=True)
class JSONPrimitive(JSONObject):
    value: t.Union[None, int, float, str]


class JSONSequence(JSONObject, t.Sequence[JSON]):
    pass


class JSONMapping(t.Mapping[str, JSON]):
    pass
