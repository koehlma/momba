# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import fractions

from .. import model


Properties = t.Mapping[str, model.Property]
Result = t.Mapping[str, t.Union[bool, int, fractions.Fraction, str]]


class Checker(abc.ABC):
    @abc.abstractmethod
    def check(
        self,
        network: model.Network,
        *,
        properties: t.Optional[Properties] = None,
        property_names: t.Optional[t.Iterable[str]] = None,
    ) -> Result:
        raise NotImplementedError()
