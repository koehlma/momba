# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc


LocationT = t.TypeVar("LocationT", bound=t.Hashable)
EdgeT = t.TypeVar("EdgeT", bound=t.Hashable)


class TS(abc.ABC, t.Generic[LocationT, EdgeT]):
    @property
    @abc.abstractmethod
    def initial_locations(self) -> t.AbstractSet[LocationT]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def edges(self) -> t.AbstractSet[EdgeT]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def locations(self) -> t.AbstractSet[LocationT]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edges_from(self, source: LocationT) -> t.AbstractSet[EdgeT]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edges_to(self, destination: LocationT) -> t.AbstractSet[EdgeT]:
        raise NotImplementedError()
