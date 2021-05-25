# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc

from .. import model

from ._engine import engine as _engine

from . import zones

from .translator import Translation, translate_network

if t.TYPE_CHECKING:
    from .explore import Parameters


class InvalidModelType(Exception):
    pass


T = t.TypeVar("T")


@d.dataclass(frozen=True)
class CompiledNetwork:
    translation: Translation
    internal: t.Any


class TimeType(abc.ABC):
    """
    Base class for time representations.
    """

    @staticmethod
    @abc.abstractmethod
    def compile(
        network: model.Network, *, parameters: Parameters = None
    ) -> CompiledNetwork:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def load_valuations(cls: t.Type[T], valuations: t.Any) -> T:
        raise NotImplementedError()


class DiscreteTime(TimeType):
    """
    A representation of time without continuous-time clocks.
    """

    @staticmethod
    def compile(
        network: model.Network, *, parameters: Parameters = None
    ) -> CompiledNetwork:
        translation = translate_network(network, parameters=parameters)
        if not network.ctx.model_type.is_untimed:
            raise InvalidModelType(
                f"{network.ctx.model_type} is not a discrete time model type"
            )
        return CompiledNetwork(
            translation, _engine.Explorer.new_no_clocks(translation.json_network)
        )

    @classmethod
    def load_valuations(cls, valuations: t.Any) -> DiscreteTime:
        return cls()


@d.dataclass(frozen=True)
class GlobalTime(TimeType):
    zone: zones.Zone[float]

    @staticmethod
    def compile(
        network: model.Network, *, parameters: Parameters = None
    ) -> CompiledNetwork:
        translation = translate_network(
            network, parameters=parameters, global_clock=True
        )
        return CompiledNetwork(
            translation, _engine.Explorer.new_global_time(translation.json_network)
        )

    @classmethod
    def load_valuations(cls, valuations: t.Any) -> GlobalTime:
        return cls(zones._wrap_zone(valuations, zones.ZoneF64))


@d.dataclass(frozen=True)
class ZoneF64(TimeType):
    zone: zones.Zone[float]

    @staticmethod
    def compile(
        network: model.Network, *, parameters: Parameters = None
    ) -> CompiledNetwork:
        translation = translate_network(
            network, parameters=parameters, global_clock=False
        )
        return CompiledNetwork(
            translation, _engine.Explorer.new_global_time(translation.json_network)
        )

    @classmethod
    def load_valuations(cls, valuations: t.Any) -> ZoneF64:
        return cls(zones._wrap_zone(valuations, zones.ZoneF64))
