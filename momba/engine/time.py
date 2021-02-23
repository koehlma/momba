# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

from .translator import Translation

import abc

from .. import model

from ._engine import engine as _engine


class InvalidModelType(Exception):
    pass


class TimeType(abc.ABC):
    """
    Base class for time representations.
    """

    @staticmethod
    @abc.abstractmethod
    def compile(network: model.Network, translation: Translation) -> t.Any:
        raise NotImplementedError()


class DiscreteTime(TimeType):
    """
    A representation of time without continuous-time clocks.
    """

    @staticmethod
    def compile(network: model.Network, translation: Translation) -> t.Any:
        if not network.ctx.model_type.is_untimed:
            raise InvalidModelType(
                f"{network.ctx.model_type} is not a discrete time model type"
            )
        return _engine.Explorer(translation.json_network)
