# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing
import json

from momba import model


def dump_structure(network: model.Network) -> typing.Mapping[typing.Any, typing.Any]:
    raise NotImplementedError()


def dump(network: model.Network, *, indent: typing.Optional[int] = None) -> bytes:
    """
    Takes a Momba automata network and exports it to the JANI format.

    Arguments:
        network: The Momba automata network to export to JANI.
        indent: Indentation of the final JSON.

    Returns:
        The model in UTF-8 encoded JANI format.
    """
    return json.dumps(dump_structure(network), indent=indent).encode('utf-8')
