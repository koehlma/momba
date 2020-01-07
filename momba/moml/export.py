# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import functools

from .. import model


class ExportContext:
    pass


@functools.singledispatch
def _export_expression(expr: model.Expression, ctx: ExportContext) -> None:
    raise NotImplementedError()


def _export_automaton(automaton: model.Automaton, ctx: ExportContext) -> None:
    pass
