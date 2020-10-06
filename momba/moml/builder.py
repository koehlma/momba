# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

from .. import model

from ..model import expressions


class Builder:
    model_type: model.ModelType

    ctx: model.Context

    def __init__(self, model_type: model.ModelType) -> None:
        self.model_type = model_type
        self.ctx = model.Context(self.model_type)

    def declare_constant(
        self,
        name: str,
        typ: model.Type,
        *,
        value: t.Optional[expressions.MaybeExpression] = None,
        comment: t.Optional[str] = None,
    ) -> expressions.Identifier:
        self.ctx.global_scope.declare_constant(name, typ, value=value, comment=comment)
        return expressions.Identifier(name)

    def declare_variable(
        self,
        name: str,
        typ: model.Type,
        *,
        is_transient: t.Optional[bool] = None,
        initial_value: t.Optional[expressions.Expression] = None,
        comment: t.Optional[str] = None,
    ) -> expressions.Identifier:
        self.ctx.global_scope.declare_variable(
            name,
            typ,
            is_transient=is_transient,
            initial_value=initial_value,
            comment=comment,
        )
        return expressions.Identifier(name)
