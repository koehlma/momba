# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.model import context, types
from momba.model.types import array_of

import pytest


def test_basic_types() -> None:
    assert not types.INT.is_assignable_from(types.REAL)
    assert not types.INT.is_assignable_from(types.BOOL)
    assert types.INT.is_assignable_from(types.INT)

    assert not types.BOOL.is_assignable_from(types.INT)
    assert not types.BOOL.is_assignable_from(types.REAL)
    assert types.BOOL.is_assignable_from(types.BOOL)

    assert types.REAL.is_assignable_from(types.INT)
    assert not types.REAL.is_assignable_from(types.BOOL)
    assert types.REAL.is_assignable_from(types.REAL)

    assert types.REAL.is_numeric
    assert types.INT.is_numeric
    assert not types.BOOL.is_numeric


# noinspection PyStatementEffect
def test_bounded_types() -> None:
    assert types.INT.bound(0, 3).is_assignable_from(types.INT)
    assert types.INT.bound(3, ...).is_numeric
    assert types.REAL.bound(..., "π").is_assignable_from(types.INT.bound(0, 5))
    assert not types.INT.bound(2, 10).is_assignable_from(types.REAL)
    assert not types.INT.bound(2, 10).is_assignable_from(types.REAL.bound("π", ...))

    with pytest.raises(types.InvalidBoundError):
        ctx = context.Context()
        scope = ctx.global_scope.create_child_scope()
        types.INT.bound(..., "π").validate_in(scope)
    with pytest.raises(types.InvalidBoundError):
        types.INT.bound(..., ...)


def test_clock_type() -> None:
    assert types.CLOCK.is_assignable_from(types.INT)
    assert types.CLOCK.is_assignable_from(types.INT.bound(0, 8))
    assert not types.CLOCK.is_assignable_from(types.REAL)
    assert not types.CLOCK.is_assignable_from(types.REAL.bound(..., "π"))
    assert not types.CLOCK.is_assignable_from(types.BOOL)
    assert types.CLOCK.is_numeric


def test_continuous_type() -> None:
    assert types.CONTINUOUS.is_assignable_from(types.INT)
    assert types.CONTINUOUS.is_assignable_from(types.INT.bound(0, 8))
    assert types.CONTINUOUS.is_assignable_from(types.REAL)
    assert types.CONTINUOUS.is_assignable_from(types.REAL.bound(..., "π"))
    assert not types.CONTINUOUS.is_assignable_from(types.BOOL)
    assert types.CONTINUOUS.is_numeric


def test_array_type() -> None:
    assert array_of(types.REAL).is_assignable_from(array_of(types.INT))
    assert not array_of(types.REAL).is_assignable_from(types.INT)
    assert not array_of(types.INT).is_assignable_from(array_of(types.BOOL))
    assert array_of(types.INT.bound(5, 10)).is_assignable_from(array_of(types.INT))

    # arrays of arrays
    assert array_of(array_of(types.REAL)).is_assignable_from(
        array_of(array_of(types.INT))
    )
    assert not array_of(array_of(types.REAL)).is_assignable_from(array_of(types.INT))
