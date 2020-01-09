# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.kit import dbm

import pytest


def test_basic_operations() -> None:
    x, y, z = dbm.create_clocks("x", "y", "z")
    valuations = dbm.DBM.create_unconstrained({x, y, z})

    assert valuations.get_interval(x).infimum == 0
    assert valuations.get_interval(x).infimum_included is True
    assert valuations.get_interval(x).supremum == float("inf")
    assert valuations.get_interval(x).supremum_included is False
    assert not valuations.is_empty

    valuations.constrain(
        dbm.difference(x, dbm.ZERO_CLOCK).less_or_equal(5),
        dbm.difference(dbm.ZERO_CLOCK, x).less_than(-2),
    )

    assert valuations.get_interval(x).infimum == 2
    assert valuations.get_interval(x).infimum_included is False
    assert valuations.get_interval(x).supremum == 5
    assert valuations.get_interval(x).supremum_included is True
    assert not valuations.is_empty

    valuations.constrain(dbm.difference(dbm.ZERO_CLOCK, x).less_than(-5))

    assert valuations.is_empty


def test_unknown_clocks() -> None:
    x, y, z = dbm.create_clocks("x", "y", "z")
    valuations = dbm.DBM.create_unconstrained({x, y})

    with pytest.raises(dbm.InvalidClockError):
        valuations.constrain(dbm.difference(x, z).less_or_equal(3))


def test_clock_reset() -> None:
    x, y, z = dbm.create_clocks("x", "y", "z")
    valuations = dbm.DBM.create_unconstrained({x, y, z})

    valuations.reset(x, 5)

    assert valuations.get_interval(x).infimum == 5
    assert valuations.get_interval(x).infimum_included is True
    assert valuations.get_interval(x).supremum == 5
    assert valuations.get_interval(x).supremum_included is True

    valuations.constrain(dbm.difference(x, y).less_or_equal(2))

    assert valuations.get_interval(y).infimum == 3
    assert valuations.get_interval(y).infimum_included is True
    assert valuations.get_interval(y).supremum == float("inf")
    assert valuations.get_interval(y).supremum_included is False

    valuations.reset(y, 10)

    assert valuations.get_interval(y).infimum == 10
    assert valuations.get_interval(y).infimum_included is True
    assert valuations.get_interval(y).supremum == 10
    assert valuations.get_interval(y).supremum_included is True

    assert valuations.get_interval(x).infimum == 5
    assert valuations.get_interval(x).infimum_included is True
    assert valuations.get_interval(x).supremum == 5
    assert valuations.get_interval(x).supremum_included is True

    assert valuations.get_bound(x, y) == dbm.Bound.less_or_equal(-5)

    assert not valuations.is_empty


def test_intersection() -> None:
    x, y, z = dbm.create_clocks("x", "y", "z")
    valuations = dbm.DBM.create_zero({x, y, z})

    valuations.advance_upper_bounds(5)
    valuations.advance_lower_bounds(2)

    # constrain x and z to have the same value
    valuations.constrain(
        dbm.difference(x, z).less_or_equal(0), dbm.difference(z, x).less_or_equal(0)
    )

    invariant = dbm.DBM.create_unconstrained({x, y})
    invariant.constrain(dbm.difference(x, dbm.ZERO_CLOCK).less_than(3))  # x < 3

    valuations.intersect(invariant)

    assert valuations.get_interval(x).infimum == 2
    assert valuations.get_interval(x).infimum_included is True
    assert valuations.get_interval(x).supremum == 3
    assert valuations.get_interval(x).supremum_included is False

    assert valuations.get_interval(z).infimum == 2
    assert valuations.get_interval(z).infimum_included is True
    assert valuations.get_interval(z).supremum == 3
    assert valuations.get_interval(z).supremum_included is False

    assert valuations.get_interval(y).infimum == 2
    assert valuations.get_interval(y).infimum_included is True
    assert valuations.get_interval(y).supremum == 3
    assert valuations.get_interval(y).supremum_included is False

    valuations.reset(z, 42)

    assert valuations.get_interval(x).infimum == 2
    assert valuations.get_interval(x).infimum_included is True
    assert valuations.get_interval(x).supremum == 3
    assert valuations.get_interval(x).supremum_included is False

    assert valuations.get_interval(z).infimum == 42
    assert valuations.get_interval(z).infimum_included is True
    assert valuations.get_interval(z).supremum == 42
    assert valuations.get_interval(z).supremum_included is True

    assert valuations.get_interval(y).infimum == 2
    assert valuations.get_interval(y).infimum_included is True
    assert valuations.get_interval(y).supremum == 3
    assert valuations.get_interval(y).supremum_included is False
