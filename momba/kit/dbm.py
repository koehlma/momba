# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

"""
An implementation of *Difference Bound Matrices* (DBMs).
"""

from __future__ import annotations

import typing as t

import abc
import dataclasses
import functools
import itertools
import math

from .interval import Interval


NumberType = t.Union[int, float]


@functools.total_ordering
@dataclasses.dataclass(frozen=True, order=False)
class Bound:
    constant: NumberType
    is_strict: bool = False

    @classmethod
    def less_than(cls, constant: NumberType) -> Bound:
        return Bound(constant, is_strict=True)

    @classmethod
    def less_or_equal(cls, constant: NumberType) -> Bound:
        return Bound(constant, is_strict=False)

    @property
    def is_infinity(self) -> bool:
        return math.isinf(self.constant)

    def add(self, other: Bound) -> Bound:
        if not isinstance(other, Bound):
            return NotImplemented
        if self.is_strict or other.is_strict:
            return Bound(self.constant + other.constant, is_strict=True)
        return Bound(self.constant + other.constant, is_strict=False)

    def __str__(self) -> str:
        if math.isinf(self.constant):
            return f'{"<" if self.is_strict else "≤"} ∞'
        else:
            return f'{"<" if self.is_strict else "≤"} {self.constant}'

    def __lt__(self, other: Bound) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        if self.is_strict is other.is_strict:
            return self.constant < other.constant
        elif self.is_strict and not other.is_strict:
            return self.constant <= other.constant
        assert not self.is_strict and other.is_strict
        return other.constant > self.constant


# perform some basic correctness tests
assert not (Bound(3, is_strict=False) < Bound(3, is_strict=False))
assert not (Bound(3, is_strict=False) < Bound(3, is_strict=True))
assert Bound(3, is_strict=True) < Bound(3, is_strict=False)
assert not (Bound(3, is_strict=True) < Bound(3, is_strict=True))
assert Bound(-3, is_strict=True) < Bound(2, is_strict=True)
assert Bound(3, is_strict=True) > Bound(2, is_strict=False)


class Clock(abc.ABC):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class _Clock(Clock):
    name: str


class _ZeroClock(Clock):
    name: str = '0'


ZERO_CLOCK = _ZeroClock()


@dataclasses.dataclass(frozen=True)
class Difference:
    left: Clock
    right: Clock

    def __str__(self) -> str:
        return f'{self.left} - {self.right}'

    def less_than(self, constant: NumberType) -> Constraint:
        return Constraint(self, bound=Bound.less_than(constant))

    def less_or_equal(self, constant: NumberType) -> Constraint:
        return Constraint(self, bound=Bound.less_or_equal(constant))


@dataclasses.dataclass(frozen=True)
class Constraint:
    difference: Difference
    bound: Bound

    def __str__(self) -> str:
        return f'{self.difference} {self.bound}'

    @property
    def clocks(self) -> t.AbstractSet[Clock]:
        return {self.difference.left, self.difference.right}

    @property
    def left(self) -> Clock:
        return self.difference.left

    @property
    def right(self) -> Clock:
        return self.difference.right

    @property
    def constant(self) -> NumberType:
        return self.bound.constant

    @property
    def is_strict(self) -> bool:
        return self.bound.is_strict


def create_clock(name: str) -> Clock:
    return _Clock(name)


def create_clocks(*names: str) -> t.Sequence[Clock]:
    return list(map(create_clock, names))


def difference(clock: Clock, other: Clock) -> Difference:
    return Difference(clock, other)


_INFINITY_BOUND = Bound.less_than(float('inf'))
_ZERO_BOUND = Bound.less_or_equal(0)

_Matrix = t.Dict[Difference, Bound]


Clocks = t.AbstractSet[Clock]


def _create_matrix(clocks: Clocks) -> _Matrix:
    matrix: _Matrix = {}
    for clock in clocks:
        # the value of clocks is always positive, and …
        matrix[difference(ZERO_CLOCK, clock)] = _ZERO_BOUND
        # … the difference of each clock and itself is zero
        matrix[difference(clock, clock)] = _ZERO_BOUND
    return matrix


def _freeze_clocks(clocks: Clocks) -> t.FrozenSet[Clock]:
    return frozenset(clocks).union({ZERO_CLOCK})


class UnknownClockError(ValueError):
    pass


@dataclasses.dataclass
class DBM:
    """
    Do not create instances directly but only with the helper functions.
    """
    _clocks: t.FrozenSet[Clock]
    _matrix: _Matrix

    def _set(self, left: Clock, right: Clock, bound: Bound) -> None:
        self._matrix[difference(left, right)] = bound

    def get_bound(self, left: Clock, right: Clock) -> Bound:
        return self._matrix.get(difference(left, right), _INFINITY_BOUND)

    @classmethod
    def create_unconstrained(cls, clocks: t.AbstractSet[Clock]) -> DBM:
        """ Creates a DBM without any constraints. """
        frozen_clocks = _freeze_clocks(clocks)
        return DBM(frozen_clocks, _create_matrix(frozen_clocks))

    @classmethod
    def create_zero(cls, clocks: t.AbstractSet[Clock]) -> DBM:
        """ Creates a DBM where all clocks are constraint to be zero. """
        frozen_clocks = _freeze_clocks(clocks)
        matrix = _create_matrix(frozen_clocks)
        for clock in frozen_clocks:
            matrix[difference(clock, ZERO_CLOCK)] = _ZERO_BOUND
        return DBM(frozen_clocks, matrix)

    @property
    def clocks(self) -> t.AbstractSet[Clock]:
        return self._clocks

    @property
    def constraints(self) -> t.AbstractSet[Constraint]:
        return {
            Constraint(diff, bound)
            for diff, bound in self._matrix.items()
        }

    @property
    def is_empty(self) -> bool:
        for clock in self._clocks:
            if self.get_bound(clock, clock) < _ZERO_BOUND:
                return True
        return False

    def get_interval(self, clock: Clock) -> Interval:
        lower_bound = self.get_bound(ZERO_CLOCK, clock)
        upper_bound = self.get_bound(clock, ZERO_CLOCK)
        return Interval(
            -lower_bound.constant, upper_bound.constant,
            infimum_included=not lower_bound.is_strict,
            supremum_included=not upper_bound.is_strict
        )

    def _constrain(self, constraint: Constraint) -> None:
        if constraint.left not in self._clocks or constraint.right not in self._clocks:
            raise UnknownClockError(
                f'cannot constrain {self} with {constraint}: unknown clocks'
            )
        if constraint.bound < self._matrix.get(constraint.difference, _INFINITY_BOUND):
            self._matrix[constraint.difference] = constraint.bound

    def reset(self, clock: Clock, value: NumberType = 0) -> None:
        assert not self.is_empty
        upper_bound = Bound.less_or_equal(value)
        lower_bound = Bound.less_or_equal(-value)
        for other in self._clocks:
            if other == clock:
                continue
            self._set(clock, other, upper_bound.add(self.get_bound(ZERO_CLOCK, other)))
            self._set(other, clock, self.get_bound(other, ZERO_CLOCK).add(lower_bound))

    def constrain(self, *constraints: Constraint) -> None:
        for constraint in constraints:
            self._constrain(constraint)
        self._canonicalize()  # XXX: not optimal

    def advance_upper_bounds(self, time_delta: t.Optional[NumberType] = None) -> None:
        """
        Advances the upper bounds of all clocks by the given amount of time.
        """
        for clock in self._clocks:
            if clock == ZERO_CLOCK:
                continue
            if time_delta is None:
                # delete the upper bound of the clock
                del self._matrix[difference(clock, ZERO_CLOCK)]
            else:
                # advance the upper bound by the given time
                self._set(
                    clock, ZERO_CLOCK,
                    self.get_bound(clock, ZERO_CLOCK).add(Bound.less_or_equal(time_delta))
                )

    def advance_lower_bounds(self, delta: NumberType) -> None:
        """
        Advances the lower bounds of all clocks by the given amount of time.
        """
        for clock in self._clocks:
            if clock == ZERO_CLOCK:
                continue
            self._set(
                ZERO_CLOCK, clock,
                self.get_bound(ZERO_CLOCK, clock).add(Bound.less_or_equal(-delta))
            )

    def future(self) -> None:
        """
        Removes the upper bounds on all clocks.
        """
        self.advance_upper_bounds()

    def past(self) -> None:
        """
        Sets the lower bound of all clocks to zero.
        """
        for clock in self._clocks:
            if clock == ZERO_CLOCK:
                continue
            self._set(ZERO_CLOCK, clock, Bound.less_or_equal(0))
        self._canonicalize()

    def intersect(self, other: DBM) -> None:
        assert other._clocks <= self._clocks
        for constraint in other.constraints:
            self._constrain(constraint)
        self._canonicalize()

    def _canonicalize(self) -> None:
        for x, y, z in itertools.product(self._clocks, repeat=3):
            xy_bound = self.get_bound(x, y)
            yz_bound = self.get_bound(y, z)
            xz_bound = self.get_bound(x, z)
            if xz_bound > xy_bound.add(yz_bound):
                self._matrix[difference(x, z)] = xy_bound.add(yz_bound)


def print_constraints(dbm: DBM) -> None:
    for constraint in dbm.constraints:
        print(constraint)
