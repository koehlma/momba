# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
An implementation of *Difference Bound Matrices* (DBMs) in pure Python.
"""

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import fractions
import itertools
import math

from .intervals import Interval


NumberType = t.Union[int, float, fractions.Fraction]


class InvalidBoundError(ValueError):
    pass


@d.dataclass(frozen=True, order=False)
class Bound:
    constant: NumberType
    is_strict: bool = False

    def __post_init__(self) -> None:
        if self.is_infinity and not self.is_strict:
            raise InvalidBoundError("bound with constant ∞ must be strict")

    @classmethod
    def less_than(cls, constant: NumberType) -> Bound:
        return Bound(constant, is_strict=True)

    @classmethod
    def less_or_equal(cls, constant: NumberType) -> Bound:
        return Bound(constant, is_strict=False)

    @property
    def is_infinity(self) -> bool:
        return math.isinf(self.constant)

    @property
    def is_integer(self) -> bool:
        if isinstance(self.constant, int):
            return True
        elif isinstance(self.constant, float):
            return self.constant.is_integer()
        else:
            assert isinstance(self.constant, fractions.Fraction)
            return self.constant.denominator == 1

    def add(self, other: Bound) -> Bound:
        if self.is_strict or other.is_strict:
            return Bound(self.constant + other.constant, is_strict=True)
        return Bound(self.constant + other.constant, is_strict=False)

    def __str__(self) -> str:
        operator = "<" if self.is_strict else "≤"
        constant = "∞" if self.is_infinity else str(self.constant)
        return f"{operator} {constant}"

    def __lt__(self, other: Bound) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        if self.is_strict is other.is_strict:
            return self.constant < other.constant
        elif self.is_strict and not other.is_strict:
            return self.constant <= other.constant
        assert not self.is_strict and other.is_strict
        return other.constant > self.constant

    def __le__(self, other: Bound) -> bool:
        if not isinstance(other, Bound):
            return NotImplemented
        return self == other or self < other


# perform some basic correctness checks
assert not (Bound(3, is_strict=False) < Bound(3, is_strict=False))
assert not (Bound(3, is_strict=False) < Bound(3, is_strict=True))
assert Bound(3, is_strict=True) < Bound(3, is_strict=False)
assert not (Bound(3, is_strict=True) < Bound(3, is_strict=True))
assert Bound(-3, is_strict=True) < Bound(2, is_strict=True)
assert Bound(3, is_strict=True) > Bound(2, is_strict=False)


@d.dataclass(frozen=True)
class ZeroClock:
    def __str__(self) -> str:
        return "0"


@d.dataclass(frozen=True)
class NamedClock:
    name: str

    def __str__(self) -> str:
        return f"Clock({self.name})"


ZERO_CLOCK = ZeroClock()


ClockT = t.TypeVar("ClockT", bound=t.Hashable)


@d.dataclass(frozen=True)
class Difference(t.Generic[ClockT]):
    left: t.Union[ClockT, ZeroClock]
    right: t.Union[ClockT, ZeroClock]

    def __str__(self) -> str:
        return f"{self.left} - {self.right}"

    def bound(self, bound: Bound) -> Constraint[ClockT]:
        return Constraint(self, bound=bound)

    def less_than(self, constant: NumberType) -> Constraint[ClockT]:
        return self.bound(Bound.less_than(constant))

    def less_or_equal(self, constant: NumberType) -> Constraint[ClockT]:
        return self.bound(Bound.less_or_equal(constant))


@d.dataclass(frozen=True)
class Constraint(t.Generic[ClockT]):
    difference: Difference[ClockT]
    bound: Bound

    def __str__(self) -> str:
        return f"{self.difference} {self.bound}"

    @property
    def clocks(self) -> t.AbstractSet[t.Union[ClockT, ZeroClock]]:
        return {self.difference.left, self.difference.right}

    @property
    def left(self) -> t.Union[ClockT, ZeroClock]:
        return self.difference.left

    @property
    def right(self) -> t.Union[ClockT, ZeroClock]:
        return self.difference.right

    @property
    def constant(self) -> NumberType:
        return self.bound.constant

    @property
    def is_strict(self) -> bool:
        return self.bound.is_strict


def difference(
    clock: t.Union[ZeroClock, ClockT], other: t.Union[ZeroClock, ClockT]
) -> Difference[ClockT]:
    return Difference(clock, other)


_INFINITY_BOUND = Bound.less_than(float("inf"))
_ZERO_BOUND = Bound.less_or_equal(0)

_Matrix = t.Dict[Difference[ClockT], Bound]


Clocks = t.AbstractSet[t.Union[ClockT, ZeroClock]]

_FrozenClocks = t.FrozenSet[t.Union[ClockT, ZeroClock]]


def _create_matrix(clocks: Clocks[ClockT]) -> _Matrix[ClockT]:
    matrix: _Matrix[ClockT] = {}
    for clock in clocks:
        # the value of clocks is always positive, and …
        matrix[difference(ZERO_CLOCK, clock)] = _ZERO_BOUND
        # … the difference of each clock and itself is zero
        matrix[difference(clock, clock)] = _ZERO_BOUND
    return matrix


def _freeze_clocks(clocks: Clocks[ClockT]) -> _FrozenClocks[ClockT]:
    return frozenset(clocks).union({ZERO_CLOCK})


class InvalidClockError(ValueError):
    pass


class AbstractDBM(abc.ABC, t.Generic[ClockT]):
    @property
    @abc.abstractmethod
    def constraints(self) -> t.AbstractSet[Constraint[ClockT]]:
        raise NotImplementedError()


@d.dataclass(frozen=True)
class FrozenDBM(AbstractDBM[ClockT], t.Generic[ClockT]):
    _clocks: _FrozenClocks[ClockT]
    _constraints: t.FrozenSet[Constraint[ClockT]]

    def create_dbm(self) -> DBM[ClockT]:
        dbm = DBM.create_unconstrained(self._clocks)
        dbm.constrain(*self._constraints)
        return dbm

    @property
    def constraints(self) -> t.AbstractSet[Constraint[ClockT]]:
        return self._constraints


@d.dataclass
class DBM(AbstractDBM[ClockT], t.Generic[ClockT]):
    _clocks: _FrozenClocks[ClockT]
    _matrix: _Matrix[ClockT]

    @classmethod
    def create_unconstrained(cls, clocks: Clocks[ClockT]) -> DBM[ClockT]:
        """Creates a DBM without any constraints."""
        frozen_clocks = _freeze_clocks(clocks)
        return DBM(frozen_clocks, _create_matrix(frozen_clocks))

    @classmethod
    def create_zero(cls, clocks: Clocks[ClockT]) -> DBM[ClockT]:
        """Creates a DBM where all clocks are constraint to be zero."""
        frozen_clocks = _freeze_clocks(clocks)
        matrix = _create_matrix(frozen_clocks)
        for clock in frozen_clocks:
            matrix[difference(clock, ZERO_CLOCK)] = _ZERO_BOUND
        dbm = DBM(frozen_clocks, matrix)
        dbm._canonicalize()
        return dbm

    def __post_init__(self) -> None:
        assert self._assert_invariant()

    def _assert_invariant(self) -> bool:
        for clock in self._clocks:
            assert self._matrix[difference(ZERO_CLOCK, clock)] <= _ZERO_BOUND
            assert self._matrix[difference(clock, clock)] <= _ZERO_BOUND
        return True

    def _canonicalize(self) -> None:
        for x, y, z in itertools.product(self._clocks, repeat=3):
            xy_bound = self.get_bound(x, y)
            yz_bound = self.get_bound(y, z)
            xz_bound = self.get_bound(x, z)
            if xz_bound > xy_bound.add(yz_bound):
                self._matrix[difference(x, z)] = xy_bound.add(yz_bound)
        assert self._assert_invariant()

    def _set_bound(
        self,
        left: t.Union[ClockT, ZeroClock],
        right: t.Union[ClockT, ZeroClock],
        bound: Bound,
    ) -> None:
        self._matrix[difference(left, right)] = bound
        assert self._assert_invariant()

    def get_bound(
        self, left: t.Union[ClockT, ZeroClock], right: t.Union[ClockT, ZeroClock]
    ) -> Bound:
        return self._matrix.get(difference(left, right), _INFINITY_BOUND)

    def extend_clocks(self, clocks: Clocks[ClockT]) -> DBM[ClockT]:
        matrix = dict(self._matrix)
        for clock in clocks:
            matrix[difference(ZERO_CLOCK, clock)] = _ZERO_BOUND
            matrix[difference(clock, clock)] = _ZERO_BOUND
        return DBM(_freeze_clocks(clocks | self.clocks), matrix)

    @property
    def clocks(self) -> t.AbstractSet[t.Union[ClockT, ZeroClock]]:
        return self._clocks

    @property
    def constraints(self) -> t.AbstractSet[Constraint[ClockT]]:
        return {Constraint(diff, bound) for diff, bound in self._matrix.items()}

    @property
    def is_empty(self) -> bool:
        for clock in self._clocks:
            if self.get_bound(clock, clock) < _ZERO_BOUND:
                return True
        return False

    def freeze(self) -> FrozenDBM[ClockT]:
        return FrozenDBM(self._clocks, frozenset(self.constraints))

    def get_interval(self, clock: ClockT) -> Interval[NumberType]:
        lower_bound = self.get_bound(ZERO_CLOCK, clock)
        upper_bound = self.get_bound(clock, ZERO_CLOCK)
        return Interval(
            -lower_bound.constant,
            upper_bound.constant,
            infimum_included=not lower_bound.is_strict,
            supremum_included=not upper_bound.is_strict,
        )

    def _constrain(self, difference: Difference[ClockT], by: Bound) -> None:
        if difference.left not in self._clocks or difference.right not in self._clocks:
            raise InvalidClockError(
                f"unable to constrain with {difference}: unknown clocks"
            )
        if by < self._matrix.get(difference, _INFINITY_BOUND):
            self._matrix[difference] = by

    def copy(self) -> DBM[ClockT]:
        return DBM(_clocks=self._clocks, _matrix=self._matrix.copy())

    def reset(self, clock: ClockT, value: NumberType = 0) -> None:
        assert not self.is_empty
        upper_bound = Bound.less_or_equal(value)
        lower_bound = Bound.less_or_equal(-value)
        for other in self._clocks:
            if other == clock:
                continue
            self._set_bound(
                clock, other, upper_bound.add(self.get_bound(ZERO_CLOCK, other))
            )
            self._set_bound(
                other, clock, self.get_bound(other, ZERO_CLOCK).add(lower_bound)
            )

    @t.overload
    def constrain(self, *constraints: Constraint[ClockT]) -> None:
        pass

    @t.overload
    def constrain(self, *, difference: Difference[ClockT], by: Bound) -> None:
        pass

    def constrain(
        self,
        *constraints: Constraint[ClockT],
        difference: t.Optional[Difference[ClockT]] = None,
        by: t.Optional[Bound] = None,
    ) -> None:
        try:
            for constraint in constraints:
                self._constrain(constraint.difference, constraint.bound)
            if difference is not None:
                assert by is not None, "use a static type checker!"
                self._constrain(difference, by)
        finally:
            # XXX: not optimal; use the knowledge about which clocks have been touched
            self._canonicalize()

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
                self._set_bound(
                    clock,
                    ZERO_CLOCK,
                    self.get_bound(clock, ZERO_CLOCK).add(
                        Bound.less_or_equal(time_delta)
                    ),
                )
        self._canonicalize()

    def advance_lower_bounds(self, delta: NumberType) -> None:
        """
        Advances the lower bounds of all clocks by the given amount of time.
        """
        for clock in self._clocks:
            if clock == ZERO_CLOCK:
                continue
            self._set_bound(
                ZERO_CLOCK,
                clock,
                self.get_bound(ZERO_CLOCK, clock).add(Bound.less_or_equal(-delta)),
            )
        self._canonicalize()

    def future(self) -> None:
        """
        Removes the upper bounds on all clocks.
        """
        self.advance_upper_bounds()

    def past(self) -> None:
        """
        Sets the lower bound of all clocks to zero.
        """
        try:
            for clock in self._clocks:
                if clock == ZERO_CLOCK:
                    continue
                self._set_bound(ZERO_CLOCK, clock, Bound.less_or_equal(0))
        finally:
            self._canonicalize()

    def intersect(self, other: DBM[ClockT]) -> None:
        assert other._clocks <= self._clocks
        try:
            for difference, bound in other._matrix.items():
                self._constrain(difference, bound)
        finally:
            self._canonicalize()


def print_constraints(dbm: AbstractDBM[ClockT]) -> None:
    for constraint in dbm.constraints:
        print(constraint)


def intersect(left: DBM[ClockT], right: DBM[ClockT]) -> DBM[ClockT]:
    result = left.copy()
    result.intersect(right)
    return result


def create_clock(name: str) -> NamedClock:
    return NamedClock(name)


def create_clocks(*names: str) -> t.Sequence[NamedClock]:
    return list(map(create_clock, names))
