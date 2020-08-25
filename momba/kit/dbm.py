# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

"""
An implementation of *Difference Bound Matrices* (DBMs) in pure Python.
"""

from __future__ import annotations

import typing as t

import abc
import dataclasses
import itertools
import math

from .interval import Interval


NumberType = t.Union[int, float]


class InvalidBoundError(ValueError):
    pass


@dataclasses.dataclass(frozen=True, order=False)
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
        return isinstance(self.constant, int) or self.constant.is_integer()

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


class BaseClock(abc.ABC):
    pass


class _ZeroClock(BaseClock):
    def __str__(self) -> str:
        return "0"


@dataclasses.dataclass(frozen=True)
class Clock(BaseClock):
    name: str

    def __str__(self) -> str:
        return f"Clock({self.name})"


ZERO_CLOCK = _ZeroClock()


@dataclasses.dataclass(frozen=True)
class Difference:
    left: BaseClock
    right: BaseClock

    def __str__(self) -> str:
        return f"{self.left} - {self.right}"

    def bound(self, bound: Bound) -> Constraint:
        return Constraint(self, bound=bound)

    def less_than(self, constant: NumberType) -> Constraint:
        return self.bound(Bound.less_than(constant))

    def less_or_equal(self, constant: NumberType) -> Constraint:
        return self.bound(Bound.less_or_equal(constant))


@dataclasses.dataclass(frozen=True)
class Constraint:
    difference: Difference
    bound: Bound

    def __str__(self) -> str:
        return f"{self.difference} {self.bound}"

    @property
    def clocks(self) -> t.AbstractSet[BaseClock]:
        return {self.difference.left, self.difference.right}

    @property
    def left(self) -> BaseClock:
        return self.difference.left

    @property
    def right(self) -> BaseClock:
        return self.difference.right

    @property
    def constant(self) -> NumberType:
        return self.bound.constant

    @property
    def is_strict(self) -> bool:
        return self.bound.is_strict


def create_clock(name: str) -> Clock:
    return Clock(name)


def create_clocks(*names: str) -> t.Sequence[Clock]:
    return list(map(create_clock, names))


def difference(clock: BaseClock, other: BaseClock) -> Difference:
    return Difference(clock, other)


_INFINITY_BOUND = Bound.less_than(float("inf"))
_ZERO_BOUND = Bound.less_or_equal(0)

_Matrix = t.Dict[Difference, Bound]


Clocks = t.AbstractSet[BaseClock]

_FrozenClocks = t.FrozenSet[BaseClock]


def _create_matrix(clocks: Clocks) -> _Matrix:
    matrix: _Matrix = {}
    for clock in clocks:
        # the value of clocks is always positive, and …
        matrix[difference(ZERO_CLOCK, clock)] = _ZERO_BOUND
        # … the difference of each clock and itself is zero
        matrix[difference(clock, clock)] = _ZERO_BOUND
    return matrix


def _freeze_clocks(clocks: Clocks) -> _FrozenClocks:
    return frozenset(clocks).union({ZERO_CLOCK})


class InvalidClockError(ValueError):
    pass


class AbstractDBM(abc.ABC):
    @property
    @abc.abstractmethod
    def constraints(self) -> t.AbstractSet[Constraint]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class FrozenDBM(AbstractDBM):
    _clocks: _FrozenClocks
    _constraints: t.FrozenSet[Constraint]

    def create_dbm(self) -> DBM:
        dbm = DBM.create_unconstrained(self._clocks)
        dbm.constrain(*self._constraints)
        return dbm

    @property
    def constraints(self) -> t.AbstractSet[Constraint]:
        return self._constraints


@dataclasses.dataclass
class DBM(AbstractDBM):
    _clocks: _FrozenClocks
    _matrix: _Matrix

    @classmethod
    def create_unconstrained(cls, clocks: Clocks) -> DBM:
        """ Creates a DBM without any constraints. """
        frozen_clocks = _freeze_clocks(clocks)
        return DBM(frozen_clocks, _create_matrix(frozen_clocks))

    @classmethod
    def create_zero(cls, clocks: Clocks) -> DBM:
        """ Creates a DBM where all clocks are constraint to be zero. """
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

    def _set_bound(self, left: BaseClock, right: BaseClock, bound: Bound) -> None:
        self._matrix[difference(left, right)] = bound
        assert self._assert_invariant()

    def get_bound(self, left: BaseClock, right: BaseClock) -> Bound:
        return self._matrix.get(difference(left, right), _INFINITY_BOUND)

    @property
    def clocks(self) -> t.AbstractSet[BaseClock]:
        return self._clocks

    @property
    def constraints(self) -> t.AbstractSet[Constraint]:
        return {Constraint(diff, bound) for diff, bound in self._matrix.items()}

    @property
    def is_empty(self) -> bool:
        for clock in self._clocks:
            if self.get_bound(clock, clock) < _ZERO_BOUND:
                return True
        return False

    def freeze(self) -> FrozenDBM:
        return FrozenDBM(self._clocks, frozenset(self.constraints))

    def get_interval(self, clock: Clock) -> Interval:
        lower_bound = self.get_bound(ZERO_CLOCK, clock)
        upper_bound = self.get_bound(clock, ZERO_CLOCK)
        return Interval(
            -lower_bound.constant,
            upper_bound.constant,
            infimum_included=not lower_bound.is_strict,
            supremum_included=not upper_bound.is_strict,
        )

    def _constrain(self, difference: Difference, by: Bound) -> None:
        if difference.left not in self._clocks or difference.right not in self._clocks:
            raise InvalidClockError(
                f"cannot constrain {self} with {difference}: unknown clocks"
            )
        if by < self._matrix.get(difference, _INFINITY_BOUND):
            self._matrix[difference] = by

    def copy(self) -> DBM:
        return DBM(_clocks=self._clocks, _matrix=self._matrix.copy())

    def reset(self, clock: Clock, value: NumberType = 0) -> None:
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
    def constrain(self, *constraints: Constraint) -> None:
        pass

    @t.overload
    def constrain(self, *, difference: Difference, by: Bound) -> None:
        pass

    def constrain(
        self,
        *constraints: Constraint,
        difference: t.Optional[Difference] = None,
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

    def intersect(self, other: DBM) -> None:
        assert other._clocks <= self._clocks
        try:
            for difference, bound in other._matrix.items():
                self._constrain(difference, bound)
        finally:
            self._canonicalize()


def print_constraints(dbm: AbstractDBM) -> None:
    for constraint in dbm.constraints:
        print(constraint)


def intersect(left: DBM, right: DBM) -> DBM:
    result = left.copy()
    result.intersect(right)
    return result


@dataclasses.dataclass
class DBMContext:
    clocks: t.AbstractSet[Clock]

    _clock_map: t.Dict[str, Clock] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for clock in self.clocks:
            assert clock.name not in self._clock_map
            self._clock_map[clock.name] = clock

    def get_clock_by_name(self, name: str) -> Clock:
        return self._clock_map[name]

    def create_unconstrained(self) -> DBM:
        return DBM.create_unconstrained(self.clocks)

    def create_zero(self) -> DBM:
        return DBM.create_zero(self.clocks)
