# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import abc
import functools

from ._engine import zones as _zones


ConstantT = t.TypeVar("ConstantT", bound=t.Union[int, float])


@functools.total_ordering
class Bound(t.Generic[ConstantT]):
    _bound: t.Any

    def __init__(self, is_strict: bool, constant: ConstantT) -> None:
        self._bound = _zones.Bound(is_strict, constant)

    def __repr__(self) -> str:
        return f"Bound(is_strict={self.is_strict!r}, constant={self.constant!r})"

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, Bound):
            return self.is_strict == other.is_strict and self.constant == other.constant
        else:
            return False

    def __hash__(self) -> int:
        return hash((Bound, self.is_strict, self.constant))

    def __lt__(self, other: t.Any) -> bool:
        if isinstance(other, Bound):
            if self.constant < other.constant:
                return True
            elif self.constant == other.constant:
                return self.is_strict and not other.is_strict
            else:
                return False
        else:
            return NotImplemented

    @property
    def is_strict(self) -> bool:
        return self._bound.is_strict

    @property
    def constant(self) -> ConstantT:
        return self._bound.constant


def _wrap_bound(bound: t.Any) -> Bound[ConstantT]:
    wrapped = Bound.__new__(Bound)
    wrapped._bound = bound
    return wrapped


class Constraint(t.Generic[ConstantT]):
    _constraint: t.Any

    def __init__(self, left: int, right: int, bound: Bound[ConstantT]) -> None:
        self._constraint = _zones.Constraint(left, right, bound._bound)

    def __repr__(self) -> str:
        return f"Constraint(left={self.left!r}, right={self.right!r}, bound={self.bound!r})"

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, Constraint):
            return (
                self.left == other.left
                and self.right == other.right
                and self.bound == other.bound
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((Constraint, self.left, self.right, self.bound))

    @property
    def left(self) -> int:
        return self._constraint.left

    @property
    def right(self) -> int:
        return self._constraint.right

    @property
    def bound(self) -> Bound[ConstantT]:
        return _wrap_bound(self._constraint.bound)


def _wrap_constraint(constraint: t.Any) -> Constraint[ConstantT]:
    wrapped = Constraint.__new__(Constraint)
    wrapped._constraint = constraint
    return wrapped


class ReadOnlyZone(t.Protocol[ConstantT]):
    _zone: t.Any

    @property
    def num_variables(self) -> int:
        pass

    @property
    def num_clocks(self) -> int:
        pass

    @property
    def is_empty(self) -> bool:
        pass

    @property
    def constraints(self) -> t.Iterator[Constraint[ConstantT]]:
        pass

    def get_constraint(self, left: int, right: int) -> Constraint[ConstantT]:
        pass

    def get_bound(self, left: int, right: int) -> Bound[ConstantT]:
        pass

    def is_unbounded(self, clock: int) -> bool:
        pass

    def get_upper_bound(self, clock: int) -> t.Optional[ConstantT]:
        pass

    def get_lower_bound(self, clock: int) -> t.Optional[ConstantT]:
        pass

    def is_satisfied(self, constraint: Constraint[ConstantT]) -> bool:
        pass

    def includes(self, other: ReadOnlyZone[ConstantT]) -> bool:
        pass


class Zone(abc.ABC, ReadOnlyZone[ConstantT], t.Generic[ConstantT]):
    _zone: t.Any

    def __init__(self, _zone: t.Any) -> None:
        self._zone = _zone

    @property
    def num_variables(self) -> int:
        return self._zone.num_variables

    @property
    def num_clocks(self) -> int:
        return self._zone.num_clocks

    @property
    def is_empty(self) -> bool:
        return self._zone.is_empty

    @property
    def constraints(self) -> t.Iterator[Constraint[ConstantT]]:
        for left in range(self.num_clocks):
            for right in range(self.num_clocks):
                constraint = self.get_constraint(left, right)
                if constraint.bound.constant is not None:
                    yield constraint

    def get_constraint(self, left: int, right: int) -> Constraint[ConstantT]:
        return _wrap_constraint(self._zone.get_constraint(left, right))

    def get_bound(self, left: int, right: int) -> Bound[ConstantT]:
        return _wrap_bound(self._zone.get_bound(left, right))

    def constrain(self, constraint: Constraint[ConstantT]) -> None:
        self._zone.add_constraint(constraint._constraint)

    def intersect(self, other: Zone[ConstantT]) -> None:
        self._zone.intersect(other._zone)

    def future(self) -> None:
        self._zone.future()

    def past(self) -> None:
        self._zone.past()

    def reset(self, clock: int, value: ConstantT) -> None:
        self._zone.reset(clock, value)

    def is_unbounded(self, clock: int) -> bool:
        return self._zone.is_unbounded(clock)

    def get_upper_bound(self, clock: int) -> t.Optional[ConstantT]:
        return self._zone.get_upper_bound(clock)

    def get_lower_bound(self, clock: int) -> t.Optional[ConstantT]:
        return self._zone.get_lower_bound(clock)

    def is_satisfied(self, constraint: Constraint[ConstantT]) -> bool:
        return self._zone.is_satisfied(constraint._constraint)

    def includes(self, other: ReadOnlyZone[ConstantT]) -> bool:
        return self._zone.includes(other._zone)

    def resize(self, num_variables: int) -> Zone[ConstantT]:
        wrapped = Zone.__new__(Zone)
        wrapped._zone = self._zone.resize(num_variables)
        return wrapped


ZoneT = t.TypeVar("ZoneT", bound=Zone[t.Union[float, int]])


def _wrap_zone(zone: t.Any, cls: t.Type[ZoneT]) -> ZoneT:
    wrapped = cls.__new__(cls)
    wrapped._zone = zone
    return wrapped


class ZoneI64(Zone[int]):
    def __init__(self, num_variables: int) -> None:
        super().__init__(_zones.Zone.new_i64_unconstrained(num_variables))

    @classmethod
    def new_unconstrained(cls, num_variables: int) -> ZoneI64:
        zone = cls.__new__(cls)
        cls._zone = _zones.Zone.new_i64_unconstrained(num_variables)
        return zone

    @classmethod
    def new_zero(cls, num_variables: int) -> ZoneI64:
        zone = cls.__new__(cls)
        cls._zone = _zones.Zone.new_i64_zero(num_variables)
        return zone


class ZoneF64(Zone[float]):
    def __init__(self, num_variables: int) -> None:
        super().__init__(_zones.Zone.new_f64_unconstrained(num_variables))

    @classmethod
    def new_unconstrained(cls, num_variables: int) -> ZoneF64:
        zone = cls.__new__(cls)
        cls._zone = _zones.Zone.new_f64_unconstrained(num_variables)
        return zone

    @classmethod
    def new_zero(cls, num_variables: int) -> ZoneF64:
        zone = cls.__new__(cls)
        cls._zone = _zones.Zone.new_f64_zero(num_variables)
        return zone
