# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc
import fractions

from .. import model


Properties = t.Mapping[str, model.Expression]
Result = t.Mapping[str, fractions.Fraction]


class Checker(abc.ABC):
    """An abstract class for model checkers."""

    @property
    def description(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def check(
        self,
        network: model.Network,
        *,
        properties: t.Optional[Properties] = None,
        property_names: t.Optional[t.Iterable[str]] = None,
    ) -> Result:
        """Model checks the given properties on the network."""
        raise NotImplementedError()


class CrossCheckError(Exception):
    pass


class PropertyMismatchError(Exception):
    pass


class DeltaExceededError(Exception):
    pass


@d.dataclass(frozen=True, eq=False)
class CrossChecker(Checker):
    checkers: t.Sequence[Checker]

    allowed_delta: fractions.Fraction = fractions.Fraction("1e-4")

    @property
    def description(self) -> str:
        return f"CrossChecker (Δ = {float(self.allowed_delta)})"

    def check(
        self,
        network: model.Network,
        *,
        properties: t.Optional[Properties] = None,
        property_names: t.Optional[t.Iterable[str]] = None,
    ) -> Result:
        assert self.checkers, "checkers for cross-checking must be non-empty"
        results = [
            checker.check(network, properties=properties, property_names=property_names)
            for checker in self.checkers
        ]
        names = frozenset(results[0].keys())
        for result in results:
            if names != result.keys():
                raise PropertyMismatchError(
                    "checkers did not provide results for the same properties"
                )
        for first in range(len(self.checkers)):
            for second in range(first + 1, len(self.checkers)):
                for name in names:
                    first_value = results[first][name]
                    second_value = results[second][name]
                    if abs(first_value - second_value) > self.allowed_delta:
                        raise DeltaExceededError(
                            f"allowed delta of {self.allowed_delta} has been exceeded for {name}"
                        )
        return results[0]
