# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import fractions
import random

from mxu.maps import FrozenMap


ElementT = t.TypeVar("ElementT", bound=t.Hashable)


class Distribution(t.Generic[ElementT]):
    """A probability distribution."""

    _mapping: FrozenMap[ElementT, fractions.Fraction]

    @classmethod
    def create_dirac(cls, element: ElementT) -> Distribution[ElementT]:
        return cls({element: 1})

    @classmethod
    def create_uniform(cls, *elements: ElementT) -> Distribution[ElementT]:
        probability = fractions.Fraction(1, len(elements))
        return cls({element: probability for element in elements})

    def __init__(
        self, mapping: t.Mapping[ElementT, t.Union[int, float, fractions.Fraction]]
    ) -> None:
        self._mapping = FrozenMap.transfer_ownership(
            {
                element: fractions.Fraction(probability)
                for element, probability in mapping.items()
            }
        )
        assert all(probability >= 0 for probability in self._mapping.values())
        # assert sum(self._mapping.values()) == 1

    def __str__(self) -> str:
        return f"Distribution({self._mapping})"

    @property
    def support(self) -> t.List[ElementT]:
        return list(
            element for element, probability in self._mapping.items() if probability > 0
        )

    @property
    def is_dirac(self) -> bool:
        return len(self.support) == 1

    def get_probability(self, element: ElementT) -> fractions.Fraction:
        return self._mapping.get(element, fractions.Fraction(0))

    def pick(self) -> ElementT:
        """Picks an element at random according to the distribution."""
        max_denominator = max(
            probability.denominator for probability in self._mapping.values()
        )
        outcome = fractions.Fraction(
            random.randint(0, max_denominator), max_denominator
        )
        total = fractions.Fraction(0)
        for element in self.support:
            total += self.get_probability(element)
            if outcome <= total:
                return element
        raise RuntimeError("empty distribution is not possible")
