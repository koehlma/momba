# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import abc


StateVector = t.Sequence[float]
AvailableVector = t.Sequence[bool]


@d.dataclass(frozen=True)
class Destination:
    state: StateVector
    reward: float
    probability: float


@d.dataclass(frozen=True)
class Transition:
    action: int
    destinations: t.Sequence[Destination]


class Explorer(abc.ABC):
    """State space explorer for training decision agents."""

    @property
    @abc.abstractmethod
    def num_actions(self) -> int:
        """The number of *actions*."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        """The number of features of the state vector."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def has_terminated(self) -> bool:
        """Indicates whether the explorer is in a terminal state."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def state_vector(self) -> StateVector:
        """The state vector of the current explorer state."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def available_actions(self) -> AvailableVector:
        """A boolean vector indicating which actions are available."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def available_transitions(self) -> t.Sequence[Transition]:
        """A sequence of available transitions."""
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action: int) -> float:
        """
        Takes a step with the given action and returns the reward.

        Precondition: The explorer must not be in a terminal state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the explorer to the initial state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def fork(self) -> Explorer:
        """Forks the explorer with the current state."""
        raise NotImplementedError()


class Oracle(t.Protocol):
    """An *oracle* selects an action based on the state and the available actions."""

    def __call__(self, state: StateVector, available: AvailableVector) -> int:
        pass
