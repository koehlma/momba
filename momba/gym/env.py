# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import gym
import numpy

from gym import spaces
from gym.error import UnsupportedMode

from . import abstract


class Renderer(t.Protocol):
    def render(self, state: abstract.StateVector, mode: str) -> None:
        raise NotImplementedError()


class MombaEnv(gym.Env):  # type: ignore
    """Implementation of an OpenAI Gym environment."""

    explorer: abstract.Explorer

    action_space: gym.Space  # type: ignore
    observation_space: gym.Space  # type: ignore

    renderer: t.Optional[Renderer]

    def __init__(
        self, explorer: abstract.Explorer, *, renderer: t.Optional[Renderer] = None
    ) -> None:
        super().__init__()
        self.explorer = explorer
        self.action_space = spaces.Discrete(self.explorer.num_actions)
        self.observation_space = spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(self.explorer.num_features,)
        )
        self.renderer = renderer

    @property
    def available_actions(self) -> numpy.ndarray:  # type: ignore
        return numpy.array(self.explorer.available_actions)

    @property
    def available_transitions(self) -> t.Sequence[abstract.Transition]:
        return self.explorer.available_transitions

    @property
    def is_done(self) -> bool:
        return self.explorer.has_terminated

    @property
    def state_vector(self) -> numpy.ndarray:  # type: ignore
        return numpy.array(self.explorer.state_vector)

    def fork(self) -> MombaEnv:
        """Forks the environment."""
        return MombaEnv(self.explorer.fork(), renderer=self.renderer)

    def step(self, action: int) -> t.Tuple[numpy.ndarray, float, bool, t.Any]:  # type: ignore
        """Takes a decision in response to the last observation."""
        reward = self.explorer.step(action)
        state = numpy.array(self.explorer.state_vector)
        return state, reward, self.explorer.has_terminated, {}

    def reset(self) -> numpy.ndarray:  # type: ignore
        """Resets the environment to an initial state and returns an initial observation."""
        self.explorer.reset()
        return numpy.array(self.explorer.state_vector)

    def render(self, mode: str = "human") -> None:
        """Renders the environment assuming a :code:`render` has been supplied."""
        if self.renderer is None:
            raise UnsupportedMode("`MombaGym` does not support rendering")
        else:
            self.renderer.render(self.explorer.state_vector, mode)
