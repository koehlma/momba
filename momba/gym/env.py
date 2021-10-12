# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>
#
# type: ignore

from __future__ import annotations

import typing as t

import gym
import numpy

from gym import spaces
from gym.error import UnsupportedMode

from .api import Explorer, StateVector


class Renderer(t.Protocol):
    def render(self, state: StateVector, mode: str) -> None:
        raise NotImplementedError()


class MombaEnv(gym.Env):
    explorer: Explorer

    action_space: gym.Space
    observation_space: gym.Space

    renderer: t.Optional[Renderer]

    def __init__(
        self, explorer: Explorer, *, renderer: t.Optional[Renderer] = None
    ) -> None:
        super().__init__()
        self.explorer = explorer
        self.action_space = spaces.Discrete(self.explorer.num_actions)
        self.observation_space = spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(self.explorer.num_features,)
        )
        self.renderer = renderer

    @property
    def available_actions(self) -> numpy.ndarray:
        return numpy.array(self.explorer.available_actions)

    def fork(self) -> MombaEnv:
        return MombaEnv(self.explorer.fork(), renderer=self.renderer)

    def step(self, action: int) -> t.Tuple[numpy.ndarray, float, bool, dict]:
        reward = self.explorer.step(action)
        state = numpy.array(self.explorer.state_vector)
        return state, reward, self.explorer.has_terminated, {}

    def reset(self) -> numpy.ndarray:
        self.explorer.reset()
        return numpy.array(self.explorer.state_vector)

    def render(self, mode: str = "human"):
        if self.renderer is None:
            raise UnsupportedMode("`MombaGym` does not support rendering")
        else:
            self.renderer.render(self.explorer.state, mode)
