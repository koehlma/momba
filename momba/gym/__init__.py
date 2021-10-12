# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
A formal methods based toolbox for reinforcement learning.
"""

from __future__ import annotations

import typing as t

from .. import model, engine

from . import api, env, checker, generic


def create_generic_env(
    network: model.Network,
    instance: model.Instance,
    property_name: str,
    *,
    parameters: engine.Parameters = None,
    rewards: generic.RewardStructure = generic.DEFAULT_REWARD_STRUCTURE,
    renderer: t.Optional[env.Renderer] = None
) -> env.MombaEnv:
    explorer = engine.Explorer.new_discrete_time(network, parameters=parameters)
    ctx = generic.GenericContext.create(
        explorer, instance, property_name, rewards=rewards
    )
    return env.MombaEnv(generic.GenericExplorer(ctx), renderer=renderer)


__all__ = ["api", "env", "checker", "create_generic_env"]
