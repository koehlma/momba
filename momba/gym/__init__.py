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

from . import abstract, env, checker, generic


def create_generic_env(
    network: model.Network,
    controlled_instance: model.Instance,
    property_name: str,
    *,
    parameters: engine.Parameters = None,
    rewards: generic.Rewards = generic.DEFAULT_REWARD_STRUCTURE,
    actions: generic.Actions = generic.Actions.EDGE_BY_INDEX,
    observations: generic.Observations = generic.Observations.GLOBAL_ONLY,
    renderer: t.Optional[env.Renderer] = None
) -> env.MombaEnv:
    """Convenience function for constructing a generic environment from a model."""
    return env.MombaEnv(
        generic.GenericExplorer.create(
            engine.Explorer.new_discrete_time(network, parameters=parameters),
            controlled_instance,
            property_name,
            rewards=rewards,
            actions=actions,
            observations=observations,
        ),
        renderer=renderer,
    )


__all__ = ["abstract", "env", "checker", "create_generic_env"]
