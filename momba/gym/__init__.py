# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>
#
# type: ignore

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
    renderer: t.Optional[env.Renderer] = None,
) -> env.MombaEnv:
    """
    Constructs a generic training environment from a JANI model based on the provided options.

    Arguments:
        network: A JANI automaton network.
        controlled_instance: An instance of an automaton in the provided network. The
            decision-making agent trained on the resulting environment is assumed to act
            by resolving the non-determinism in this automaton.
        property_name: The name of a reach-avoid JANI property (specified as part of the
            JANI model the network originates from) for which the agent should be trained.
        parameters: Allows defining values for parameters of the JANI model.
        rewards: Specifies the reward structure used for training.
        actions: Specifies the action space for the environment.
        observations: Specifies the observation space for the environment.
        renderer: Is an optional renderer for the OpenAI Gym API.
    """
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
