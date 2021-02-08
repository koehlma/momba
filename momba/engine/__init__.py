# -*- coding:utf-8 -*-
#
# Copyright (C) 2020-2021, Saarland University
# Copyright (C) 2020-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

from .explore import Action, Destination, Transition, State, Explorer
from .time import TimeType, DiscreteTime
from .values import Value


from . import _ipython  # noqa:


__all__ = [
    "Action",
    "Destination",
    "Transition",
    "State",
    "Explorer",
    "TimeType",
    "DiscreteTime",
    "Value",
]

# import dataclasses as d
# import typing as t

# import functools

# from momba_engine import momba_engine as _engine

# from . import translator

# from .. import model


# @d.dataclass(frozen=True)
# class Action:
#     action_type: model.ActionType
#     arguments: t.Tuple[t.Union[int, float, bool], ...]


# @d.dataclass(frozen=True, repr=False)
# class MDPDestination:
#     _explorer: MDPExplorer
#     _state: t.Any
#     _transition: t.Any
#     _destination: t.Any

#     @property
#     def probability(self) -> float:
#         return self._destination.probability()

#     @functools.cached_property
#     def successor(self) -> MDPState:
#         return MDPState(
#             self._explorer,
#             self._destination.successor(self._state, self._transition),
#         )


# @d.dataclass(frozen=True, repr=False)
# class MDPTransition:
#     _explorer: MDPExplorer
#     _state: t.Any
#     _transition: t.Any

#     @functools.cached_property
#     def destinations(self) -> t.Sequence[MDPDestination]:
#         return tuple(
#             MDPDestination(self._explorer, self._state, self._transition, transition)
#             for transition in self._transition.destinations(self._state)
#         )

#     @functools.cached_property
#     def action(self) -> t.Optional[Action]:
#         rust_action = self._transition.result_action()
#         if rust_action.is_silent():
#             return None
#         label = rust_action.label()
#         assert isinstance(label, str)
#         arguments = rust_action.arguments()
#         return Action(
#             self._explorer.network.ctx.get_action_type_by_name(label),
#             tuple(arguments),
#         )


# @d.dataclass(frozen=True, repr=False)
# class MDPState:
#     _explorer: MDPExplorer
#     _state: t.Any

#     @functools.cached_property
#     def transitions(self) -> t.Sequence[MDPTransition]:
#         return tuple(
#             MDPTransition(self._explorer, self._state, transition)
#             for transition in self._state.transitions()
#         )

#     @functools.cached_property
#     def global_env(self) -> t.Mapping[str, t.Any]:
#         return {
#             name: self._state.get_global_value(declaration.identifier)
#             for name, declaration in self._explorer.translation.declarations.globals_table.items()
#         }

#     def get_local_env(self, instance: model.Instance) -> t.Mapping[str, t.Any]:
#         return {
#             name: self._state.get_global_value(declaration.identifier)
#             for name, declaration in self._explorer.translation.declarations.locals_table[
#                 instance
#             ].items()
#         }

#     @functools.cached_property
#     def locations(self) -> t.Mapping[model.Instance, model.Location]:
#         return {
#             instance: self._explorer.translation.reversed_instance_to_location_names[
#                 instance
#             ][self._state.get_location_of(name)]
#             for instance, name in self._explorer.translation.instance_names.items()
#         }


# @d.dataclass(frozen=True, repr=False)
# class MDPExplorer:
#     translation: translator.Translation
#     network: model.Network
#     _compiled: t.Any

#     @functools.cached_property
#     def initial_states(self) -> t.Sequence[MDPState]:
#         return tuple(MDPState(self, state) for state in self._compiled.initial_states())


# def compile_mdp(network: model.Network) -> MDPExplorer:
#     translation = translator.translate_network(network)
#     return MDPExplorer(
#         translation, network, _engine.MDPExplorer(translation.json_network)  # type: ignore
#     )
