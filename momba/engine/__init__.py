# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Interfaces to `momba_engine` hiding away complexity.
"""

from __future__ import annotations

import dataclasses as d
import typing as t

import functools

from . import translator

from momba_engine import momba_engine as _engine

from .. import model


@d.dataclass(frozen=True, repr=False)
class MDPDestination:
    _state: t.Any
    _transition: t.Any
    _destination: t.Any

    @property
    def probability(self) -> float:
        return self._destination.probability()


@d.dataclass(frozen=True, repr=False)
class MDPTransition:
    _state: t.Any
    _transition: t.Any

    @functools.cached_property
    def destinations(self) -> t.Sequence[MDPDestination]:
        return tuple(
            MDPDestination(self._state, self._transition, transition)
            for transition in self._transition.destinations(self._state)
        )


@d.dataclass(frozen=True, repr=False)
class MDPState:
    _translation: translator.Translation
    _state: t.Any

    @functools.cached_property
    def transitions(self) -> t.Sequence[MDPTransition]:
        return tuple(
            MDPTransition(self._state, transition)
            for transition in self._state.transitions()
        )

    @functools.cached_property
    def global_env(self) -> t.Mapping[str, t.Any]:
        return {
            name: self._state.get_global_value(declaration.identifier)
            for name, declaration in self._translation.declarations.globals_table.items()
        }

    def get_local_env(self, instance: model.Instance) -> t.Mapping[str, t.Any]:
        return {
            name: self._state.get_global_value(declaration.identifier)
            for name, declaration in self._translation.declarations.locals_table[
                instance
            ].items()
        }

    @functools.cached_property
    def locations(self) -> t.Mapping[model.Instance, model.Location]:
        return {
            instance: self._translation.reversed_instance_to_location_names[instance][
                self._state.get_location_of(name)
            ]
            for instance, name in self._translation.instance_names.items()
        }


@d.dataclass(frozen=True, repr=False)
class MDPExplorer:
    translation: translator.Translation
    _compiled: t.Any

    @functools.cached_property
    def initial_states(self) -> t.Sequence[MDPState]:
        return tuple(
            MDPState(self.translation, state)
            for state in self._compiled.initial_states()
        )


def compile_mdp(network: model.Network) -> MDPExplorer:
    translation = translator.translate_network(network)
    return MDPExplorer(
        translation, _engine.MDPExplorer(translation.json_network)  # type: ignore
    )
