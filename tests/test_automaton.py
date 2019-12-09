# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

from momba.model.automata import Automaton, Edge, Location, Destination
from momba.model.expressions import var
from momba.model import context, errors, types

import pytest


def test_basic_inferences():
    automaton = Automaton(context.Context())
    automaton.scope.declare_constant('T', types.INT)
    automaton.scope.declare_variable('x', types.CLOCK)

    location_1 = Location('Location1', progress_invariant=var('x').lt(var('T')))
    location_2 = Location('Location2')

    edge = Edge(
        location_1,
        guard=var('x').gt(var('T')),
        destinations=frozenset({
            Destination(location_2)
        })
    )

    automaton.add_edge(edge)

    with pytest.raises(errors.ModelingError):
        automaton.add_location(
            Location('InvalidInvariant', progress_invariant=var('z'))
        )
