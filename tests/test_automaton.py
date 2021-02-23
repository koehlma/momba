# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

from momba import model
from momba.model.automata import Automaton, Location, Destination
from momba.model.expressions import name, less, greater
from momba.model import context, errors, types

import pytest


def test_basic_inferences() -> None:
    automaton = Automaton(context.Context(model.ModelType.SHA))
    automaton.scope.declare_constant("T", types.INT)
    automaton.scope.declare_variable("x", types.CLOCK)

    location_1 = automaton.create_location(
        "Location1", progress_invariant=less(name("x"), name("T"))
    )
    location_2 = automaton.create_location("Location2")

    automaton.create_edge(
        location_1,
        guard=greater(name("x"), name("T")),
        destinations=frozenset({Destination(location_2)}),
    )

    with pytest.raises(errors.ModelingError):
        automaton.add_location(
            Location("InvalidInvariant", progress_invariant=name("z"))
        )
