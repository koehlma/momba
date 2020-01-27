# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

"""
Data-structures for the representation of quantitative models.

Automata and automata networks are mutable. This is an intentional design
choice. The validity of the model is partly assured during construction,
for instance, a location can only be added if the invariant is a boolean
expression in the scope of the automaton. As usual, if an algorithm works
on a model, e.g., analyzing it, the model must not be modified. In general
models can only be extended with further entities, e.g., locations. The
library does not support removing locations or swapping expressions et
cetera.

A modeling context can be locked for modification.
"""

from __future__ import annotations

from . import expressions, types, distributions

from .action import ActionType, ActionParameter, ActionPattern

from .automata import (
    Automaton,
    Destination,
    Edge,
    Instance,
    Location,
    create_destination,
)
from .context import (
    ConstantDeclaration,
    Context,
    ModelType,
    Scope,
    VariableDeclaration,
)
from .effects import Assignment
from .expressions import (
    BinaryConstructor,
    Expression,
    convert,
    ite,
    logic_not,
    identifier,
)
from .network import Composition, Network, Synchronization
from .properties import Property
from .types import Type


__all__ = [
    "expressions",
    "Property",
    "ActionType",
    "ActionParameter",
    "ActionPattern",
    "Assignment",
    "distributions",
    "Context",
    "ConstantDeclaration",
    "VariableDeclaration",
    "logic_not",
    "ite",
    "convert",
    "identifier",
    "BinaryConstructor",
    "types",
    "values",
    "Automaton",
    "Location",
    "Edge",
    "Destination",
    "create_destination",
    "Instance",
    "Identifier",
    "ModelType",
    "Scope",
    "Expression",
    "Network",
    "Synchronization",
    "Composition",
    "Value",
    "Type",
]
