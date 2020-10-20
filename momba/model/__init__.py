# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Data-structures for the representation of quantitative models.

Automata and automata networks are mutable. This is an intentional design
choice. The validity of the model is partly assured during construction,
for instance, a location can only be added if the invariant is a boolean
expression in the scope of the automaton. As usual, if an algorithm works
on a model, e.g., analyzing it, the model must not be modified.
"""

from __future__ import annotations

from . import effects, errors, expressions, operators, properties, types

from .actions import (
    ActionType,
    ActionPattern,
    ActionParameter,
    ActionArgument,
    ReadArgument,
    WriteArgument,
    GuardArgument,
)

from .automata import (
    Instance,
    Automaton,
    Location,
    Destination,
    Edge,
    create_destination,
)

from .context import (
    ModelType,
    Declaration,
    VariableDeclaration,
    ConstantDeclaration,
    PropertyDefinition,
    Scope,
    Context,
)

from .distributions import DistributionType

from .expressions import Expression

from .networks import Link, Network

from .properties import Property

from .types import Type


__all__ = [
    "effects",
    "errors",
    "expressions",
    "operators",
    "properties",
    "types",
    # from `.actions`
    "ActionType",
    "ActionPattern",
    "ActionParameter",
    "ActionArgument",
    "ReadArgument",
    "WriteArgument",
    "GuardArgument",
    # from `.automata`
    "Instance",
    "Automaton",
    "Location",
    "Destination",
    "Edge",
    "create_destination",
    # from `.context`
    "ModelType",
    "Declaration",
    "VariableDeclaration",
    "ConstantDeclaration",
    "PropertyDefinition",
    "Scope",
    "Context",
    # from `.distributions`
    "DistributionType",
    # from `.expressions`
    "Expression",
    # from `.networks`
    "Network",
    "Link",
    # from `.property`
    "Property",
    # from `.types`
    "Type",
]
