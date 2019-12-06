# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import json
import typing

from ..model import assignments, automata, context, distributions, expressions, types
from ..model import Model


_TYPE_MAP = {
    'bool': types.BOOL,
    'int': types.INT,
    'real': types.REAL,
    'clock': types.CLOCK,
    'continuous': types.CONTINUOUS
}

_BINARY_OP_MAP: typing.Mapping[str, expressions.BinaryConstructor] = {
    '∨': expressions.lor,
    '∧': expressions.land,
    '=': expressions.eq,
    '≠': expressions.neq,
    '<': expressions.lt,
    '≤': expressions.ge,
    '>': expressions.gt,
    '≥': expressions.ge,
    '+': expressions.add,
    '-': expressions.sub,
    '*': expressions.mul,
    '%': expressions.mod,
    '/': expressions.div,
    'pow': expressions.power,
    'log': expressions.log
}

_UNARY_OP_MAP: typing.Mapping[str, expressions.UnaryConstructor] = {
    '¬': expressions.lnot,
    'floor': expressions.floor,
    'ceil': expressions.ceil
}


def _expression(jani_expression: typing.Any) -> expressions.Expression:
    if isinstance(jani_expression, (float, bool, int)):
        return expressions.cast(jani_expression)
    elif isinstance(jani_expression, str):
        return expressions.var(jani_expression)
    elif isinstance(jani_expression, dict):
        if 'constant' in jani_expression:
            return expressions.cast(jani_expression['constant'])
        elif 'op' in jani_expression:
            op = jani_expression['op']
            if op in _BINARY_OP_MAP:
                left = _expression(jani_expression['left'])
                right = _expression(jani_expression['right'])
                return _BINARY_OP_MAP[op](left, right)
            elif op in _UNARY_OP_MAP:
                operand = _expression(jani_expression['exp'])
                return _UNARY_OP_MAP[op](operand)
            elif op == 'ite':
                condition = _expression(jani_expression['if'])
                consequence = _expression(jani_expression['then'])
                alternative = _expression(jani_expression['else'])
                return expressions.ite(condition, consequence, alternative)
            elif op == 'der':
                variable = jani_expression['var']
                return expressions.Derivative(variable)
        elif 'distribution' in jani_expression:
            arguments = list(map(_expression, jani_expression['args']))
            distribution = distributions.by_name(jani_expression['distribution'])
            return expressions.Sample(distribution, arguments)
    raise ValueError(
        f'{jani_expression} does not seem to be a valid JANI expression'
    )


def _type(typ: typing.Any) -> types.Type:
    if isinstance(typ, str):
        return _TYPE_MAP[typ]
    assert isinstance(typ, dict)
    if typ['kind'] == 'bounded':
        base = _type(typ['base'])
        assert isinstance(base, types.Numeric)
        lower_bound = _expression(typ['lower-bound']) if 'lower-bound' in typ else None
        upper_bound = _expression(typ['upper-bound']) if 'upper-bound' in typ else None
        return base[lower_bound, upper_bound]
    assert False, 'this should never happen'


def _variable_declaration(jani_declaration: typing.Any) -> context.VariableDeclaration:
    initial_value: typing.Optional[expressions.Expression]
    if 'initial-value' in jani_declaration:
        initial_value = _expression(jani_declaration['initial-value'])
    else:
        initial_value = None
    return context.VariableDeclaration(
        identifier=jani_declaration['name'],
        typ=_type(jani_declaration['type']),
        initial_value=initial_value
    )


def _location(jani_location: typing.Any) -> automata.Location:
    assert 'transient-values' not in jani_location, 'unsupported'
    invariant: typing.Optional[expressions.Expression]
    if 'time-progress' in jani_location:
        invariant = _expression(jani_location['time-progress']['exp'])
    else:
        invariant = None
    return automata.Location(name=jani_location['name'], invariant=invariant)


_Locations = typing.Dict[str, automata.Location]


def _target(jani_target: typing.Any) -> assignments.Target:
    if isinstance(jani_target, str):
        return assignments.Identifier(jani_target)
    assert False


def _edge(locations: _Locations, jani_edge: typing.Any) -> automata.Edge:
    location = locations[jani_edge['location']]
    action = jani_edge['action'] if 'action' in jani_edge else None
    rate = _expression(jani_edge['rate']['exp']) if 'rate' in jani_edge else None
    guard = _expression(jani_edge['guard']['exp']) if 'guard' in jani_edge else None
    destinations = frozenset(
        automata.Destination(
            location=locations[jani_destination['location']],
            probability=(
                _expression(jani_destination['probability']['exp'])
                if 'probability' in jani_destination else None
            ),
            assignments=frozenset(
                assignments.Assignment(
                    target=_target(jani_assignment['ref']),
                    value=_expression(jani_assignment['value']),
                    index=jani_assignment.get('index', 0)
                ) for jani_assignment in jani_destination['assignments']
            )
        ) for jani_destination in jani_edge['destinations']
    )
    return automata.Edge(
        location=location,
        destinations=destinations,
        action=action,
        guard=guard,
        rate=rate
    )


def loads(source: str) -> Model:
    jani_model = json.loads(source)
    model = Model()
    if 'variables' in jani_model:
        for jani_declaration in jani_model['variables']:
            model.ctx.global_scope.declare(_variable_declaration(jani_declaration))
    for jani_automaton in jani_model['automata']:
        automaton = model.new_automaton()
        if 'variables' in jani_automaton:
            for jani_declaration in jani_automaton['variables']:
                automaton.scope.declare(_variable_declaration(jani_declaration))
        locations = {
            location.name: location for location in map(_location, jani_automaton['locations'])
        }
        for jani_edge in jani_automaton['edges']:
            automaton.add_edge(_edge(locations, jani_edge))
        for jani_location in jani_automaton['initial-locations']:
            automaton.add_initial_location(locations[jani_location])
    return model
