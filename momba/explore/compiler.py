# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Compiles a Momba model to MombaCR.
"""

from __future__ import annotations

import dataclasses as d
from momba.model.actions import ActionArgument, GuardArgument
import typing as t

import functools
import json

from .. import model
from ..model import effects, expressions, operators


_JSONObject = t.Dict[str, t.Any]


@functools.singledispatch
def _compile_type(typ: model.Type) -> t.Mapping[str, t.Any]:
    raise NotImplementedError(f"'_compile_type' not implemented for {type(typ)}")


@_compile_type.register
def _compile_integer_type(typ: model.types.IntegerType) -> t.Mapping[str, t.Any]:
    return {"type": "INT64"}


@_compile_type.register
def _compile_real_type(typ: model.types.RealType) -> t.Mapping[str, t.Any]:
    return {"type": "FLOAT64"}


@_compile_type.register
def _compile_bounded_type(typ: model.types.BoundedType) -> t.Mapping[str, t.Any]:
    return _compile_type(typ.base)


@_compile_type.register
def _compile_bool_type(typ: model.types.BoolType) -> t.Mapping[str, t.Any]:
    return {"type": "BOOL"}


@_compile_type.register
def _compile_array_type(typ: model.types.ArrayType) -> t.Mapping[str, t.Any]:
    return {"type": "VECTOR", "element_type": _compile_type(typ.element)}


@d.dataclass
class _CompilationContext:
    scope: model.Scope
    names: t.Dict[str, str] = d.field(default_factory=dict)


@functools.singledispatch
def _compile_expr(expr: model.Expression, ctx: _CompilationContext) -> _JSONObject:
    raise NotImplementedError(f"'_compile_expr' not implemented for {type(expr)}")


@_compile_expr.register
def _compile_not_expr(expr: expressions.Not, ctx: _CompilationContext) -> _JSONObject:
    return {
        "kind": "UNARY",
        "operator": "NOT",
        "operand": _compile_expr(expr.operand, ctx),
    }


@_compile_expr.register
def _compile_boolean_expr(
    expr: expressions.Boolean, ctx: _CompilationContext
) -> _JSONObject:
    operator = expr.operator
    operands: t.List[_JSONObject] = []
    stack = [expr.right, expr.left]
    while stack:
        top = stack.pop()
        if isinstance(top, expressions.Boolean) and top.operator == operator:
            stack.append(top.right)
            stack.append(top.left)
        elif isinstance(top, expressions.BooleanConstant):
            if operator is operators.BooleanOperator.AND:
                if not top.boolean:
                    return {"kind": "CONSTANT", "value": False}
            else:
                assert operator is operators.BooleanOperator.OR
                if top.boolean:
                    return {"kind": "CONSTANT", "value": True}
        else:
            operands.append(_compile_expr(top, ctx))
    return {"kind": "BOOLEAN", "operator": operator.name, "operands": operands}


@_compile_expr.register
def _compile_comparison_expr(
    expr: expressions.Comparison, ctx: _CompilationContext
) -> _JSONObject:
    return {
        "kind": "COMPARISON",
        "operator": expr.operator.name,
        "left": _compile_expr(expr.left, ctx),
        "right": _compile_expr(expr.right, ctx),
    }


_EQUALITY_OPERATOR_MAP = {
    operators.EqualityOperator.EQ: "EQ",
    operators.EqualityOperator.NEQ: "NE",
}


@_compile_expr.register
def _compile_equality_expr(
    expr: expressions.Equality, ctx: _CompilationContext
) -> _JSONObject:
    return {
        "kind": "COMPARISON",
        "operator": _EQUALITY_OPERATOR_MAP[expr.operator],
        "left": _compile_expr(expr.left, ctx),
        "right": _compile_expr(expr.right, ctx),
    }


@_compile_expr.register
def _compile_name_expr(expr: expressions.Name, ctx: _CompilationContext) -> _JSONObject:
    declaration = ctx.scope.lookup(expr.identifier)
    if isinstance(declaration, model.ConstantDeclaration):
        assert declaration.value is not None
        return _compile_expr(declaration.value, ctx)
    return {
        "kind": "NAME",
        "identifier": ctx.names.get(expr.identifier, expr.identifier),
    }


@_compile_expr.register
def _compile_arithmetic_binary_expr(
    expr: expressions.ArithmeticBinary, ctx: _CompilationContext
) -> _JSONObject:
    return {
        "kind": "BINARY",
        "operator": expr.operator.name,
        "left": _compile_expr(expr.left, ctx),
        "right": _compile_expr(expr.right, ctx),
    }


@_compile_expr.register
def _compile_integer_constant_expr(
    expr: expressions.IntegerConstant, ctx: _CompilationContext
) -> _JSONObject:
    return {"kind": "CONSTANT", "value": expr.integer}


@_compile_expr.register
def _compile_arithmetic_unary_expr(
    expr: expressions.ArithmeticUnary, ctx: _CompilationContext
) -> _JSONObject:
    return {
        "kind": "UNARY",
        "operator": expr.operator.name,
        "operand": _compile_expr(expr.operand, ctx),
    }


def _populate_variables(
    variables: t.Dict[str, t.Any],
    prefix: str,
    declarations: t.Iterable[model.VariableDeclaration],
) -> None:
    for declaration in declarations:
        variables[f"{prefix}{declaration.identifier}"] = _compile_type(declaration.typ)


def compile_network(network: model.Network) -> str:
    assert network.ctx.model_type is model.ModelType.MDP

    variables: t.Dict[str, t.Any] = {}
    _populate_variables(
        variables, "global_", network.ctx.global_scope.variable_declarations
    )

    global_names: t.Dict[str, str] = {}

    initial_values: _JSONObject = {}

    for declaration in network.ctx.global_scope.variable_declarations:
        identifier = f"global_{declaration.identifier}"
        global_names[declaration.identifier] = identifier
        assert declaration.initial_value
        if isinstance(declaration.initial_value, expressions.BooleanConstant):
            initial_values[identifier] = declaration.initial_value.boolean
        elif isinstance(declaration.initial_value, expressions.IntegerConstant):
            initial_values[identifier] = declaration.initial_value.integer
        else:
            raise NotImplementedError("invalid initial value")

    instance_indices: t.Dict[model.Automaton, t.Dict[model.Instance, int]] = {}
    for instance in network.instances:
        try:
            instance_index = len(instance_indices[instance.automaton])
        except KeyError:
            instance_indices[instance.automaton] = {}
            instance_index = 0
        instance_indices[instance.automaton][instance] = instance_index
        prefix = f"local_{instance.automaton.name}{instance_index}_"
        _populate_variables(
            variables, prefix, instance.automaton.scope.variable_declarations
        )

    actions: _JSONObject = {}
    for action_type in network.ctx.action_types.values():
        actions[action_type.name] = list(
            _compile_type(parameter.typ) for parameter in action_type.parameters
        )

    clocks: t.List[str] = []  # TODO: construct a set of clocks

    automata: _JSONObject = {}

    initial_locations: _JSONObject = {}

    for instance in network.instances:
        index = instance_indices[instance.automaton][instance]
        name = f"{instance.automaton.name}{index}"

        locations: _JSONObject = {}
        automaton: _JSONObject = {"locations": locations}

        location_names: t.Dict[model.Location, str] = {}

        names = dict(global_names)

        for declaration in instance.automaton.scope.variable_declarations:
            names[
                declaration.identifier
            ] = f"local_{instance.automaton.name}{index}_{declaration.identifier}"

        for index, location in enumerate(instance.automaton.locations):
            location_name = f"{index}_{location.name or ''}"
            location_names[location] = location_name
            locations[location_name] = {"invariant": [], "edges": []}
            if location in instance.automaton.initial_locations:
                assert name not in initial_locations
                initial_locations[name] = location_name

        compilation_ctx = _CompilationContext(instance.automaton.scope, names)

        for edge in instance.automaton.edges:
            outgoing = locations[location_names[edge.location]]["edges"]
            action: _JSONObject = {}
            if edge.action_pattern is None:
                action["kind"] = "INTERNAL"
            else:
                action["kind"] = "PATTERN"
                action["name"] = edge.action_pattern.action_type.name
                assert not edge.action_pattern.arguments
                action["arguments"] = []
            guard: _JSONObject
            if edge.guard is None:
                guard = {"kind": "CONSTANT", "value": True}
            else:
                guard = _compile_expr(edge.guard, compilation_ctx)

            destinations: t.List[_JSONObject] = []

            for destination in edge.destinations:
                probability: _JSONObject
                if destination.probability is None:
                    probability = {"kind": "CONSTANT", "value": 1.0}
                else:
                    probability = _compile_expr(
                        destination.probability, compilation_ctx
                    )

                assignments: t.List[_JSONObject] = []

                for assignment in destination.assignments:
                    assert isinstance(assignment.target, effects.Name)
                    assignments.append(
                        {
                            "target": {
                                "kind": "NAME",
                                "identifier": f"global_{assignment.target.identifier}",
                            },
                            "value": _compile_expr(assignment.value, compilation_ctx),
                            "index": assignment.index,
                        }
                    )

                destinations.append(
                    {
                        "location": location_names[destination.location],
                        "probability": probability,
                        "assignments": assignments,
                    }
                )

            outgoing.append(
                {
                    "action": action,
                    "guard": guard,
                    "destinations": destinations,
                }
            )

        automata[name] = automaton

    initial: t.List[_JSONObject] = [
        {"values": initial_values, "locations": initial_locations}
    ]

    links: t.List[_JSONObject] = []

    for link in network.links:
        slots: t.Set[str] = set()

        def _compile_arguments(arguments: t.Iterable[ActionArgument]) -> t.List[str]:
            slot_vector: t.List[str] = []
            for argument in arguments:
                assert isinstance(argument, GuardArgument)
                slots.add(argument.identifier)
                slot_vector.append(argument.identifier)
            return slot_vector

        assert link.condition is None
        result: _JSONObject
        if link.result is None:
            result = {"kind": "INTERNAL"}
        else:
            result = {
                "kind": "PATTERN",
                "name": link.result.action_type.name,
                "arguments": _compile_arguments(link.result.arguments),
            }
        vector: _JSONObject = {}
        for instance, pattern in link.vector.items():
            index = instance_indices[instance.automaton][instance]
            name = f"{instance.automaton.name}{index}"
            vector[name] = {
                "name": pattern.action_type.name,
                "arguments": _compile_arguments(pattern.arguments),
            }

        links.append({"slots": list(slots), "vector": vector, "result": result})

    return json.dumps(
        {
            "variables": variables,
            "clocks": clocks,
            "actions": actions,
            "automata": automata,
            "initial": initial,
            "links": links,
        },
        indent=2,
    )
