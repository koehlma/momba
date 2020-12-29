# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

"""
Translates a Model into Momba's intermediate representation for state
space exploration.
"""

from __future__ import annotations

import dataclasses as d
import typing as t

import functools
import json

from .. import model
from ..model import actions, effects, expressions, operators


_JSONObject = t.Dict[str, t.Any]


class CompileError(Exception):
    pass


@functools.singledispatch
def _translate_type(typ: model.Type) -> _JSONObject:
    raise NotImplementedError(f"'_translate_type' not implemented for {type(typ)}")


@_translate_type.register
def _translate_integer_type(typ: model.types.IntegerType) -> _JSONObject:
    return {"type": "INT64"}


@_translate_type.register
def _translate_real_type(typ: model.types.RealType) -> _JSONObject:
    return {"type": "FLOAT64"}


@_translate_type.register
def _translate_bounded_type(typ: model.types.BoundedType) -> _JSONObject:
    return _translate_type(typ.base)


@_translate_type.register
def _translate_bool_type(typ: model.types.BoolType) -> _JSONObject:
    return {"type": "BOOL"}


@_translate_type.register
def _translate_array_type(typ: model.types.ArrayType) -> _JSONObject:
    return {"type": "VECTOR", "element_type": _translate_type(typ.element)}


@d.dataclass(frozen=True)
class VariableDeclaration:
    identifier: str
    typ: model.types.Type
    initial_value: model.Expression


GlobalsTable = t.Mapping[str, VariableDeclaration]
LocalsTable = t.Mapping[model.Instance, t.Mapping[str, VariableDeclaration]]


@d.dataclass
class Declarations:
    globals_table: GlobalsTable = d.field(default_factory=dict)
    locals_table: LocalsTable = d.field(default_factory=dict)


@d.dataclass
class _TranslationContext:
    parameters: t.Mapping[str, model.Expression]
    declarations: Declarations
    scope: model.Scope
    instance: t.Optional[model.Instance] = None

    @property
    def instance_table(self) -> t.Optional[t.Mapping[str, VariableDeclaration]]:
        if self.instance is None:
            return None
        return self.declarations.locals_table[self.instance]


def _translate_identifier(identifier: str, ctx: _TranslationContext) -> _JSONObject:
    # TODO: How do we deal with edge and action scopes?
    try:
        declaration = ctx.scope.lookup(identifier)
        if isinstance(declaration, model.ConstantDeclaration):
            try:
                return _translate_expr(ctx.parameters[declaration.identifier], ctx)
            except KeyError:
                pass
            if declaration.value is None:
                raise CompileError(
                    f"No value for constant {declaration.identifier!r} provided."
                )
            return _translate_expr(declaration.value, ctx)
        instance_table = ctx.instance_table
        if instance_table is not None:
            try:
                return {
                    "kind": "NAME",
                    "identifier": instance_table[identifier].identifier,
                }
            except KeyError:
                pass
        return {
            "kind": "NAME",
            "identifier": ctx.declarations.globals_table[identifier].identifier,
        }
    except KeyError or model.errors.UnboundIdentifierError:
        # We may run in this case due to *anonymous scopes* occuring, for
        # instance, as part of array constructors.
        return {"kind": "NAME", "identifier": identifier}


@functools.singledispatch
def _translate_expr(expr: model.Expression, ctx: _TranslationContext) -> _JSONObject:
    raise NotImplementedError(f"'_translate_expr' not implemented for {type(expr)}")


@_translate_expr.register
def _translate_boolean_constant_expr(
    expr: expressions.BooleanConstant, ctx: _TranslationContext
) -> _JSONObject:
    return {"kind": "CONSTANT", "value": expr.boolean}


@_translate_expr.register
def _translate_integer_constant_expr(
    expr: expressions.IntegerConstant, ctx: _TranslationContext
) -> _JSONObject:
    return {"kind": "CONSTANT", "value": expr.integer}


@_translate_expr.register
def _translate_real_constant_expr(
    expr: expressions.RealConstant, ctx: _TranslationContext
) -> _JSONObject:
    return {"kind": "CONSTANT", "value": expr.as_float}


@_translate_expr.register
def _translate_name_expr(
    expr: expressions.Name, ctx: _TranslationContext
) -> _JSONObject:
    return _translate_identifier(expr.identifier, ctx)


@_translate_expr.register
def _translate_boolean_expr(
    expr: expressions.Boolean, ctx: _TranslationContext
) -> _JSONObject:
    # We flatten trees of boolean expressions. The IR comes with boolean
    # operators taking an arbitrary amount of expressions.
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
            operands.append(_translate_expr(top, ctx))
    return {"kind": "BOOLEAN", "operator": operator.name, "operands": operands}


@_translate_expr.register
def _translate_arithmetic_binary_expr(
    expr: expressions.ArithmeticBinary, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "BINARY",
        "operator": expr.operator.name,
        "left": _translate_expr(expr.left, ctx),
        "right": _translate_expr(expr.right, ctx),
    }


_EQUALITY_OPERATOR_MAP = {
    operators.EqualityOperator.EQ: "EQ",
    operators.EqualityOperator.NEQ: "NE",
}


@_translate_expr.register
def _translate_equality_expr(
    expr: expressions.Equality, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "COMPARISON",
        "operator": _EQUALITY_OPERATOR_MAP[expr.operator],
        "left": _translate_expr(expr.left, ctx),
        "right": _translate_expr(expr.right, ctx),
    }


@_translate_expr.register
def _translate_comparison_expr(
    expr: expressions.Comparison, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "COMPARISON",
        "operator": expr.operator.name,
        "left": _translate_expr(expr.left, ctx),
        "right": _translate_expr(expr.right, ctx),
    }


@_translate_expr.register
def _translate_conditional_expr(
    expr: expressions.Conditional, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "CONDITIONAL",
        "condition": _translate_expr(expr.condition),
        "consequence": _translate_expr(expr.consequence),
        "alternative": _translate_expr(expr.alternative),
    }


@_translate_expr.register
def _translate_arithmetic_unary_expr(
    expr: expressions.ArithmeticUnary, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "UNARY",
        "operator": expr.operator.name,
        "operand": _translate_expr(expr.operand, ctx),
    }


@_translate_expr.register
def _translate_not_expr(expr: expressions.Not, ctx: _TranslationContext) -> _JSONObject:
    return {
        "kind": "UNARY",
        "operator": "NOT",
        "operand": _translate_expr(expr.operand, ctx),
    }


@_translate_expr.register
def _translate_array_access_expr(
    expr: expressions.ArrayAccess, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "INDEX",
        "vector": _translate_expr(expr.array),
        "index": _translate_expr(expr.index),
    }


@_translate_expr.register
def _translate_array_value_expr(
    expr: expressions.ArrayValue, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "VECTOR",
        "elements": [_translate_expr(element) for element in expr.elements],
    }


@_translate_expr.register
def _translate_array_constructor_expr(
    expr: expressions.ArrayConstructor, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "COMPREHENSION",
        "variable": expr.variable,
        "length": _translate_expr(expr.length),
        "element": _translate_expr(expr.expression),
    }


@_translate_expr.register
def _translate_trigonometric_expr(
    expr: expressions.Trigonometric, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "COMPREHENSION",
        "function": expr.operator.name,
        "operand": _translate_expr(expr.operand),
    }


def _compute_instance_names(network: model.Network) -> t.Mapping[model.Instance, str]:
    instance_names: t.Dict[model.Instance, str] = {}
    for number, instance in enumerate(network.instances):
        instance_names[instance] = f"_{number}_{instance.automaton.name}"
    return instance_names


def _extract_initial_value(declaration: model.VariableDeclaration) -> model.Expression:
    if declaration.initial_value is None:
        raise CompileError(
            f"No initial value for variable {declaration.identifier!r} provided."
        )
    return declaration.initial_value


def _compute_declarations_with_prefix(
    prefix: str, declarations: t.Iterable[model.VariableDeclaration]
) -> t.Mapping[str, VariableDeclaration]:
    return {
        declaration.identifier: VariableDeclaration(
            f"{prefix}{declaration.identifier}",
            declaration.typ,
            _extract_initial_value(declaration),
        )
        for declaration in declarations
    }


def _compute_declarations(
    network: model.Network, instance_names: t.Mapping[model.Instance, str]
) -> Declarations:
    return Declarations(
        _compute_declarations_with_prefix(
            "global_", network.ctx.global_scope.variable_declarations
        ),
        {
            instance: _compute_declarations_with_prefix(
                f"local_{instance_name}_",
                instance.automaton.scope.variable_declarations,
            )
            for instance, instance_name in instance_names.items()
        },
    )


def _insert_variable_declaration(
    global_variables: _JSONObject, declaration: VariableDeclaration
) -> None:
    identifier = declaration.identifier
    typ = _translate_type(declaration.typ)
    global_variables[identifier] = typ


def _compute_global_variables(declarations: Declarations) -> _JSONObject:
    global_variables: _JSONObject = {}
    for variable_declaration in declarations.globals_table.values():
        _insert_variable_declaration(global_variables, variable_declaration)
    for instance_declrations in declarations.locals_table.values():
        for variable_declaration in instance_declrations.values():
            _insert_variable_declaration(global_variables, variable_declaration)
    return global_variables


def _compute_action_types(network: model.Network) -> _JSONObject:
    return {
        action_type.name: [
            _translate_type(parameter.typ) for parameter in action_type.parameters
        ]
        for action_type in network.ctx.action_types.values()
    }


def _translate_action_pattern(
    action_pattern: t.Optional[model.ActionPattern],
) -> _JSONObject:
    if action_pattern is None:
        return {"kind": "SILENT"}
    else:
        assert (
            not action_pattern.arguments
        ), "Arguments for action patterns not implemented!"
        return {
            "kind": "LABELED",
            "label": action_pattern.action_type.name,
            "arguments": [],  # TODO: implement arguments for action patterns
        }


def _compute_location_names(instance: model.Instance) -> t.Mapping[model.Location, str]:
    return {
        location: f"_{location_index}_{location.name or ''}"
        for location_index, location in enumerate(instance.automaton.locations)
    }


def _translate_instance(
    instance: model.Instance,
    declarations: Declarations,
    location_names: t.Mapping[model.Location, str],
    parameters: t.Mapping[str, model.Expression],
) -> _JSONObject:
    ctx = _TranslationContext(
        parameters, declarations, instance.automaton.scope, instance
    )
    locations: t.Mapping[str, _JSONObject] = {
        location_name: {
            "invariant": [],  # TODO: implement location invariants
            "edges": [],
        }
        for location_name in location_names.values()
    }

    for edge in instance.automaton.edges:
        outgoing = locations[location_names[edge.location]]["edges"]
        action = _translate_action_pattern(edge.action_pattern)

        guard: _JSONObject
        if edge.guard is None:
            guard = {
                "boolean_condition": {"kind": "CONSTANT", "value": True},
                "clock_constraints": [],
            }
        else:
            guard = {
                "boolean_condition": _translate_expr(edge.guard, ctx),
                "clock_constraints": [],  # TODO: implement clock constraints
            }

        destinations: t.List[_JSONObject] = []

        for destination in edge.destinations:
            probability: _JSONObject
            if destination.probability is None:
                probability = {"kind": "CONSTANT", "value": 1.0}
            else:
                probability = _translate_expr(destination.probability, ctx)

            assignments: t.List[_JSONObject] = []

            for assignment in destination.assignments:
                assert isinstance(assignment.target, effects.Name)
                assignments.append(
                    {
                        "target": _translate_identifier(
                            assignment.target.identifier, ctx
                        ),
                        "value": _translate_expr(assignment.value, ctx),
                        "index": assignment.index,
                    }
                )

            destinations.append(
                {
                    "location": location_names[destination.location],
                    "probability": probability,
                    "assignments": assignments,
                    "reset": [],  # TODO: implement clocks to reset
                }
            )

        outgoing.append(
            {
                "pattern": action,
                "guard": guard,
                "destinations": destinations,
            }
        )

    return {"locations": locations}


def _translate_link(
    link: model.Link, instance_names: t.Mapping[model.Instance, str]
) -> _JSONObject:
    slots: t.Set[str] = set()

    def _translate_arguments(
        arguments: t.Iterable[model.ActionArgument],
    ) -> t.List[str]:
        slot_vector: t.List[str] = []
        for argument in arguments:
            assert isinstance(argument, actions.GuardArgument)
            slots.add(argument.identifier)
            slot_vector.append(argument.identifier)
        return slot_vector

    assert link.condition is None
    result: _JSONObject
    if link.result is None:
        result = {"kind": "SILENT"}
    else:
        result = {
            "kind": "LABELED",
            "action_type": link.result.action_type.name,
            "arguments": _translate_arguments(link.result.arguments),
        }
    vector: _JSONObject = {}
    for instance, pattern in link.vector.items():
        vector[instance_names[instance]] = {
            "action_type": pattern.action_type.name,
            "arguments": _translate_arguments(pattern.arguments),
        }

    return {"slots": list(slots), "vector": vector, "result": result}


def _update_initial_values(
    initial_values: _JSONObject,
    declarations: t.Iterable[VariableDeclaration],
) -> None:
    for declaration in declarations:
        initial_value = declaration.initial_value
        assert initial_value is not None
        if isinstance(initial_value, expressions.IntegerConstant):
            initial_values[declaration.identifier] = initial_value.integer
        elif isinstance(initial_value, expressions.BooleanConstant):
            initial_values[declaration.identifier] = initial_value.boolean
        elif isinstance(initial_value, expressions.RealConstant):
            initial_values[declaration.identifier] = initial_value.as_float
        else:
            raise CompileError(
                f"Invalid initial value {initial_value!r} for "
                f"variable {declaration.identifier!r}."
            )


def _translate_initial_states(
    network: model.Network,
    instance_names: t.Mapping[model.Instance, str],
    instance_to_location_names: t.Mapping[
        model.Instance, t.Mapping[model.Location, str]
    ],
    declarations: Declarations,
    parameters: t.Mapping[str, model.Expression],
) -> t.List[_JSONObject]:
    initial_locations: t.Dict[str, str] = {}
    for instance, instance_name in instance_names.items():
        # TODO: support more than one initial location
        (initial_location,) = instance.automaton.initial_locations
        location_name = instance_to_location_names[instance][initial_location]
        initial_locations[instance_name] = location_name
    initial_values: _JSONObject = {}
    _update_initial_values(initial_values, declarations.globals_table.values())
    for variable_declarations in declarations.locals_table.values():
        _update_initial_values(initial_values, variable_declarations.values())
    return [{"locations": initial_locations, "values": initial_values, "zone": []}]


def _translate_network(
    network: model.Network,
    instance_names: t.Mapping[model.Instance, str],
    instance_to_location_names: t.Mapping[
        model.Instance, t.Mapping[model.Location, str]
    ],
    declarations: Declarations,
    parameters: t.Mapping[str, model.Expression],
) -> str:
    return json.dumps(
        {
            "declarations": {
                "global_variables": _compute_global_variables(declarations),
                "transient_variables": {},  # TODO: implement transient variables
                "clock_variables": [],  # TODO: implement clock variables
                "action_labels": _compute_action_types(network),
            },
            "automata": {
                instance_name: _translate_instance(
                    instance,
                    declarations,
                    instance_to_location_names[instance],
                    parameters,
                )
                for instance, instance_name in instance_names.items()
            },
            "links": [_translate_link(link, instance_names) for link in network.links],
            "initial_states": _translate_initial_states(
                network,
                instance_names,
                instance_to_location_names,
                declarations,
                parameters,
            ),
        },
        indent=2,
    )


@d.dataclass(frozen=True)
class Translation:
    json_network: str
    instance_names: t.Mapping[model.Instance, str]
    declarations: Declarations
    instance_to_location_names: t.Mapping[
        model.Instance, t.Mapping[model.Location, str]
    ]

    @functools.cached_property
    def reversed_instance_to_location_names(
        self,
    ) -> t.Mapping[model.Instance, t.Mapping[str, model.Location]]:
        return {
            instance: {name: location for location, name in mapping.items()}
            for instance, mapping in self.instance_to_location_names.items()
        }


_NO_PARAMETERS: t.Mapping[str, model.Expression] = {}


def translate_network(
    network: model.Network,
    *,
    parameters: t.Mapping[str, model.Expression] = _NO_PARAMETERS,
) -> Translation:
    instance_names = _compute_instance_names(network)
    declarations = _compute_declarations(network, instance_names)
    instance_to_location_names = {
        instance: _compute_location_names(instance) for instance in network.instances
    }
    return Translation(
        _translate_network(
            network,
            instance_names,
            instance_to_location_names,
            declarations,
            parameters,
        ),
        instance_names,
        declarations,
        instance_to_location_names,
    )
