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
import fractions
import json

from .. import model
from ..model import actions, expressions, operators


if t.TYPE_CHECKING:
    from .explore import Parameters

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
    is_transient: bool


GlobalsTable = t.Mapping[str, VariableDeclaration]
LocalsTable = t.Mapping[model.Instance, t.Mapping[str, VariableDeclaration]]


@d.dataclass
class Declarations:
    globals_table: GlobalsTable = d.field(default_factory=dict)
    locals_table: LocalsTable = d.field(default_factory=dict)

    @property
    def all_declarations(self) -> t.List[VariableDeclaration]:
        all_declarations = list(self.globals_table.values())
        for local_declarations in self.locals_table.values():
            all_declarations.extend(local_declarations.values())
        return all_declarations


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
        "condition": _translate_expr(expr.condition, ctx),
        "consequence": _translate_expr(expr.consequence, ctx),
        "alternative": _translate_expr(expr.alternative, ctx),
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
        "vector": _translate_expr(expr.array, ctx),
        "index": _translate_expr(expr.index, ctx),
    }


@_translate_expr.register
def _translate_array_value_expr(
    expr: expressions.ArrayValue, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "VECTOR",
        "elements": [_translate_expr(element, ctx) for element in expr.elements],
    }


@_translate_expr.register
def _translate_array_constructor_expr(
    expr: expressions.ArrayConstructor, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "COMPREHENSION",
        "variable": expr.variable,
        "length": _translate_expr(expr.length, ctx),
        "element": _translate_expr(expr.expression, ctx),
    }


@_translate_expr.register
def _translate_trigonometric_expr(
    expr: expressions.Trigonometric, ctx: _TranslationContext
) -> _JSONObject:
    return {
        "kind": "TRIGONOMETRIC",
        "function": expr.operator.name,
        "operand": _translate_expr(expr.operand, ctx),
    }


def _compute_instance_names(network: model.Network) -> t.Mapping[model.Instance, str]:
    instance_names: t.Dict[model.Instance, str] = {}
    for number, instance in enumerate(network.instances):
        instance_names[instance] = f"{number}_{instance.automaton.name}"
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
            declaration.is_transient or False,
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
        if variable_declaration.is_transient:
            continue
        elif variable_declaration.typ == model.types.CLOCK:
            continue
        _insert_variable_declaration(global_variables, variable_declaration)
    for instance_declarations in declarations.locals_table.values():
        for variable_declaration in instance_declarations.values():
            if variable_declaration.is_transient:
                continue
            elif variable_declaration.typ == model.types.CLOCK:
                continue
            _insert_variable_declaration(global_variables, variable_declaration)
    return global_variables


def _compute_action_types(network: model.Network) -> _JSONObject:
    return {
        action_type.label: [
            _translate_type(parameter.typ) for parameter in action_type.parameters
        ]
        for action_type in network.ctx.action_types.values()
    }


def _translate_pattern_argument(
    argument: model.ActionArgument, ctx: _TranslationContext
) -> _JSONObject:
    if isinstance(argument, model.ReadArgument):
        return {"direction": "READ", "identifier": argument.identifier}
    else:
        assert isinstance(
            argument, model.WriteArgument
        ), "guard arguments are not supported"
        return {
            "direction": "WRITE",
            "value": _translate_expr(argument.expression, ctx),
        }


def _translate_action_pattern(
    action_pattern: t.Optional[model.ActionPattern],
    ctx: _TranslationContext,
) -> _JSONObject:
    if action_pattern is None:
        return {"kind": "SILENT"}
    else:
        return {
            "kind": "LABELED",
            "label": action_pattern.action_type.label,
            "arguments": [
                _translate_pattern_argument(argument, ctx)
                for argument in action_pattern.arguments
            ],
        }


def _contains_clock_identifier(
    expression: expressions.Expression, scope: model.Scope
) -> bool:
    for used_name in expression.used_names:
        if scope.lookup(used_name.identifier).typ == model.types.CLOCK:
            return True
    return False


@d.dataclass(frozen=True)
class ExtractedConstraints:
    conjuncts: t.List[expressions.Expression]
    constraints: t.List[_JSONObject]


def _extract_constraints(
    expr: model.Expression,
    ctx: _TranslationContext,
) -> ExtractedConstraints:
    constraints: t.List[_JSONObject] = []
    conjuncts: t.List[expressions.Expression] = []
    pending: t.List[expressions.Expression] = [expr]
    while pending:
        head = pending.pop()
        if isinstance(head, expressions.Boolean):
            if head.operator is operators.BooleanOperator.AND:
                pending.append(head.left)
                pending.append(head.right)
            else:
                conjuncts.append(head)
        elif isinstance(head, expressions.Comparison):
            if _contains_clock_identifier(head.left, ctx.scope):
                difference = head.left
                operator = head.operator
                bound_expression = head.right
            elif _contains_clock_identifier(head.right, ctx.scope):
                difference = head.right
                operator = head.operator.swap()
                bound_expression = head.left
            else:
                conjuncts.append(head)
                continue
            assert not _contains_clock_identifier(bound_expression, ctx.scope)
            left: _JSONObject
            right: _JSONObject
            if isinstance(difference, expressions.Name):
                left = {
                    "kind": "VARIABLE",
                    "identifier": _translate_identifier(difference.identifier, ctx)[
                        "identifier"
                    ],
                }
                right = {"kind": "ZERO"}
            else:
                assert (
                    isinstance(difference, expressions.ArithmeticBinary)
                    and difference.operator is operators.ArithmeticBinaryOperator.SUB
                    and isinstance(difference.left, expressions.Name)
                    and isinstance(difference.right, expressions.Name)
                )
                left = {
                    "kind": "VARIABLE",
                    "identifier": _translate_identifier(
                        difference.left.identifier, ctx
                    )["identifier"],
                }
                right = {
                    "kind": "VARIABLE",
                    "identifier": _translate_identifier(
                        difference.right.identifier, ctx
                    )["identifier"],
                }
            bound = _translate_expr(bound_expression, ctx)
            if operator.is_less:
                constraints.append(
                    {
                        "left": left,
                        "right": right,
                        "is_strict": operator.is_strict,
                        "bound": bound,
                    }
                )
            else:
                assert operator.is_greater
                constraints.append(
                    {
                        "left": right,
                        "right": left,
                        "is_strict": operator.is_strict,
                        "bound": {
                            "kind": "BINARY",
                            "operator": "SUB",
                            "left": {"kind": "CONSTANT", "value": 0},
                            "right": bound,
                        },
                    }
                )
        else:
            conjuncts.append(head)
    return ExtractedConstraints(conjuncts, constraints)


def _compute_location_names(instance: model.Instance) -> t.Mapping[model.Location, str]:
    return {
        location: f"{location_index}_{location.name or ''}"
        for location_index, location in enumerate(instance.automaton.locations)
    }


def _extract_invariant(
    location: model.Location, ctx: _TranslationContext
) -> t.List[_JSONObject]:
    if location.progress_invariant is None:
        return []
    extracted_constraints = _extract_constraints(location.progress_invariant, ctx)
    assert (
        not extracted_constraints.conjuncts
    ), "invariant must be a conjunction of clock constraints"
    return extracted_constraints.constraints


def _translate_instance(
    instance: model.Instance,
    declarations: Declarations,
    location_names: t.Mapping[model.Location, str],
    parameters: t.Mapping[str, model.Expression],
) -> _JSONObject:
    parameters = dict(parameters)

    for parameter_name, parameter_value in zip(
        instance.automaton.parameters, instance.arguments
    ):
        parameters[parameter_name] = parameter_value

    ctx = _TranslationContext(
        parameters, declarations, instance.automaton.scope, instance
    )
    locations: t.Mapping[str, _JSONObject] = {
        location_name: {
            "invariant": _extract_invariant(location, ctx),
            "edges": [],
        }
        for location, location_name in location_names.items()
    }

    for number, edge in enumerate(instance.automaton.edges):
        outgoing = locations[location_names[edge.location]]["edges"]
        action = _translate_action_pattern(edge.action_pattern, ctx)

        guard: _JSONObject
        if edge.guard is None:
            guard = {
                "boolean_condition": {"kind": "CONSTANT", "value": True},
                "clock_constraints": [],
            }
        else:
            extracted_constraints = _extract_constraints(edge.guard, ctx)
            guard = {
                "boolean_condition": _translate_expr(
                    expressions.logic_all(*extracted_constraints.conjuncts), ctx
                ),
                "clock_constraints": extracted_constraints.constraints,
            }

        destinations: t.List[_JSONObject] = []

        edge_ctx = _TranslationContext(
            parameters,
            declarations,
            edge.create_edge_scope(instance.automaton.scope),
            instance,
        )

        for destination in edge.destinations:
            probability: _JSONObject
            if destination.probability is None:
                probability = {"kind": "CONSTANT", "value": 1.0}
            else:
                probability = _translate_expr(destination.probability, edge_ctx)

            assignments: t.List[_JSONObject] = []

            reset_clocks: t.List[_JSONObject] = []

            for assignment in destination.assignments:
                assert isinstance(assignment.target, expressions.Name)
                declaration = edge_ctx.scope.lookup(assignment.target.identifier)
                if declaration.typ == model.types.CLOCK:
                    assert assignment.index == 0
                    assert isinstance(assignment.value, expressions.IntegerConstant)
                    reset_clocks.append(
                        {
                            "kind": "VARIABLE",
                            "identifier": _translate_identifier(
                                assignment.target.identifier, ctx
                            )["identifier"],
                        }
                    )
                else:
                    assignments.append(
                        {
                            "target": _translate_identifier(
                                assignment.target.identifier, edge_ctx
                            ),
                            "value": _translate_expr(assignment.value, edge_ctx),
                            "index": assignment.index,
                        }
                    )

            destinations.append(
                {
                    "location": location_names[destination.location],
                    "probability": probability,
                    "assignments": assignments,
                    "reset": reset_clocks,
                }
            )

        outgoing.append(
            {
                "number": number,
                "pattern": action,
                "guard": guard,
                "destinations": destinations,
                "observations": [
                    {
                        "label": observation.action_type.label,
                        "arguments": [
                            _translate_expr(argument, edge_ctx)
                            for argument in observation.arguments
                        ],
                        "probability": _translate_expr(
                            observation.probability
                            or expressions.RealConstant(fractions.Fraction(1.0)),
                            edge_ctx,
                        ),
                    }
                    for observation in edge.observation
                ],
            }
        )

    return {"locations": locations}


def _translate_link(
    link: model.Link, instance_names: t.Mapping[model.Instance, str]
) -> _JSONObject:
    slots: t.List[str] = []

    def _translate_arguments(
        arguments: t.Iterable[model.ActionArgument],
    ) -> t.List[str]:
        slot_vector: t.List[str] = []
        for argument in arguments:
            assert isinstance(argument, actions.GuardArgument)
            if argument.identifier not in slots:
                slots.append(argument.identifier)
            slot_vector.append(argument.identifier)
        return slot_vector

    assert link.condition is None
    result: _JSONObject
    if link.result is None:
        result = {"kind": "SILENT"}
    else:
        result = {
            "kind": "LABELED",
            "action_type": link.result.action_type.label,
            "arguments": _translate_arguments(link.result.arguments),
        }
    vector: _JSONObject = {}
    for instance, pattern in link.vector.items():
        vector[instance_names[instance]] = {
            "action_type": pattern.action_type.label,
            "arguments": _translate_arguments(pattern.arguments),
        }

    return {"slots": slots, "vector": vector, "result": result}


def _extract_constant_value(expr: expressions.Expression) -> t.Any:
    if isinstance(expr, expressions.IntegerConstant):
        return expr.integer
    elif isinstance(expr, expressions.BooleanConstant):
        return expr.boolean
    elif isinstance(expr, expressions.RealConstant):
        return expr.as_float
    elif isinstance(expr, expressions.ArrayValue):
        return [_extract_constant_value(element) for element in expr.elements]
    else:
        raise CompileError(f"Unable to extract constant value from {expr}.")


def _update_initial_values(
    initial_values: _JSONObject,
    clock_constraints: t.List[_JSONObject],
    declarations: t.Iterable[VariableDeclaration],
) -> None:
    for declaration in declarations:
        initial_value = declaration.initial_value
        assert initial_value is not None
        value = _extract_constant_value(initial_value)

        if declaration.is_transient:
            continue
        elif declaration.typ == model.types.CLOCK:
            assert isinstance(value, (int, float))
            clock_constraints.append(
                {
                    "left": {"kind": "VARIABLE", "identifier": declaration.identifier},
                    "right": {"kind": "ZERO"},
                    "is_strict": False,
                    "bound": {"kind": "CONSTANT", "value": value},
                }
            )
            clock_constraints.append(
                {
                    "left": {"kind": "ZERO"},
                    "right": {"kind": "VARIABLE", "identifier": declaration.identifier},
                    "is_strict": False,
                    "bound": {"kind": "CONSTANT", "value": -value},
                }
            )
        else:
            initial_values[declaration.identifier] = value


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
    clock_constraints: t.List[_JSONObject] = []
    _update_initial_values(
        initial_values, clock_constraints, declarations.globals_table.values()
    )
    for variable_declarations in declarations.locals_table.values():
        _update_initial_values(
            initial_values, clock_constraints, variable_declarations.values()
        )
    # for declaration in declarations.all_declarations:
    #     if declaration.typ != model.types.CLOCK:
    #         continue
    #     for other_declaration in declarations.all_declarations:
    #         if (
    #             declaration == other_declaration
    #             or other_declaration.typ != model.types.CLOCK
    #         ):
    #             continue
    #         clock_constraints.append(
    #             {
    #                 "left": {"kind": "VARIABLE", "identifier": declaration.identifier},
    #                 "right": {"kind": "VARIABLE", "identifier": other_declaration.identifier},
    #                 "is_strict": False,
    #                 "bound": {"kind": "CONSTANT", "value": },
    #             }
    #         )
    return [
        {
            "locations": initial_locations,
            "values": initial_values,
            "zone": clock_constraints,
        }
    ]


def _translate_network(
    network: model.Network,
    instance_names: t.Mapping[model.Instance, str],
    instance_to_location_names: t.Mapping[
        model.Instance, t.Mapping[model.Location, str]
    ],
    declarations: Declarations,
    parameters: t.Mapping[str, model.Expression],
    global_clock: bool,
) -> str:
    ctx = _TranslationContext(parameters, declarations, network.ctx.global_scope, None)
    return json.dumps(
        {
            "declarations": {
                "global_variables": _compute_global_variables(declarations),
                "transient_variables": {
                    declaration.identifier: _translate_expr(
                        declaration.initial_value, ctx
                    )
                    for declaration in declarations.all_declarations
                    if declaration.is_transient
                },
                "clock_variables": [
                    declaration.identifier
                    for declaration in declarations.all_declarations
                    if declaration.typ == model.types.CLOCK
                ]
                + (["global_clock"] if global_clock else []),
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
    network: model.Network
    parameters: t.Mapping[str, model.Expression]
    json_network: str
    instance_names: t.Mapping[model.Instance, str]
    declarations: Declarations
    instance_to_location_names: t.Mapping[
        model.Instance, t.Mapping[model.Location, str]
    ]
    instance_vector: t.Tuple[model.Instance, ...]

    @functools.cached_property
    def instance_name_to_instance(self) -> t.Mapping[str, model.Instance]:
        return {name: instance for instance, name in self.instance_names.items()}

    @functools.cached_property
    def reversed_instance_to_location_names(
        self,
    ) -> t.Mapping[model.Instance, t.Mapping[str, model.Location]]:
        return {
            instance: {name: location for location, name in mapping.items()}
            for instance, mapping in self.instance_to_location_names.items()
        }

    def translate_global_expression(self, expr: model.Expression) -> str:
        ctx = _TranslationContext(
            self.parameters,
            self.declarations,
            self.network.ctx.global_scope,
            None,
        )
        return json.dumps(_translate_expr(expr, ctx))


def translate_network(
    network: model.Network, *, parameters: Parameters = None, global_clock: bool = False
) -> Translation:
    instance_names = _compute_instance_names(network)
    declarations = _compute_declarations(network, instance_names)
    instance_to_location_names = {
        instance: _compute_location_names(instance) for instance in network.instances
    }
    ctx_parameters = {
        name: expressions.ensure_expr(expr) for name, expr in (parameters or {}).items()
    }
    return Translation(
        network,
        ctx_parameters,
        _translate_network(
            network,
            instance_names,
            instance_to_location_names,
            declarations,
            ctx_parameters,
            global_clock,
        ),
        instance_names,
        declarations,
        instance_to_location_names,
        # HACK: this relies on dictionaries being ordered on both the Python and Rust side
        instance_vector=tuple(instance_names.keys()),
    )
