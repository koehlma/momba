# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations


import typing as t

import functools

from momba.model import functions

from .. import model
from ..model import expressions, operators


JSONObject = t.Dict[str, t.Any]


class TranslationError(Exception):
    pass


@functools.singledispatch
def translate_type(typ: model.Type) -> JSONObject:
    raise NotImplementedError(f"'_translate_type' not implemented for {type(typ)}")


@translate_type.register
def _translate_integer_type(typ: model.types.IntegerType) -> JSONObject:
    return {"type": "Int"}


@translate_type.register
def _translate_real_type(typ: model.types.RealType) -> JSONObject:
    return {"type": "Real"}


@translate_type.register
def _translate_bounded_type(typ: model.types.BoundedType) -> JSONObject:
    json: JSONObject = {}
    if isinstance(typ.base, model.types.IntegerType):
        json["type"] = "Int"
    elif isinstance(typ.base, model.types.RealType):
        json["type"] = "Real"
    else:
        raise TranslationError(
            f"Bounded type has invalid base type `{type(typ.base)}`."
        )
    if typ.lower_bound is not None:
        json["lowerBound"] = translate_expr(typ.lower_bound)
    if typ.upper_bound is not None:
        json["upperBound"] = translate_expr(typ.upper_bound)
    return json


@translate_type.register
def _translate_bool_type(typ: model.types.BoolType) -> JSONObject:
    return {"type": "Bool"}


@translate_type.register
def _translate_array_type(typ: model.types.ArrayType) -> JSONObject:
    return {"type": "Array", "element_type": translate_type(typ.element)}


@functools.singledispatch
def translate_expr(expr: model.Expression) -> JSONObject:
    raise NotImplementedError(f"'_translate_expr' not implemented for {type(expr)}")


@translate_expr.register
def _translate_boolean_constant_expr(expr: expressions.BooleanConstant) -> JSONObject:
    return {"kind": "Constant", "value": {"type": "Bool", "value": expr.boolean}}


@translate_expr.register
def _translate_integer_constant_expr(expr: expressions.IntegerConstant) -> JSONObject:
    return {"kind": "Constant", "value": {"type": "Int", "value": expr.integer}}


@translate_expr.register
def _translate_real_constant_expr(expr: expressions.RealConstant) -> JSONObject:
    return {"kind": "Constant", "value": {"type": "Float", "value": expr.as_float}}


@translate_expr.register
def _translate_name_expr(expr: expressions.Name) -> JSONObject:
    return {"kind": "Identifier", "identifier": expr.identifier}


_BOOLEAN_OPERATOR_MAP = {
    operators.BooleanOperator.AND: "And",
    operators.BooleanOperator.OR: "Or",
    operators.BooleanOperator.XOR: "Xor",
    operators.BooleanOperator.EQUIV: "Equiv",
    operators.BooleanOperator.IMPLY: "Imply",
}


@translate_expr.register
def _translate_boolean_expr(expr: expressions.Boolean) -> JSONObject:
    return {
        "kind": "Binary",
        "operator": _BOOLEAN_OPERATOR_MAP[expr.operator],
        "left": translate_expr(expr.left),
        "right": translate_expr(expr.right),
    }


_ARITHMETIC_BINARY_OPERATOR_MAP = {
    operators.ArithmeticBinaryOperator.ADD: "Add",
    operators.ArithmeticBinaryOperator.SUB: "Sub",
    operators.ArithmeticBinaryOperator.MUL: "Mul",
    operators.ArithmeticBinaryOperator.MOD: "Mod",
    operators.ArithmeticBinaryOperator.REAL_DIV: "RealDiv",
    operators.ArithmeticBinaryOperator.LOG: "Log",
    operators.ArithmeticBinaryOperator.POW: "Pow",
    operators.ArithmeticBinaryOperator.MIN: "Min",
    operators.ArithmeticBinaryOperator.MAX: "Max",
    operators.ArithmeticBinaryOperator.FLOOR_DIV: "FloorDiv",
}


@translate_expr.register
def _translate_arithmetic_binary_expr(
    expr: expressions.ArithmeticBinary,
) -> JSONObject:
    return {
        "kind": "Binary",
        "operator": _ARITHMETIC_BINARY_OPERATOR_MAP[expr.operator],
        "left": translate_expr(expr.left),
        "right": translate_expr(expr.right),
    }


_EQUALITY_OPERATOR_MAP = {
    operators.EqualityOperator.EQ: "Eq",
    operators.EqualityOperator.NEQ: "Ne",
}


@translate_expr.register
def _translate_equality_expr(expr: expressions.Equality) -> JSONObject:
    return {
        "kind": "Binary",
        "operator": _EQUALITY_OPERATOR_MAP[expr.operator],
        "left": translate_expr(expr.left),
        "right": translate_expr(expr.right),
    }


_COMPARISON_OPERATOR_MAP = {
    operators.ComparisonOperator.LE: "Le",
    operators.ComparisonOperator.LT: "Lt",
    operators.ComparisonOperator.GE: "Ge",
    operators.ComparisonOperator.GT: "Gt",
}


@translate_expr.register
def _translate_comparison_expr(expr: expressions.Comparison) -> JSONObject:
    return {
        "kind": "Binary",
        "operator": _COMPARISON_OPERATOR_MAP[expr.operator],
        "left": translate_expr(expr.left),
        "right": translate_expr(expr.right),
    }


@translate_expr.register
def _translate_conditional_expr(expr: expressions.Conditional) -> JSONObject:
    return {
        "kind": "Conditional",
        "condition": translate_expr(expr.condition),
        "consequence": translate_expr(expr.consequence),
        "alternative": translate_expr(expr.alternative),
    }


_ARITHMETIC_UNARY_OPERATOR_MAP = {
    operators.ArithmeticUnaryOperator.ABS: "Abs",
    operators.ArithmeticUnaryOperator.CEIL: "Ceil",
    operators.ArithmeticUnaryOperator.FLOOR: "Floor",
    operators.ArithmeticUnaryOperator.SGN: "Sgn",
    operators.ArithmeticUnaryOperator.TRC: "Trc",
}


@translate_expr.register
def _translate_arithmetic_unary_expr(expr: expressions.ArithmeticUnary) -> JSONObject:
    return {
        "kind": "Unary",
        "operator": _ARITHMETIC_UNARY_OPERATOR_MAP[expr.operator],
        "operand": translate_expr(expr.operand),
    }


@translate_expr.register
def _translate_not_expr(expr: expressions.Not) -> JSONObject:
    return {
        "kind": "Unary",
        "operator": "Not",
        "operand": translate_expr(expr.operand),
    }


_TRIGONOMETRIC_OPERATOR_MAP = {
    operators.TrigonometricFunction.SIN: "Sin",
    operators.TrigonometricFunction.COS: "Cos",
    operators.TrigonometricFunction.TAN: "Tan",
    operators.TrigonometricFunction.COT: "Cot",
    operators.TrigonometricFunction.SEC: "Sec",
    operators.TrigonometricFunction.CSC: "Csc",
    operators.TrigonometricFunction.ARC_SIN: "ArcSin",
    operators.TrigonometricFunction.ARC_COS: "ArcCos",
    operators.TrigonometricFunction.ARC_TAN: "ArcTan",
    operators.TrigonometricFunction.ARC_COT: "ArcCot",
    operators.TrigonometricFunction.ARC_SEC: "ArcSec",
    operators.TrigonometricFunction.ARC_CSC: "ArcCsc",
}


@translate_expr.register
def _translate_trigonometric_expr(expr: expressions.Trigonometric) -> JSONObject:
    return {
        "kind": "Unary",
        "operator": _TRIGONOMETRIC_OPERATOR_MAP[expr.operator],
        "operand": translate_expr(expr.operand),
    }


@translate_expr.register
def _translate_array_access_expr(expr: expressions.ArrayAccess) -> JSONObject:
    return {
        "kind": "Index",
        "slice": translate_expr(expr.array),
        "index": translate_expr(expr.index),
    }


@translate_expr.register
def _translate_array_value_expr(expr: expressions.ArrayValue) -> JSONObject:
    return {
        "kind": "Array",
        "elements": [translate_expr(element) for element in expr.elements],
    }


@translate_expr.register
def _translate_array_constructor_expr(
    expr: expressions.ArrayConstructor,
) -> JSONObject:
    return {
        "kind": "Comprehension",
        "variable": expr.variable,
        "length": translate_expr(expr.length),
        "value": translate_expr(expr.expression),
    }


@translate_expr.register
def _translate_call_expr(expr: functions.CallExpression) -> JSONObject:
    return {
        "kind": "Call",
        "function": expr.function,
        "arguments": [translate_expr(argument) for argument in expr.arguments],
    }


def _translate_constant_declaration(decl: model.ConstantDeclaration) -> JSONObject:
    json: JSONObject = {"name": decl.identifier, "typ": translate_type(decl.typ)}
    if decl.value is not None:
        json["default"] = translate_expr(decl.value)
    return json


def _translate_variable_declaration(decl: model.VariableDeclaration) -> JSONObject:
    json: JSONObject = {
        "name": decl.identifier,
        "typ": translate_type(decl.typ),
        "transient": decl.is_transient or False,
    }
    if decl.initial_value is not None:
        json["default"] = translate_expr(decl.initial_value)
    return json


def _translate_action_type(typ: model.ActionType) -> JSONObject:
    return {
        "label": typ.label,
        "arguments": [translate_type(param.typ) for param in typ.parameters],
    }


def _translate_assignment(assignment: model.Assignment) -> JSONObject:
    return {
        "target": translate_expr(assignment.target),
        "value": translate_expr(assignment.value),
        "index": assignment.index,
    }


def _translate_location(
    automaton: model.Automaton, location: model.Location
) -> JSONObject:
    json: JSONObject = {
        "name": location.name,
        "initial": location in automaton.initial_locations,
    }
    if location.progress_invariant is not None:
        json["invariant"] = translate_expr(location.progress_invariant)
    if location.transient_values:
        json["assignments"] = [
            _translate_assignment(assignment)
            for assignment in location.transient_values
        ]
    return json


def _translate_pattern_argument(arg: model.ActionArgument) -> JSONObject:
    if isinstance(arg, model.ReadArgument):
        return {"kind": "Read", "identifier": arg.identifier}
    elif isinstance(arg, model.WriteArgument):
        return {"kind": "Write", "value": translate_expr(arg.expression)}
    else:
        raise TranslationError(f"Unsupported action argument `{type(arg)}`.")


def _translate_action_pattern(pattern: model.ActionPattern) -> JSONObject:
    return {
        "kind": "Labeled",
        "label": pattern.action_type.label,
        "arguments": [_translate_pattern_argument(arg) for arg in pattern.arguments],
    }


def _translate_edge(edge: model.Edge) -> JSONObject:
    json: JSONObject = {"source": edge.location.name}
    if edge.rate is not None:
        json["rate"] = translate_expr(edge.rate)
    if edge.guard is not None:
        json["guard"] = translate_expr(edge.guard)
    json["destinations"] = [
        _translate_destination(destination) for destination in edge.destinations
    ]
    if edge.action_pattern is None:
        json["action"] = {"kind": "Silent"}
    else:
        json["action"] = _translate_action_pattern(edge.action_pattern)
    return json


def _translate_destination(destination: model.Destination) -> JSONObject:
    json: JSONObject = {
        "target": destination.location.name,
        "assignments": [
            _translate_assignment(assignment) for assignment in destination.assignments
        ],
    }
    if destination.probability is not None:
        json["probability"] = translate_expr(destination.probability)
    return json


def _translate_automaton(automaton: model.Automaton) -> JSONObject:
    if automaton.name is None:
        raise TranslationError("Automata must be named!")
    json: JSONObject = {
        "name": automaton.name,
        "locals": [
            _translate_variable_declaration(decl)
            for decl in automaton.scope.variable_declarations
        ],
        "locations": [
            _translate_location(automaton, location) for location in automaton.locations
        ],
        "edges": [_translate_edge(edge) for edge in automaton.edges],
    }
    if automaton.initial_restriction:
        json["initial"] = translate_expr(automaton.initial_restriction)
    return json


def _translate_instance(instance: model.Instance) -> JSONObject:
    return {"automaton": instance.automaton.name}


def _translate_link_pattern(pattern: model.ActionPattern) -> JSONObject:
    return {
        "label": pattern.action_type.label,
        "arguments": [
            argument.identifier
            for argument in pattern.arguments
            if isinstance(argument, model.GuardArgument)
        ],
    }


def _translate_link(network: model.Network, link: model.Link) -> JSONObject:
    slots = list(
        set(
            arg.identifier
            for pattern in link.vector.values()
            for arg in pattern.arguments
            if isinstance(arg, model.GuardArgument)
        )
    )
    if link.result is None:
        result = {"kind": "Silent"}
    else:
        result = _translate_link_pattern(link.result)
        result["kind"] = "Labeled"
    return {
        "slots": slots,
        "vector": {
            network.instances.index(instance): _translate_link_pattern(pattern)
            for instance, pattern in link.vector.items()
        },
        "result": result,
    }


def _translate_function(function: functions.FunctionDefinition) -> JSONObject:
    return {
        "name": function.name,
        "parameters": [
            {"name": parameter.name, "typ": translate_type(parameter.typ)}
            for parameter in function.parameters
        ],
        "output": translate_type(function.returns),
        "body": translate_expr(function.body),
    }


_MODEL_TYPES_MAP = {
    model.ModelType.LTS: "Lts",
    model.ModelType.DTMC: "Dtmc",
    model.ModelType.CTMC: "Ctmc",
    model.ModelType.MDP: "Mdp",
    model.ModelType.TA: "Ta",
    model.ModelType.PTA: "Pta",
}


def translate_model(network: model.Network) -> JSONObject:
    json: JSONObject = {
        "typ": _MODEL_TYPES_MAP[network.ctx.model_type],
        "constants": [
            _translate_constant_declaration(decl)
            for decl in network.ctx.global_scope.constant_declarations
        ],
        "globals": [
            _translate_variable_declaration(decl)
            for decl in network.ctx.global_scope.variable_declarations
        ],
        "actions": [
            _translate_action_type(typ) for typ in network.ctx.action_types.values()
        ],
        "functions": [
            _translate_function(function)
            for function in network.ctx.global_scope.functions
        ],
        "automata": [
            _translate_automaton(automaton) for automaton in network.ctx.automata
        ],
        "links": [_translate_link(network, link) for link in network.links],
        "instances": [_translate_instance(instance) for instance in network.instances],
    }
    if network.initial_restriction is not None:
        json["initial"] = translate_expr(network.initial_restriction)
    return json
