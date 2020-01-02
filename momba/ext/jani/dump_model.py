# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses
import enum
import functools
import json
import warnings

from ... import model
from ...model import effects, context, expressions, operators, types, values
from ...utils import checks


# XXX: ignore this type definition, mypy does not support recursive types
JSON = t.Union[None, int, float, str, t.Sequence["JSON"], t.Mapping[str, "JSON"]]  # type: ignore

_JANIMap = t.Dict[str, JSON]  # type: ignore


class ModelFeature(enum.Enum):
    ARRAYS = "arrays"
    DATATYPES = "datatypes"
    DERIVED_OPERATORS = "derived-operators"
    EDGE_PRIORITIES = "edge-priorities"
    FUNCTIONS = "functions"
    HYPERBOLIC_FUNCTIONS = "hyperbolic-functions"
    NAMED_EXPRESSIONS = "named-expressions"
    NONDET_EXPRESSIONS = "nondet-expressions"
    STATE_EXIT_REWARDS = "state-exit-rewards"
    TRADEOFF_PROPERTIES = "tradeoff-properties"
    TRIGONOMETRIC_FUNCTIONS = "trigonometric-functions"

    X_MOMBA_OPERATORS = "x-momba-operators"


_NamedObject = t.Union[model.Location, model.Automaton]


@dataclasses.dataclass
class JANIContext:
    allow_momba_operators: bool = False

    _name_counter: int = 0

    _actions: t.Set[str] = dataclasses.field(default_factory=set)

    _names: t.Dict[_NamedObject, str] = dataclasses.field(default_factory=dict)

    _features: t.Set[ModelFeature] = dataclasses.field(default_factory=set)

    @property
    def features(self) -> t.AbstractSet[ModelFeature]:
        return self._features

    @property
    def actions(self) -> t.AbstractSet[str]:
        return self._actions

    def get_unique_name(self) -> str:
        name = f"__momba_{self._name_counter}"
        self._name_counter += 1
        return name

    def get_name(self, obj: _NamedObject) -> str:
        if obj.name is None:
            try:
                return self._names[obj]
            except KeyError:
                name = self._names[obj] = self.get_unique_name()
                return name
        return obj.name

    def require_action(self, action: str) -> str:
        self._actions.add(action)
        return action

    def require(self, feature: ModelFeature) -> None:
        self._features.add(feature)


@functools.singledispatch
def _dump_type(typ: model.Type, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"dump has not been implemented for type {typ}")


@_dump_type.register
def _dump_type_integer(typ: types.IntegerType, ctx: JANIContext) -> JSON:
    return "int"


@_dump_type.register
def _dump_type_real(typ: types.RealType, ctx: JANIContext) -> JSON:
    return "real"


@_dump_type.register
def _dump_type_bool(typ: types.BoolType, ctx: JANIContext) -> JSON:
    return "bool"


@_dump_type.register
def _dump_type_clock(typ: types.ClockType, ctx: JANIContext) -> JSON:
    return "clock"


@_dump_type.register
def _dump_type_continuous(typ: types.ContinuousType, ctx: JANIContext) -> JSON:
    return "continuous"


@_dump_type.register
def _dump_bounded_type(typ: types.BoundedType, ctx: JANIContext) -> JSON:
    jani_type: _JANIMap = {"kind": "bounded", "base": _dump_type(typ.base, ctx)}
    if typ.upper_bound:
        jani_type["upper-bound"] = _dump_expr(typ.upper_bound, ctx)
    if typ.lower_bound:
        jani_type["lower-bound"] = _dump_expr(typ.lower_bound, ctx)
    return jani_type


@_dump_type.register
def _dump_array_type(typ: types.ArrayType, ctx: JANIContext) -> JSON:
    ctx.require(ModelFeature.ARRAYS)
    return {"kind": "array", "base": _dump_type(typ.base, ctx)}


checks.check_singledispatch(_dump_type, types.Type)


@functools.singledispatch
def _dump_model_value(value: model.Value, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"dump has not been implemented for model value {value}")


@_dump_model_value.register
def _dump_boolean_value(value: values.BooleanValue, ctx: JANIContext) -> JSON:
    return value.boolean


@_dump_model_value.register
def _dump_integer_value(value: values.IntegerValue, ctx: JANIContext) -> JSON:
    return value.integer


@_dump_model_value.register
def _dump_real_value(value: values.RealValue, ctx: JANIContext) -> JSON:
    if isinstance(value.real, values.NamedReal):
        return {"constant": value.real.symbol}
    if not isinstance(value.real, float):
        warnings.warn(
            f"imprecise conversion: JSON does not support the number type {type(value.real)}"
        )
    return float(value.real)


checks.check_singledispatch(_dump_model_value, model.Value)


@functools.singledispatch
def _dump_expr(expr: model.Expression, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"dump has not been implemented for expression {expr}")


@_dump_expr.register
def _dump_identifier(expr: expressions.Identifier, ctx: JANIContext) -> JSON:
    return expr.identifier


@_dump_expr.register
def _dump_constant(expr: expressions.Constant, ctx: JANIContext) -> JSON:
    return _dump_model_value(expr.value, ctx)


@_dump_expr.register
def _dump_conditional(expr: expressions.Conditional, ctx: JANIContext) -> JSON:
    return {
        "op": "ite",
        "if": _dump_expr(expr.condition, ctx),
        "then": _dump_expr(expr.consequence, ctx),
        "else": _dump_expr(expr.alternative, ctx),
    }


_DERIVED_OPERATORS = {
    operators.Boolean.IMPLY,
    operators.Comparison.GT,
    operators.Comparison.GE,
    operators.ArithmeticOperator.MIN,
    operators.ArithmeticOperator.MAX,
}


_Transform = t.Callable[[expressions.Expression], expressions.Expression]

_MOMBA_OPERATORS: t.Mapping[operators.BinaryOperator, _Transform] = {
    operators.Boolean.XOR: expressions.normalize_xor,
    operators.Boolean.EQUIV: expressions.normalize_equiv,
    operators.ArithmeticOperator.FLOOR_DIV: expressions.normalize_floor_div,
}


@_dump_expr.register
def _dump_binary_expression(
    expr: expressions.BinaryExpression, ctx: JANIContext
) -> JSON:
    if expr.operator in _DERIVED_OPERATORS:
        ctx.require(ModelFeature.DERIVED_OPERATORS)
    if expr.operator in _MOMBA_OPERATORS:
        if ctx.allow_momba_operators:
            ctx.require(ModelFeature.X_MOMBA_OPERATORS)
        else:
            return _dump_expr(_MOMBA_OPERATORS[expr.operator](expr), ctx)
    return {
        "op": expr.operator.symbol,
        "left": _dump_expr(expr.left, ctx),
        "right": _dump_expr(expr.right, ctx),
    }


@_dump_expr.register
def _dump_unary_expression(expr: expressions.UnaryExpression, ctx: JANIContext) -> JSON:
    return {"op": expr.operator.symbol, "exp": _dump_expr(expr.operand, ctx)}


@_dump_expr.register
def _dump_derivative(expr: expressions.Derivative, ctx: JANIContext) -> JSON:
    return {"op": "der", "var": expr.identifier}


@_dump_expr.register
def _dump_sample(expr: expressions.Sample, ctx: JANIContext) -> JSON:
    return {
        "distribution": expr.distribution.jani_name,
        "args": [_dump_expr(argument, ctx) for argument in expr.arguments],
    }


@_dump_expr.register
def _dump_selection(expr: expressions.Selection, ctx: JANIContext) -> JSON:
    ctx.require(ModelFeature.NONDET_EXPRESSIONS)
    return {
        "op": "nondet",
        "var": expr.identifier,
        "exp": _dump_expr(expr.condition, ctx),
    }


checks.check_singledispatch(_dump_expr, model.Expression)


@functools.singledispatch
def _dump_target(target: effects.Target, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"_dump_target has not been implemented for {target}")


@_dump_target.register
def _dump_target_identifier(target: effects.Identifier, ctx: JANIContext) -> JSON:
    return target.identifier


checks.check_singledispatch(_dump_target, effects.Target)


def _dump_var_decl(decl: context.VariableDeclaration, ctx: JANIContext) -> JSON:
    jani_declaration: _JANIMap = {
        "name": decl.identifier,
        "type": _dump_type(decl.typ, ctx),
    }
    if decl.transient is not None:
        jani_declaration["transient"] = decl.transient
    if decl.initial_value is not None:
        jani_declaration["initial-value"] = _dump_expr(decl.initial_value, ctx)
    return jani_declaration


def _dump_const_decl(decl: context.ConstantDeclaration, ctx: JANIContext) -> JSON:
    jani_declaration: _JANIMap = {
        "name": decl.identifier,
        "type": _dump_type(decl.typ, ctx),
    }
    if decl.value is not None:
        jani_declaration["value"] = _dump_expr(decl.value, ctx)
    return jani_declaration


def _dump_assignment(assignment: effects.Assignment, ctx: JANIContext) -> JSON:
    jani_assignment: _JANIMap = {
        "ref": _dump_target(assignment.target, ctx),
        "value": _dump_expr(assignment.value, ctx),
    }
    if assignment.index != 0:
        jani_assignment["index"] = assignment.index
    return jani_assignment


def _dump_location(loc: model.Location, ctx: JANIContext) -> JSON:
    jani_location: _JANIMap = {
        "name": ctx.get_name(loc),
        "x-momba-anonymous": loc.name is None,
    }
    if loc.progress_invariant is not None:
        jani_location["time-progress"] = _dump_expr(loc.progress_invariant, ctx)
    if loc.transient_values is not None:
        jani_location["transient-values"] = [
            _dump_assignment(assignment, ctx) for assignment in loc.transient_values
        ]
    return jani_location


def _dump_destination(dst: model.Destination, ctx: JANIContext) -> JSON:
    jani_destination: _JANIMap = {"location": ctx.get_name(dst.location)}
    if dst.probability is not None:
        jani_destination["probability"] = {"exp": _dump_expr(dst.probability, ctx)}
    if dst.assignments:
        jani_destination["assignments"] = [
            _dump_assignment(assignment, ctx) for assignment in dst.assignments
        ]
    return jani_destination


def _dump_edge(edge: model.Edge, ctx: JANIContext) -> JSON:
    jani_edge: _JANIMap = {
        "location": ctx.get_name(edge.location),
        "destinations": [_dump_destination(dst, ctx) for dst in edge.destinations],
    }
    if edge.action is not None:
        jani_edge["action"] = ctx.require_action(str(edge.action))
    if edge.rate is not None:
        jani_edge["rate"] = {"exp": _dump_expr(edge.rate, ctx)}
    if edge.guard is not None:
        jani_edge["guard"] = {"exp": _dump_expr(edge.guard, ctx)}
    return jani_edge


def _dump_automaton(automaton: model.Automaton, ctx: JANIContext) -> JSON:
    return {
        "name": ctx.get_name(automaton),
        "x-momba-anonymous": automaton.name is None,
        "variables": [
            _dump_var_decl(var_decl, ctx)
            for var_decl in automaton.scope.variable_declarations
        ],
        "locations": [_dump_location(loc, ctx) for loc in automaton.locations],
        "edges": [_dump_edge(edge, ctx) for edge in automaton.edges],
        "initial-locations": [ctx.get_name(loc) for loc in automaton.initial_locations],
    }


def _dump_sync(
    instance_vector: t.Sequence[model.Instance],
    sync: model.Synchronization,
    ctx: JANIContext,
) -> JSON:
    jani_sync: _JANIMap = {
        "synchronise": [
            sync.vector.get(instance, None) for instance in instance_vector
        ],
    }
    if sync.result is not None:
        jani_sync["result"] = sync.result
    return jani_sync


def _dump_system(network: model.Network, ctx: JANIContext) -> JSON:
    instances: t.Set[model.Instance] = set()
    for composition in network.system:
        instances |= composition.instances
    instance_vector = list(instances)
    synchronizations: t.Set[model.Synchronization] = set()
    for composition in network.system:
        synchronizations |= composition.synchronizations
    return {
        "elements": [
            {
                "automaton": ctx.get_name(instance.automaton),
                "input-enabled": list(instance.input_enable),
            }
            for instance in instance_vector
        ],
        "syncs": [
            _dump_sync(instance_vector, synchronization, ctx)
            for synchronization in synchronizations
        ],
    }


def dump_structure(
    network: model.Network, *, allow_momba_operators: bool = False
) -> JSON:
    ctx = JANIContext(allow_momba_operators=allow_momba_operators)
    jani_model: _JANIMap = {
        "jani-version": 1,
        "name": "XXX-momba",  # names are not supported yet
        "type": network.ctx.model_type.abbreviation,
        "variables": [
            _dump_var_decl(var_decl, ctx)
            for var_decl in network.ctx.global_scope.variable_declarations
        ],
        "constants": [
            _dump_const_decl(const_decl, ctx)
            for const_decl in network.ctx.global_scope.constant_declarations
        ],
        "automata": [_dump_automaton(automaton, ctx) for automaton in network.automata],
        "system": _dump_system(network, ctx),
        # important: has to be at the end, because we collect
        # the features and actions while building
        "actions": [{"name": action} for action in ctx.actions],
        "features": [feature.value for feature in ctx.features],
    }
    return jani_model


def dump_model(
    network: model.Network,
    *,
    indent: t.Optional[int] = None,
    allow_momba_operators: bool = False,
) -> bytes:
    """
    Takes a Momba automata network and exports it to the JANI format.

    Arguments:
        network: The Momba automata network to export to JANI.
        indent: Indentation of the final JSON.

    Returns:
        The model in UTF-8 encoded JANI format.
    """
    return json.dumps(
        dump_structure(network, allow_momba_operators=allow_momba_operators),
        indent=indent,
        ensure_ascii=False,
    ).encode("utf-8")
