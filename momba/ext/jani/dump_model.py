# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses
import enum
import functools
import json
import warnings

from ... import model
from ...model import effects, context, expressions, operators, properties, types
from ...utils import checks
from ...metadata import version


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
    X_MOMBA_VALUE_PASSING = "x-momba-value-passing"


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
        jani_type["upper-bound"] = _dump_prop(typ.upper_bound, ctx)
    if typ.lower_bound:
        jani_type["lower-bound"] = _dump_prop(typ.lower_bound, ctx)
    return jani_type


@_dump_type.register
def _dump_array_type(typ: types.ArrayType, ctx: JANIContext) -> JSON:
    ctx.require(ModelFeature.ARRAYS)
    return {"kind": "array", "base": _dump_type(typ.base, ctx)}


checks.check_singledispatch(_dump_type, types.Type)


@functools.singledispatch
def _dump_prop(prop: model.Property, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"dump has not been implemented for property {prop}")


@_dump_prop.register
def _dump_identifier(expr: expressions.Identifier, ctx: JANIContext) -> JSON:
    return expr.name


@_dump_prop.register
def _dump_boolean_constant(expr: expressions.BooleanConstant, ctx: JANIContext) -> JSON:
    return expr.boolean


@_dump_prop.register
def _dump_integer_constant(expr: expressions.IntegerConstant, ctx: JANIContext) -> JSON:
    return expr.integer


@_dump_prop.register
def _dump_real_constant(expr: expressions.RealConstant, ctx: JANIContext) -> JSON:
    if isinstance(expr.real, expressions.NamedReal):
        return {"constant": expr.real.symbol}
    if not isinstance(expr.real, float):
        warnings.warn(
            f"imprecise conversion: JSON does not support the number type {type(expr.real)}"
        )
    return float(expr.real)


@_dump_prop.register
def _dump_conditional(expr: expressions.Conditional, ctx: JANIContext) -> JSON:
    return {
        "op": "ite",
        "if": _dump_prop(expr.condition, ctx),
        "then": _dump_prop(expr.consequence, ctx),
        "else": _dump_prop(expr.alternative, ctx),
    }


_DERIVED_OPERATORS = {
    operators.BooleanOperator.IMPLY,
    operators.ComparisonOperator.GT,
    operators.ComparisonOperator.GE,
    operators.ArithmeticOperator.MIN,
    operators.ArithmeticOperator.MAX,
}


_Transform = t.Callable[[expressions.Expression], expressions.Expression]

_MOMBA_OPERATORS: t.Mapping[operators.BinaryOperator, _Transform] = {
    operators.BooleanOperator.XOR: expressions.normalize_xor,
    operators.BooleanOperator.EQUIV: expressions.normalize_equiv,
    operators.ArithmeticOperator.FLOOR_DIV: expressions.normalize_floor_div,
}


@_dump_prop.register
def _dump_binary_expression(
    expr: expressions.BinaryExpression, ctx: JANIContext
) -> JSON:
    if expr.operator in _DERIVED_OPERATORS:
        ctx.require(ModelFeature.DERIVED_OPERATORS)
    if expr.operator in _MOMBA_OPERATORS:
        if ctx.allow_momba_operators:
            ctx.require(ModelFeature.X_MOMBA_OPERATORS)
        else:
            return _dump_prop(_MOMBA_OPERATORS[expr.operator](expr), ctx)
    return {
        "op": expr.operator.symbol,
        "left": _dump_prop(expr.left, ctx),
        "right": _dump_prop(expr.right, ctx),
    }


@_dump_prop.register
def _dump_unary_expression(expr: expressions.UnaryExpression, ctx: JANIContext) -> JSON:
    return {"op": expr.operator.symbol, "exp": _dump_prop(expr.operand, ctx)}


@_dump_prop.register
def _dump_derivative(expr: expressions.Derivative, ctx: JANIContext) -> JSON:
    return {"op": "der", "var": expr.identifier}


@_dump_prop.register
def _dump_sample(expr: expressions.Sample, ctx: JANIContext) -> JSON:
    return {
        "distribution": expr.distribution.jani_name,
        "args": [_dump_prop(argument, ctx) for argument in expr.arguments],
    }


@_dump_prop.register
def _dump_selection(expr: expressions.Selection, ctx: JANIContext) -> JSON:
    ctx.require(ModelFeature.NONDET_EXPRESSIONS)
    return {
        "op": "nondet",
        "var": expr.name,
        "exp": _dump_prop(expr.condition, ctx),
    }


def _dump_prop_interval(pi: properties.PropertyInterval, ctx: JANIContext) -> JSON:
    prop_interval: _JANIMap = {}
    if pi.lower is not None:
        prop_interval["lower"] = _dump_prop(pi.lower, ctx)
    if pi.lower_exclusive is not None:
        prop_interval["lower-exclusive"] = _dump_prop(pi.lower_exclusive, ctx)
    if pi.upper is not None:
        prop_interval["upper"] = _dump_prop(pi.upper, ctx)
    if pi.lower_exclusive is not None:
        prop_interval["upper-exclusive"] = _dump_prop(pi.upper_exclusive, ctx)
    return prop_interval


def _dump_reward_bound(rb: properties.RewardBound, ctx: JANIContext) -> JSON:
    return {
        "exp": _dump_prop(rb.expression, ctx),
        "accumulate": [elem.value for elem in rb.accumulate],
        "bounds": _dump_prop_interval(rb.bounds, ctx),
    }


def _dump_reward_instant(ri: properties.RewardInstant, ctx: JANIContext) -> JSON:
    return {
        "exp": _dump_prop(ri.expression, ctx),
        "accumulate": [elem.value for elem in ri.accumulate],
        "instant": _dump_prop(ri.instant, ctx),
    }


@_dump_prop.register
def _dump_filter(prop: properties.Filter, ctx: JANIContext) -> JSON:
    return {
        "op": "filter",
        "fun": prop.function.value,
        "values": _dump_prop(prop.values, ctx),
        "states": _dump_prop(prop.states, ctx),
    }


@_dump_prop.register
def _dump_probability(prop: properties.ProbabilityProp, ctx: JANIContext) -> JSON:
    return {
        "op": prop.operator.value,
        "exp": _dump_prop(prop.expression, ctx),
    }


@_dump_prop.register
def _dump_path_formula(prop: properties.PathFormula, ctx: JANIContext) -> JSON:
    return {
        "op": prop.operator.value,
        "exp": _dump_prop(prop.expression, ctx),
    }


@_dump_prop.register
def _dump_expected(prop: properties.Expected, ctx: JANIContext) -> JSON:
    expected: _JANIMap = {
        "op": prop.operator.value,
        "exp": _dump_prop(prop.expression, ctx),
    }
    if prop.accumulate is not None:
        expected["accumulate"] = [elem.value for elem in prop.accumulate]
    if prop.reach is not None:
        expected["reach"] = _dump_prop(prop.reach, ctx)
    if prop.step_instant is not None:
        expected["step-instant"] = _dump_prop(prop.step_instant, ctx)
    if prop.time_instant is not None:
        expected["time-instant"] = _dump_prop(prop.time_instant, ctx)
    if prop.reward_instants is not None:
        expected["reward-instants"] = [
            _dump_reward_instant(ri, ctx) for ri in prop.reward_instants
        ]
    return expected


@_dump_prop.register
def _dump_steady(prop: properties.Steady, ctx: JANIContext) -> JSON:
    steady: _JANIMap = {
        "op": prop.operator.value,
        "exp": _dump_prop(prop.expression, ctx),
    }
    if prop.accumulate is not None:
        steady["accumulate"] = [elem.value for elem in prop.accumulate]
    return steady


@_dump_prop.register
def _dump_timed(prop: properties.Timed, ctx: JANIContext) -> JSON:
    timed: _JANIMap = {
        "op": prop.operator.value,
        "left": _dump_prop(prop.left, ctx),
        "right": _dump_prop(prop.right, ctx),
    }
    if prop.step_bounds is not None:
        timed["step-bounds"] = _dump_prop_interval(prop.step_bounds, ctx)
    if prop.time_bounds is not None:
        timed["time-bounds"] = _dump_prop_interval(prop.time_bounds, ctx)
    if prop.reward_bounds is not None:
        timed["reward-bounds"] = [
            _dump_reward_bound(rb, ctx) for rb in prop.reward_bounds
        ]
    return timed


@_dump_prop.register
def _dump_state_predicates(sp: properties.StatePredicates, ctx: JANIContext) -> JSON:
    return {"op": sp.value}


checks.check_singledispatch(_dump_prop, model.Property)


@functools.singledispatch
def _dump_target(target: effects.Target, ctx: JANIContext) -> JSON:
    raise NotImplementedError(f"_dump_target has not been implemented for {target}")


@_dump_target.register
def _dump_target_identifier(target: effects.Identifier, ctx: JANIContext) -> JSON:
    return target.name


checks.check_singledispatch(_dump_target, effects.Target)


def _dump_var_decl(decl: context.VariableDeclaration, ctx: JANIContext) -> JSON:
    jani_declaration: _JANIMap = {
        "name": decl.identifier,
        "type": _dump_type(decl.typ, ctx),
    }
    if decl.is_transient is not None:
        jani_declaration["transient"] = decl.is_transient
    if decl.initial_value is not None:
        jani_declaration["initial-value"] = _dump_prop(decl.initial_value, ctx)
    return jani_declaration


def _dump_const_decl(decl: context.ConstantDeclaration, ctx: JANIContext) -> JSON:
    jani_declaration: _JANIMap = {
        "name": decl.identifier,
        "type": _dump_type(decl.typ, ctx),
    }
    if decl.value is not None:
        jani_declaration["value"] = _dump_prop(decl.value, ctx)
    return jani_declaration


def _dump_assignment(assignment: effects.Assignment, ctx: JANIContext) -> JSON:
    jani_assignment: _JANIMap = {
        "ref": _dump_target(assignment.target, ctx),
        "value": _dump_prop(assignment.value, ctx),
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
        jani_location["time-progress"] = {
            "exp": _dump_prop(loc.progress_invariant, ctx)
        }
    if loc.transient_values is not None:
        jani_location["transient-values"] = [
            _dump_assignment(assignment, ctx) for assignment in loc.transient_values
        ]
    return jani_location


def _dump_destination(dst: model.Destination, ctx: JANIContext) -> JSON:
    jani_destination: _JANIMap = {"location": ctx.get_name(dst.location)}
    if dst.probability is not None:
        jani_destination["probability"] = {"exp": _dump_prop(dst.probability, ctx)}
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
    if edge.action_pattern is not None:
        jani_edge["action"] = _dump_action_pattern(edge.action_pattern, ctx)
    if edge.rate is not None:
        jani_edge["rate"] = {"exp": _dump_prop(edge.rate, ctx)}
    if edge.guard is not None:
        jani_edge["guard"] = {"exp": _dump_prop(edge.guard, ctx)}
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


def _dump_action_pattern(
    pattern: t.Optional[model.ActionPattern], ctx: JANIContext
) -> JSON:
    if pattern is not None:
        if pattern.arguments:
            ctx.require(ModelFeature.X_MOMBA_VALUE_PASSING)
            return {
                "name": pattern.action_type.name,
                "arguments": list(pattern.arguments),
            }
        else:
            return pattern.action_type.name
    return None


def _dump_sync(
    instance_vector: t.Sequence[model.Instance],
    sync: model.Synchronization,
    ctx: JANIContext,
) -> JSON:
    jani_sync: _JANIMap = {
        "synchronise": [
            _dump_action_pattern(sync.vector.get(instance, None), ctx)
            for instance in instance_vector
        ],
    }
    if sync.result is not None:
        jani_sync["result"] = _dump_action_pattern(sync.result, ctx)
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


def _dump_metadata(model_ctx: model.Context) -> _JANIMap:
    jani_metadata: _JANIMap = {}
    for field in {"version", "author", "description", "doi", "url"}:
        try:
            jani_metadata[field] = model_ctx.metadata[field]
        except KeyError:
            pass
    return jani_metadata


def _dump_action_type(action_type: model.ActionType, ctx: JANIContext) -> _JANIMap:
    jani_action: _JANIMap = {"name": action_type.name}
    jani_parameters: t.List[_JANIMap] = []
    for parameter in action_type.parameters:
        ctx.require(ModelFeature.X_MOMBA_VALUE_PASSING)
        jani_parameter: _JANIMap = {"type": _dump_type(parameter.typ, ctx)}
        if parameter.comment is not None:
            jani_parameter["comment"] = parameter.comment
        jani_parameters.append(jani_parameter)
    if jani_parameters:
        jani_action["parameters"] = jani_parameters
    if action_type.comment is not None:
        jani_action["comment"] = action_type.comment
    return jani_action


def dump_structure(
    network: model.Network, *, allow_momba_operators: bool = False
) -> JSON:
    ctx = JANIContext(allow_momba_operators=allow_momba_operators)
    jani_metadata: t.Dict[str, str] = {}
    if "name" in network.ctx.metadata:
        jani_metadata
    jani_model: _JANIMap = {
        "jani-version": 1,
        "x-generator": f"Momba (v{version})",
        "x-momba-release": version,
        "name": network.name or "A Momba Model",
        "x-momba-anonymous": network.name is None,
        "metadata": _dump_metadata(network.ctx),
        "x-momba-metadata": dict(network.ctx.metadata),
        "type": network.ctx.model_type.name.lower(),
        "variables": [
            _dump_var_decl(var_decl, ctx)
            for var_decl in network.ctx.global_scope.variable_declarations
        ],
        "constants": [
            _dump_const_decl(const_decl, ctx)
            for const_decl in network.ctx.global_scope.constant_declarations
        ],
        "actions": [
            _dump_action_type(action_type, ctx)
            for action_type in network.ctx.action_types
        ],
        "automata": [
            _dump_automaton(automaton, ctx) for automaton in network.ctx.automata
        ],
        "properties": [
            {"name": prop.name, "expression": _dump_prop(prop.prop, ctx)}
            for prop in network.ctx.properties
        ],
        "system": _dump_system(network, ctx),
        # important: has to be at the end, because we collect the features while dumping
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
