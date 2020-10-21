# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import json
import warnings

from momba import model
from momba.model import (
    effects,
    automata,
    context,
    distributions,
    expressions,
    operators,
    properties,
    types,
)


class InvalidJANIError(Exception):
    pass


_TYPE_MAP = {
    "bool": types.BOOL,
    "int": types.INT,
    "real": types.REAL,
    "clock": types.CLOCK,
    "continuous": types.CONTINUOUS,
}

_BINARY_OP_MAP: t.Mapping[str, expressions.BinaryConstructor] = {
    "∨": expressions.logic_or,
    "∧": expressions.logic_and,
    "⇒": expressions.logic_implies,  # defined by the `derived-operators` extension
    "⊕": expressions.logic_xor,  # defined by the `x-momba-operators` extension
    "⇔": expressions.logic_equiv,  # defined by the `x-momba-operators` extension
    "=": expressions.equals,
    "≠": expressions.not_equals,
    "<": expressions.less_than,
    "≤": expressions.less_or_equal_than,
    ">": expressions.greater_than,  # defined by the `derived-operators` extension
    "≥": expressions.greater_or_equal_than,  # defined by the `derived-operators` extension
    "+": expressions.add,
    "-": expressions.sub,
    "*": expressions.mul,
    "%": expressions.mod,
    "/": expressions.real_div,
    "//": expressions.floor_div,  # defined by the `x-momba-operators`extension
    "pow": expressions.power,
    "log": expressions.log,
}

_UNARY_OP_MAP: t.Mapping[str, expressions.UnaryConstructor] = {
    "¬": expressions.logic_not,
    "floor": expressions.floor,
    "ceil": expressions.ceil,
    "abs": expressions.absolute,
    "trc": expressions.trunc,
    "sgn": expressions.sgn,
}


_AGGREGATION_FUNCTIONS: t.Mapping[str, operators.AggregationFunction] = {
    "min": operators.AggregationFunction.MIN,
    "max": operators.AggregationFunction.MAX,
    "sum": operators.AggregationFunction.SUM,
    "avg": operators.AggregationFunction.AVG,
    "count": operators.AggregationFunction.COUNT,
    "argmin": operators.AggregationFunction.ARGMIN,
    "argmax": operators.AggregationFunction.ARGMAX,
    "values": operators.AggregationFunction.VALUES,
    "∀": operators.AggregationFunction.FORALL,
    "∃": operators.AggregationFunction.EXISTS,
}

_ACCUMULATION_INSTANTS: t.Mapping[str, properties.AccumulationInstant] = {
    "steps": properties.AccumulationInstant.STEPS,
    "time": properties.AccumulationInstant.TIME,
    "exit": properties.AccumulationInstant.EXIT,
}


_STATE_SELECTORS: t.Mapping[str, properties.StateSelector] = {
    "initial": properties.INITIAL_STATES,
    "deadlock": properties.DEADLOCK_STATES,
    "timelock": properties.TIMELOCK_STATES,
}

_MIN_MAX_OPERATORS: t.Mapping[str, operators.MinMax] = {
    "Pmin": operators.MinMax.MIN,
    "Pmax": operators.MinMax.MAX,
    "Emin": operators.MinMax.MIN,
    "Emax": operators.MinMax.MAX,
    "Smin": operators.MinMax.MIN,
    "Smax": operators.MinMax.MAX,
}

_BINARY_PATH_OPERATORS: t.Mapping[str, operators.BinaryPathOperator] = {
    "U": operators.BinaryPathOperator.UNTIL,
    "W": operators.BinaryPathOperator.WEAK_UNTIL,
    "R": operators.BinaryPathOperator.RELEASE,
}


def _expression(jani_expression: t.Any) -> expressions.Expression:
    if isinstance(jani_expression, (float, bool, int)):
        return expressions.ensure_expr(jani_expression)
    elif isinstance(jani_expression, str):
        return expressions.name(jani_expression)
    elif isinstance(jani_expression, dict):
        if "constant" in jani_expression:
            return expressions.ensure_expr(jani_expression["constant"])
        elif "op" in jani_expression:
            op = jani_expression["op"]
            if op in _BINARY_OP_MAP:
                _check_fields(jani_expression, required={"op", "left", "right"})
                left = _expression(jani_expression["left"])
                right = _expression(jani_expression["right"])
                return _BINARY_OP_MAP[op](left, right)
            elif op in _UNARY_OP_MAP:
                _check_fields(jani_expression, required={"op", "exp"})
                operand = _expression(jani_expression["exp"])
                return _UNARY_OP_MAP[op](operand)
            elif op == "ite":
                _check_fields(jani_expression, required={"op", "if", "then", "else"})
                condition = _expression(jani_expression["if"])
                consequence = _expression(jani_expression["then"])
                alternative = _expression(jani_expression["else"])
                return expressions.ite(condition, consequence, alternative)
            elif op == "der":
                _check_fields(jani_expression, required={"op", "var"})
                variable = jani_expression["var"]
                return expressions.Derivative(variable)
        elif "distribution" in jani_expression:
            _check_fields(jani_expression, required={"op", "args", "distribution"})
            arguments = list(map(_expression, jani_expression["args"]))
            distribution = distributions.DistributionType.by_name(
                jani_expression["distribution"]
            )
            return expressions.Sample(distribution, arguments)
    raise InvalidJANIError(f"{jani_expression!r} is not a valid JANI expression")


def _property_interval(jani_property: t.Any) -> properties.Interval:
    _check_fields(
        jani_property, optional={"lower", "lower-exclusive", "upper", "upper-exclusive"}
    )
    lower: t.Optional[expressions.Expression]
    lower_exclusive: t.Optional[expressions.Expression]
    upper: t.Optional[expressions.Expression]
    upper_exclusive: t.Optional[expressions.Expression]
    if "lower" in jani_property:
        lower = _expression(jani_property["lower"])
    else:
        lower = None
    if "lower-exclusive" in jani_property:
        lower_exclusive = _expression(jani_property["lower-exclusive"])
    else:
        lower_exclusive = None
    if "upper" in jani_property:
        upper = _expression(jani_property["upper"])
    else:
        upper = None
    if "upper-exclusive" in jani_property:
        upper_exclusive = _expression(jani_property["upper-exclusive"])
    else:
        upper_exclusive = None
    return properties.Interval(lower, lower_exclusive, upper, upper_exclusive)


def _reward_instant(jani_property: t.Any) -> properties.RewardInstant:
    _check_fields(jani_property, required={"exp", "accumulate", "instant"})
    return properties.RewardInstant(
        _expression(jani_property["exp"]),
        frozenset(_ACCUMULATION_INSTANTS[elem] for elem in jani_property["accumulate"]),
        _expression(jani_property["instant"]),
    )


def _reward_bound(jani_property: t.Any) -> properties.RewardBound:
    _check_fields(jani_property, required={"exp", "accumulate", "bounds"})
    return properties.RewardBound(
        _expression(jani_property["exp"]),
        frozenset(_ACCUMULATION_INSTANTS[elem] for elem in jani_property["accumulate"]),
        _property_interval(jani_property["bounds"]),
    )


def _property(jani_property: t.Any) -> properties.Property:
    try:
        return _expression(jani_property)
    except InvalidJANIError:
        pass
    if "op" not in jani_property:
        raise InvalidJANIError(f"{jani_property!r} is not a valid JANI property")
    if jani_property["op"] == "filter":
        _check_fields(jani_property, required={"op", "fun", "values", "states"})
        return properties.Aggregate(
            _AGGREGATION_FUNCTIONS[jani_property["fun"]],
            _property(jani_property["values"]),
            _property(jani_property["states"]),
        )
    if jani_property["op"] in {"Pmin", "Pmax"}:
        _check_fields(jani_property, required={"op", "exp"})
        return properties.Probability(
            _MIN_MAX_OPERATORS[jani_property["op"]], _property(jani_property["exp"])
        )
    if jani_property["op"] in {"∀", "∃"}:
        _check_fields(jani_property, required={"op", "exp"})
        return properties.PathQuantifier(
            operators.Quantifier.FORALL
            if jani_property["op"] == "∀"
            else operators.Quantifier.EXISTS,
            _property(jani_property["exp"]),
        )
    if jani_property["op"] in {"Emin", "Emax"}:
        _check_fields(
            jani_property,
            required={"op", "exp"},
            optional={
                "accumulate",
                "reach",
                "step-instant",
                "time-instant",
                "reward-instants",
            },
        )
        accumulate: t.Optional[t.FrozenSet[properties.AccumulationInstant]]
        reachability: t.Optional[properties.Property]
        step_instant: t.Optional[expressions.Expression]
        time_instant: t.Optional[expressions.Expression]
        reward_instants: t.Optional[t.Sequence[properties.RewardInstant]]
        if "accumulate" in jani_property:
            accumulate = frozenset(
                _ACCUMULATION_INSTANTS[elem] for elem in jani_property["accumulate"]
            )
        else:
            accumulate = None
        if "reach" in jani_property:
            reachability = _property(jani_property["reach"])
        else:
            reachability = None
        if "step-instant" in jani_property:
            step_instant = _expression(jani_property["step-instant"])
        else:
            step_instant = None
        if "time-instant" in jani_property:
            time_instant = _expression(jani_property["time-instant"])
        else:
            time_instant = None
        if "reward-instants" in jani_property:
            reward_instants = [
                _reward_instant(elem) for elem in jani_property["reward-instants"]
            ]
        else:
            reward_instants = None
        return properties.ExpectedReward(
            operator=_MIN_MAX_OPERATORS[jani_property["op"]],
            reward=_property(jani_property["exp"]),
            accumulate=accumulate,
            reachability=reachability,
            step_instant=step_instant,
            time_instant=time_instant,
            reward_instants=reward_instants,
        )
    if jani_property["op"] in {"Smin", "Smax"}:
        _check_fields(jani_property, required={"op", "exp"}, optional={"accumulate"})
        if "accumulate" in jani_property:
            return properties.SteadyState(
                _MIN_MAX_OPERATORS[jani_property["op"]],
                _property(jani_property["exp"]),
                jani_property["accumulate"],
            )
        else:
            return properties.SteadyState(
                _MIN_MAX_OPERATORS[jani_property["op"]], _property(jani_property["exp"])
            )
    if jani_property["op"] in {"U", "W", "R"}:
        _check_fields(
            jani_property,
            required={"op", "left", "right"},
            optional={"step-bounds", "time-bounds", "reward-bounds"},
        )
        step_bounds: t.Optional[properties.Interval]
        time_bounds: t.Optional[properties.Interval]
        reward_bounds: t.Optional[t.Sequence[properties.RewardBound]]
        time_operator = _BINARY_PATH_OPERATORS[jani_property["op"]]
        if "step-bounds" in jani_property:
            step_bounds = _property_interval(jani_property["step-bounds"])
        else:
            step_bounds = None
        if "time-bounds" in jani_property:
            time_bounds = _property_interval(jani_property["time-bounds"])
        else:
            time_bounds = None
        if "reward-bounds" in jani_property:
            reward_bounds = list(map(_reward_bound, jani_property["reward-bounds"]))
        else:
            reward_bounds = None
        return properties.BinaryPathFormula(
            time_operator,
            _property(jani_property["left"]),
            _property(jani_property["right"]),
            step_bounds,
            time_bounds,
            reward_bounds,
        )
    if jani_property["op"] in _STATE_SELECTORS:
        return _STATE_SELECTORS[jani_property["op"]]
    raise ValueError(f"{jani_property} does not seem to be a valid JANI property")


def _type(typ: t.Any) -> types.Type:
    if isinstance(typ, str):
        return _TYPE_MAP[typ]
    assert isinstance(typ, dict)
    if typ["kind"] == "bounded":
        _check_fields(
            typ,
            required={"kind", "base"},
            optional={"lower-bound", "upper-bound"},
        )
        base = _type(typ["base"])
        assert isinstance(base, types.NumericType)
        lower_bound = _expression(typ["lower-bound"]) if "lower-bound" in typ else None
        upper_bound = _expression(typ["upper-bound"]) if "upper-bound" in typ else None
        return base.bound(lower_bound, upper_bound)
    raise InvalidJANIError(f"{typ!r} is not a valid JANI type")


def _comment_warning(structure: t.Any) -> None:
    if "comment" in structure:
        warnings.warn(
            "comments are currently not supported, comment information will be lost"
        )


def _warn_fields(structure: t.Any, expected: t.Collection[str]) -> None:
    if hasattr(structure, "keys"):
        fields = set(structure.keys())
        for field in expected:
            fields.remove(field)
        if fields:
            for unknown in fields:
                warnings.warn(f"encountered unknown field {unknown} in {structure}")


_Fields = t.Collection[str]


def _check_fields(
    jani_structure: t.Any,
    required: _Fields = frozenset(),
    optional: _Fields = frozenset(),
    unsupported: _Fields = frozenset({"comment"}),
) -> None:
    if not isinstance(jani_structure, dict):
        raise InvalidJANIError(f"expected map but found {jani_structure!r}")
    fields = {field for field in jani_structure.keys() if not field.startswith("x-")}
    for field in unsupported:
        if (field in optional or field in required) and field in fields:
            warnings.warn(
                f"field {field!r} in {jani_structure!r} is currently unsupported",
                stacklevel=2,
            )
    for field in required:
        if field not in fields:
            raise InvalidJANIError(
                f"field {field!r} is required but not found in {jani_structure!r}"
            )
        fields.discard(field)
    for field in optional:
        fields.discard(field)
    if fields:
        warnings.warn(f"unknown fields {fields!r} in {jani_structure!r}", stacklevel=2)


def _variable_declaration(jani_declaration: t.Any) -> context.VariableDeclaration:
    _check_fields(
        jani_declaration,
        required={"name", "type"},
        optional={"transient", "initial-value", "comment"},
    )
    transient: bool = jani_declaration.get("transient", None)
    initial_value: t.Optional[expressions.Expression] = (
        _expression(jani_declaration["initial-value"])
        if "initial-value" in jani_declaration
        else None
    )
    return context.VariableDeclaration(
        identifier=jani_declaration["name"],
        typ=_type(jani_declaration["type"]),
        is_transient=transient,
        initial_value=initial_value,
    )


def _constant_declaration(jani_declaration: t.Any) -> context.ConstantDeclaration:
    _check_fields(
        jani_declaration, required={"name", "type"}, optional={"value", "comment"}
    )
    value: t.Optional[expressions.Expression] = (
        _expression(jani_declaration["value"]) if "value" in jani_declaration else None
    )
    return context.ConstantDeclaration(
        identifier=jani_declaration["name"],
        typ=_type(jani_declaration["type"]),
        value=value,
    )


def _location(jani_location: t.Any) -> automata.Location:
    _check_fields(
        jani_location,
        required={"name"},
        optional={"time-progress", "transient-values", "x-momba-anonymous"},
    )
    progress_invariant: t.Optional[expressions.Expression]
    if "time-progress" in jani_location:
        _check_fields(
            jani_location["time-progress"], required={"exp"}, optional={"comment"}
        )
        progress_invariant = _expression(jani_location["time-progress"]["exp"])
    else:
        progress_invariant = None
    transient_values: t.Set[effects.Assignment] = set()
    if "transient-values" in jani_location:
        for jani_transient_value in jani_location["transient-values"]:
            _check_fields(
                jani_transient_value, required={"ref", "value"}, optional={"comment"}
            )
            assignment = effects.Assignment(
                target=_target(jani_transient_value["ref"]),
                value=_expression(jani_transient_value["value"]),
            )
            transient_values.add(assignment)
    return automata.Location(
        name=None
        if jani_location.get("x-momba-anonymous", False)
        else jani_location["name"],
        progress_invariant=progress_invariant,
        transient_values=frozenset(transient_values),
    )


_Locations = t.Dict[str, automata.Location]


def _target(jani_target: t.Any) -> effects.Target:
    if isinstance(jani_target, str):
        return effects.Name(jani_target)
    raise InvalidJANIError(f"{jani_target!r} is not a valid lvalue")


def _edge(ctx: model.Context, locations: _Locations, jani_edge: t.Any) -> automata.Edge:
    _check_fields(
        jani_edge,
        required={"location", "destinations"},
        optional={"action", "rate", "guard", "comment"},
    )
    location = locations[jani_edge["location"]]
    action_pattern: t.Optional[model.ActionPattern] = None
    if "action" in jani_edge:
        if isinstance(str, jani_edge["action"]):
            action_pattern = model.ActionPattern(
                ctx.get_action_type_by_name(jani_edge["action"])
            )
        else:
            _check_fields(
                jani_edge["action"], required={"name"}, optional={"arguments"}
            )
            action_pattern = model.ActionPattern(
                ctx.get_action_type_by_name(jani_edge["action"]["name"]),
                # FIXME: support import of action patterns
                # identifiers=jani_edge["action"].get("arguments", ()),
            )
    rate = _expression(jani_edge["rate"]["exp"]) if "rate" in jani_edge else None
    guard = _expression(jani_edge["guard"]["exp"]) if "guard" in jani_edge else None
    destinations = frozenset(
        automata.Destination(
            location=locations[jani_destination["location"]],
            probability=(
                _expression(jani_destination["probability"]["exp"])
                if "probability" in jani_destination
                else None
            ),
            assignments=frozenset(
                effects.Assignment(
                    target=_target(jani_assignment["ref"]),
                    value=_expression(jani_assignment["value"]),
                    index=jani_assignment.get("index", 0),
                )
                for jani_assignment in jani_destination["assignments"]
            ),
        )
        for jani_destination in jani_edge["destinations"]
    )
    return automata.Edge(
        location=location,
        destinations=destinations,
        action_pattern=action_pattern,
        guard=guard,
        rate=rate,
    )


def _action_parameter(jani_action_parameter: t.Any) -> model.ActionParameter:
    _check_fields(
        jani_action_parameter,
        required={"type"},
        optional={"comment"},
    )
    typ = _type(jani_action_parameter["type"])
    comment = jani_action_parameter.get("comment", None)
    return model.ActionParameter(typ, comment=comment)


def _action(jani_action: t.Any) -> model.ActionType:
    _check_fields(jani_action, required={"name"}, optional={"comment", "parameters"})
    name = jani_action["name"]
    comment = jani_action.get("comment", None)
    parameters: t.List[model.ActionParameter] = []
    if "parameters" in jani_action:
        for jani_parameter in jani_action["parameters"]:
            parameters.append(_action_parameter(jani_parameter))
    return model.ActionType(name, tuple(parameters), comment=comment)


JANIModel = t.Union[bytes, str]


def load_model(source: JANIModel) -> model.Network:
    """
    Constructs a Momba automata network based on the provided JANI model.

    :param source:
        The source of the JANI model to load.
    :return:
        The resulting network of Momba automata network.
    """
    if isinstance(source, bytes):
        jani_model = json.loads(source.decode("utf-8"))
    else:
        jani_model = json.loads(source)
    _check_fields(
        jani_model,
        required={"jani-version", "name", "type", "automata", "system"},
        optional={
            "metadata",
            "features",
            "actions",
            "constants",
            "variables",
            "restrict-initial",
            "properties",
            "comment",
        },
        unsupported={"comment"},
    )
    network = model.Network(model.Context())
    if "variables" in jani_model:
        for jani_declaration in jani_model["variables"]:
            var_declaration = _variable_declaration(jani_declaration)
            network.ctx.global_scope.add_declaration(var_declaration)
    if "constants" in jani_model:
        for jani_declaration in jani_model["constants"]:
            const_declaration = _constant_declaration(jani_declaration)
            network.ctx.global_scope.add_declaration(const_declaration)
    if "restrict-initial" in jani_model:
        _check_fields(
            jani_model["restrict-initial"], required={"exp"}, optional={"comment"}
        )
        initial_restriction = _expression(jani_model["restrict-initial"]["exp"])
        network.initial_restriction = initial_restriction
    if "actions" in jani_model:
        for jani_action in jani_model["actions"]:
            network.ctx.add_action_type(_action(jani_action))
    for jani_automaton in jani_model["automata"]:
        _check_fields(
            jani_automaton,
            required={"name", "locations", "initial-locations", "edges"},
            optional={"variables", "restrict-initial", "comment", "x-momba-anonymous"},
        )
        name: t.Optional[str]
        if jani_automaton.get("x-momba-anonymous", False):
            name = None
        else:
            name = jani_automaton["name"]
        automaton = network.ctx.create_automaton(name=name)
        if "variables" in jani_automaton:
            for jani_declaration in jani_automaton["variables"]:
                declaration = _variable_declaration(jani_declaration)
                automaton.scope.add_declaration(declaration)
        locations = {
            jani_location["name"]: _location(jani_location)
            for jani_location in jani_automaton["locations"]
        }
        if "restrict-initial" in jani_automaton:
            _check_fields(
                jani_automaton["restrict-initial"],
                required={"exp"},
                optional={"comment"},
            )
            initial_restriction = _expression(jani_automaton["restrict-initial"]["exp"])
            automaton.initial_restriction = initial_restriction
        for jani_edge in jani_automaton["edges"]:
            automaton.add_edge(_edge(network.ctx, locations, jani_edge))
        for jani_location in jani_automaton["initial-locations"]:
            automaton.add_initial_location(locations[jani_location])
    for jani_prop in jani_model["properties"]:
        _check_fields(
            jani_prop,
            required={"expression", "name"},
        )
        network.ctx.define_property(
            jani_prop["name"], _property(jani_prop["expression"])
        )
    return network
