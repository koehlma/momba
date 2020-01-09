# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

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
    "∨": expressions.lor,
    "∧": expressions.land,
    "⇒": expressions.implies,  # defined by the `derived-operators` extension
    "⊕": expressions.xor,  # defined by the `x-momba-operators` extension
    "⇔": expressions.equiv,  # defined by the `x-momba-operators` extension
    "=": expressions.eq,
    "≠": expressions.neq,
    "<": expressions.lt,
    "≤": expressions.ge,
    ">": expressions.gt,  # defined by the `derived-operators` extension
    "≥": expressions.ge,  # defined by the `derived-operators` extension
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
    "¬": expressions.lnot,
    "floor": expressions.floor,
    "ceil": expressions.ceil,
}


_EXPECTED_OPERATOR_MAP: t.Mapping[str, operators.Expected] = {
    "Emax": operators.Expected.EMAX,
    "Emin": operators.Expected.EMIN,
}


_PROBABILITY_OPERATOR_MAP: t.Mapping[str, operators.Probability] = {
    "Pmax": operators.Probability.PMAX,
    "Pmin": operators.Probability.PMIN,
}

_STEADY_OPERATOR_MAP: t.Mapping[str, operators.Steady] = {
    "Smax": operators.Steady.SMAX,
    "Smin": operators.Steady.SMIN,
}


_PATH_OPERATOR_MAP: t.Mapping[str, operators.PathOperator] = {
    "∀": operators.PathOperator.FORALL,
    "∃": operators.PathOperator.EXISTS,
}


_TIME_OPERATOR_MAP: t.Mapping[str, operators.TimeOperator] = {
    "U": operators.TimeOperator.UNTIL,
    "W": operators.TimeOperator.WEAKU,
}


_FILTER_FUNCTIONS_MAP: t.Mapping[str, operators.FilterFunction] = {
    "min": operators.FilterFunction.MIN,
    "max": operators.FilterFunction.MAX,
    "sum": operators.FilterFunction.SUM,
    "avg": operators.FilterFunction.AVG,
    "count": operators.FilterFunction.COUNT,
    "argmin": operators.FilterFunction.ARGMIN,
    "argmax": operators.FilterFunction.ARGMAX,
    "values": operators.FilterFunction.VALUES,
    "∀": operators.FilterFunction.FORALL,
    "∃": operators.FilterFunction.EXISTS,
}


_REWARD_ACCUMULATION_MAP: t.Mapping[str, properties.RewardAccumulationInstant] = {
    "steps": properties.RewardAccumulationInstant.STEPS,
    "time": properties.RewardAccumulationInstant.TIME,
    "exit": properties.RewardAccumulationInstant.EXIT,
}


_STATE_PREDICATES_MAP: t.Mapping[str, properties.StatePredicates] = {
    "initial": properties.StatePredicates.INITIAL,
    "deadlock": properties.StatePredicates.DEADLOCK,
    "timelock": properties.StatePredicates.TIMELOCK,
}


def _expression(jani_expression: t.Any) -> expressions.Expression:
    if isinstance(jani_expression, (float, bool, int)):
        return expressions.convert(jani_expression)
    elif isinstance(jani_expression, str):
        return expressions.var(jani_expression)
    elif isinstance(jani_expression, dict):
        if "constant" in jani_expression:
            return expressions.convert(jani_expression["constant"])
        elif "op" in jani_expression:
            op = jani_expression["op"]
            if op in _BINARY_OP_MAP:
                left = _expression(jani_expression["left"])
                right = _expression(jani_expression["right"])
                return _BINARY_OP_MAP[op](left, right)
            elif op in _UNARY_OP_MAP:
                operand = _expression(jani_expression["exp"])
                return _UNARY_OP_MAP[op](operand)
            elif op == "ite":
                condition = _expression(jani_expression["if"])
                consequence = _expression(jani_expression["then"])
                alternative = _expression(jani_expression["else"])
                return expressions.ite(condition, consequence, alternative)
            elif op == "der":
                variable = jani_expression["var"]
                return expressions.Derivative(variable)
        elif "distribution" in jani_expression:
            arguments = list(map(_expression, jani_expression["args"]))
            distribution = distributions.by_name(jani_expression["distribution"])
            return expressions.Sample(distribution, arguments)
    raise InvalidJANIError(f"{jani_expression!r} is not a valid JANI expression")


def _property_interval(jani_expression: t.Any) -> properties.PropertyInterval:
    lower: t.Optional[expressions.Expression]
    lower_exclusive: t.Optional[expressions.Expression]
    upper: t.Optional[expressions.Expression]
    upper_exclusive: t.Optional[expressions.Expression]
    if "lower" in jani_expression:
        lower = _expression(jani_expression["lower"])
    else:
        lower = None
    if "lower_exclusive" in jani_expression:
        lower_exclusive = _expression(jani_expression["lower_exclusive"])
    else:
        lower_exclusive = None
    if "upper" in jani_expression:
        upper = _expression(jani_expression["upper"])
    else:
        upper = None
    if "upper_exclusive" in jani_expression:
        upper_exclusive = _expression(jani_expression["upper_exclusive"])
    else:
        upper_exclusive = None
    return properties.PropertyInterval(lower, lower_exclusive, upper, upper_exclusive)


def _reward_instant(jani_expression: t.Any) -> properties.RewardInstant:
    return properties.RewardInstant(
        _expression(jani_expression["exp"]),
        [_REWARD_ACCUMULATION_MAP[elem] for elem in jani_expression["accumulate"]],
        _expression(jani_expression["instant"]),
    )


def _reward_bound(jani_expression: t.Any) -> properties.RewardBound:
    return properties.RewardBound(
        _expression(jani_expression["exp"]),
        [_REWARD_ACCUMULATION_MAP[elem] for elem in jani_expression["accumulate"]],
        _property_interval(jani_expression["bounds"]),
    )


def _property(jani_property: t.Any) -> properties.Property:
    try:
        return _expression(jani_property)
    except InvalidJANIError:
        pass
    _check_fields(jani_property, required={"op"})
    if jani_property["op"] == "filter":
        return properties.Filter(
            _FILTER_FUNCTIONS_MAP[jani_property["fun"]],
            _property(jani_property["values"]),
            _property(jani_property["states"]),
        )
    if jani_property["op"] in _PROBABILITY_OPERATOR_MAP:
        probability_operator = _PROBABILITY_OPERATOR_MAP[jani_property["op"]]
        return properties.ProbabilityProp(
            operator=probability_operator, expression=_property(jani_property["exp"])
        )
    if jani_property["op"] in _PATH_OPERATOR_MAP:
        path_operator = _PATH_OPERATOR_MAP[jani_property["op"]]
        return properties.PathFormula(
            operator=path_operator, expression=_property(jani_property["exp"])
        )
    if jani_property["op"] in _EXPECTED_OPERATOR_MAP:
        accumulate: t.Optional[t.Sequence[properties.RewardAccumulationInstant]]
        reach: t.Optional[properties.Property]
        step_instant: t.Optional[expressions.Expression]
        time_instant: t.Optional[expressions.Expression]
        reward_instants: t.Optional[t.Sequence[properties.RewardInstant]]
        expected_operator = _EXPECTED_OPERATOR_MAP[jani_property["op"]]
        if "accumulate" in jani_property:
            accumulate = [
                _REWARD_ACCUMULATION_MAP[elem] for elem in jani_property["accumulate"]
            ]
        else:
            accumulate = None
        if "reach" in jani_property:
            reach = _property(jani_property["reach"])
        else:
            reach = None
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
        return properties.Expected(
            operator=expected_operator,
            expression=_property(jani_property["exp"]),
            accumulate=accumulate,
            reach=reach,
            step_instant=step_instant,
            time_instant=time_instant,
            reward_instants=reward_instants,
        )
    if jani_property["op"] in _STEADY_OPERATOR_MAP:
        steady_operator = _STEADY_OPERATOR_MAP[jani_property["op"]]
        if "accumulate" in jani_property:
            return properties.Steady(
                operator=steady_operator,
                expression=_property(jani_property["exp"]),
                accumulate=jani_property["accumulate"],
            )
        else:
            return properties.Steady(
                operator=steady_operator, expression=_property(jani_property["exp"])
            )
    if jani_property["op"] in _TIME_OPERATOR_MAP:
        step_bounds: t.Optional[properties.PropertyInterval]
        time_bounds: t.Optional[properties.PropertyInterval]
        reward_bounds: t.Optional[t.Sequence[properties.RewardBound]]
        time_operator = _TIME_OPERATOR_MAP[jani_property["op"]]
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
        return properties.Timed(
            time_operator,
            _property(jani_property["left"]),
            _property(jani_property["right"]),
            step_bounds,
            time_bounds,
            reward_bounds,
        )
    if jani_property["op"] in _STATE_PREDICATES_MAP:
        return _STATE_PREDICATES_MAP[jani_property["op"]]
    raise ValueError(f"{jani_property} does not seem to be a valid JANI property")


def _type(typ: t.Any) -> types.Type:
    if isinstance(typ, str):
        return _TYPE_MAP[typ]
    assert isinstance(typ, dict)
    if typ["kind"] == "bounded":
        base = _type(typ["base"])
        assert isinstance(base, types.Numeric)
        lower_bound = _expression(typ["lower-bound"]) if "lower-bound" in typ else None
        upper_bound = _expression(typ["upper-bound"]) if "upper-bound" in typ else None
        return base[lower_bound, upper_bound]
    assert False, "this should never happen"


def _comment_warning(structure: t.Any) -> None:
    if "comment" in structure:
        warnings.warn(
            f"comments are currently not supported, comment information will be lost"
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
    fields = set(jani_structure.keys())
    for field in unsupported:
        if (field in optional or field in required) and field in fields:
            warnings.warn(
                f"field {field!r} in {jani_structure!r} is currently unsupported"
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
        warnings.warn(f"unknown fields {fields!r} in {jani_structure!r}")


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
        return effects.Identifier(jani_target)
    assert False


def _edge(locations: _Locations, jani_edge: t.Any) -> automata.Edge:
    location = locations[jani_edge["location"]]
    action = jani_edge["action"] if "action" in jani_edge else None
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
        action=action,
        guard=guard,
        rate=rate,
    )


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
            network.ctx.global_scope.declare(var_declaration)
    if "constants" in jani_model:
        for jani_declaration in jani_model["constants"]:
            const_declaration = _constant_declaration(jani_declaration)
            network.ctx.global_scope.declare(const_declaration)
    if "restrict-initial" in jani_model:
        _check_fields(
            jani_model["restrict-initial"], required={"exp"}, optional={"comment"}
        )
        restrict_initial = _expression(jani_model["restrict-initial"]["exp"])
        network.restrict_initial = restrict_initial
    for jani_automaton in jani_model["automata"]:
        _check_fields(
            jani_automaton,
            required={"name", "locations", "initial-locations", "edges"},
            optional={"variables", "restrict-initial", "comment"},
        )
        automaton = network.create_automaton()
        if "variables" in jani_automaton:
            for jani_declaration in jani_automaton["variables"]:
                declaration = _variable_declaration(jani_declaration)
                automaton.scope.declare(declaration)
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
            restrict_initial = _expression(jani_automaton["restrict-initial"]["exp"])
            automaton.restrict_initial = restrict_initial
        for jani_edge in jani_automaton["edges"]:
            automaton.add_edge(_edge(locations, jani_edge))
        for jani_location in jani_automaton["initial-locations"]:
            automaton.add_initial_location(locations[jani_location])
    for jani_prop in jani_model["properties"]:
        _check_fields(
            jani_prop["expression"], required={"op"},
        )
        network.ctx.define_property(
            _property(jani_prop["expression"]), name=jani_prop["name"]
        )
    return network
