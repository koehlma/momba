# -*- coding:utf-8 -*-
#
# Copyright (C) 2019, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses

from . import lexer
from .. import model
from ..model import expressions, types, effects


_IGNORE = {lexer.TokenType.WHITESPACE, lexer.TokenType.COMMENT}


class MomlSyntaxError(ValueError):
    pass


Expected = t.Union[
    lexer.TokenType, str, t.AbstractSet[lexer.TokenType], t.AbstractSet[str]
]


_PRIMITIVE_TYPES = {
    "bool": types.BOOL,
    "int": types.INT,
    "real": types.REAL,
    "clock": types.CLOCK,
    "continuous": types.CONTINUOUS,
}


class TokenStream:
    code: str
    iterator: t.Iterator[lexer.Token]

    _current_token: t.Optional[lexer.Token]
    _next_token: t.Optional[lexer.Token]

    def __init__(self, code: str) -> None:
        self.code = code
        self.iterator = lexer.lex(code)
        self._current_token = self._forward()
        self._next_token = self._forward()

    def _forward(self) -> t.Optional[lexer.Token]:
        try:
            token = next(self.iterator)
            while token.token_type in _IGNORE:
                token = next(self.iterator)
            return token
        except StopIteration:
            return None

    def consume(self) -> lexer.Token:
        if self._current_token is None:
            raise ValueError(f"no token to consume")
        token = self._current_token
        self._current_token = self._next_token
        self._next_token = self._forward()
        return token

    def expect(self, expected: Expected, *, consume: bool = True) -> lexer.Token:
        token = self.accept(expected, consume=consume)
        if token is None:
            raise self.make_error(f"expected {expected}")
        return token

    def accept(
        self, expected: Expected, *, consume: bool = True
    ) -> t.Optional[lexer.Token]:
        current_token = self._current_token
        if current_token is None:
            return None
        if isinstance(expected, lexer.TokenType):
            if current_token.token_type is not expected:
                return None
        elif isinstance(expected, str):
            if current_token.text != expected:
                return None
        elif (
            current_token.token_type not in expected
            and current_token.text not in expected
        ):
            return None
        if consume:
            return self.consume()
        else:
            return current_token

    def check(self, expected: Expected) -> bool:
        return bool(self.accept(expected, consume=False))

    @property
    def finished_parsing(self) -> bool:
        return self._current_token is None

    @property
    def current_token(self) -> lexer.Token:
        if self._current_token is None:
            raise self.make_error(f"expected token but found EOF")
        return self._current_token

    def make_error(self, message: str) -> MomlSyntaxError:
        start_row = max(0, self.current_token.start_row - 3)
        end_row = self.current_token.start_row + 1
        rows = "\n".join(self.code.split("\n")[start_row:end_row])
        return MomlSyntaxError(
            f"{message} at {self.current_token.start_row}:{self.current_token.start_column} "
            f"(token: `{self.current_token.text}`)\n\n"
            f"{rows}\n"
            f"{' ' * self.current_token.start_column}^"
        )


@dataclasses.dataclass
class Context:
    stream: TokenStream
    scope: model.Scope
    automaton: t.Optional[model.Automaton]


def parse_primitive_type(stream: TokenStream) -> types.Type:
    return _PRIMITIVE_TYPES[stream.expect(_PRIMITIVE_TYPES.keys()).text]


def parse_type(stream: TokenStream) -> types.Type:
    typ = parse_primitive_type(stream)
    while stream.accept("["):
        if stream.accept("]"):
            typ = types.ArrayType(typ)
        else:
            if not isinstance(typ, types.Numeric):
                raise stream.make_error(
                    f"{typ} is not numeric; only numeric types can be bounded"
                )
            lower_bound = parse_expression(stream)
            stream.expect(",")
            upper_bound = parse_expression(stream)
            stream.expect("]")
            typ = types.BoundedType(typ, lower_bound, upper_bound)
    return typ


_PRECEDENCE = {
    lexer.TokenType.STAR: 31,
    lexer.TokenType.SLASH: 31,
    lexer.TokenType.SLASH_SLASH: 31,
    lexer.TokenType.PERCENTAGE: 31,
    lexer.TokenType.PLUS: 30,
    lexer.TokenType.MINUS: 30,
    lexer.TokenType.COMP_EQ: 20,
    lexer.TokenType.COMP_NEQ: 20,
    lexer.TokenType.COMP_LT: 20,
    lexer.TokenType.COMP_LE: 20,
    lexer.TokenType.COMP_GE: 20,
    lexer.TokenType.COMP_GT: 20,
    lexer.TokenType.LOGIC_AND: 12,
    lexer.TokenType.LOGIC_OR: 11,
    lexer.TokenType.LOGIC_XOR: 11,
    lexer.TokenType.LOGIC_IMPLIES: 10,
    lexer.TokenType.LOGIC_RIMPLIES: 10,
    lexer.TokenType.LOGIC_EQUIV: 10,
    lexer.TokenType.QUESTIONMARK: 5,
}

_RIGHT_ASSOCIATIVE = {lexer.TokenType.LOGIC_IMPLIES}

_BINARY_CONSTRUCTORS: t.Mapping[lexer.TokenType, model.BinaryConstructor] = {
    lexer.TokenType.STAR: expressions.mul,
    lexer.TokenType.SLASH: expressions.real_div,
    lexer.TokenType.SLASH_SLASH: expressions.floor_div,
    lexer.TokenType.PERCENTAGE: expressions.mod,
    lexer.TokenType.PLUS: expressions.add,
    lexer.TokenType.MINUS: expressions.sub,
    lexer.TokenType.COMP_EQ: expressions.eq,
    lexer.TokenType.COMP_NEQ: expressions.neq,
    lexer.TokenType.COMP_LT: expressions.lt,
    lexer.TokenType.COMP_LE: expressions.le,
    lexer.TokenType.COMP_GE: expressions.ge,
    lexer.TokenType.COMP_GT: expressions.gt,
    lexer.TokenType.LOGIC_AND: expressions.logic_and,
    lexer.TokenType.LOGIC_OR: expressions.logic_or,
    lexer.TokenType.LOGIC_XOR: expressions.logic_xor,
    lexer.TokenType.LOGIC_IMPLIES: expressions.logic_implies,
    lexer.TokenType.LOGIC_RIMPLIES: expressions.logic_rimplies,
    lexer.TokenType.LOGIC_EQUIV: expressions.logic_equiv,
}


_BuiltinFunctionConstructor = t.Callable[[t.List[model.Expression]], model.Expression]


def _construct_floor(arguments: t.List[model.Expression]) -> model.Expression:
    if len(arguments) != 1:
        raise Exception(
            f"floor takes exactly 1 argument but {len(arguments)} are given"
        )
    return expressions.floor(arguments[0])


_BUILTIN_FUNCTIONS: t.Mapping[str, _BuiltinFunctionConstructor] = {
    "floor": _construct_floor
}


def _parse_primary_expression(stream: TokenStream) -> model.Expression:
    if stream.accept(lexer.TokenType.INTEGER, consume=False):
        return model.convert(int(stream.consume().text))
    elif stream.accept(lexer.TokenType.REAL, consume=False):
        return model.convert(float(stream.consume().text))
    elif stream.accept(lexer.TokenType.IDENTIFIER, consume=False):
        name = stream.consume().text
        if stream.accept(lexer.TokenType.LEFT_PAR):
            arguments: t.List[model.Expression] = []
            if not stream.accept(lexer.TokenType.RIGHT_PAR):
                arguments.append(parse_expression(stream))
                while not stream.accept(lexer.TokenType.RIGHT_PAR):
                    stream.expect(",")
                    arguments.append(parse_expression(stream))
            if name not in _BUILTIN_FUNCTIONS:
                raise Exception(f"unknown function {name}")
            return _BUILTIN_FUNCTIONS[name](arguments)
        else:
            return model.identifier(name)
    elif stream.accept(lexer.TokenType.LEFT_PAR):
        expr = parse_expression(stream)
        stream.expect(lexer.TokenType.RIGHT_PAR)
        return expr
    else:
        raise stream.make_error(f"expected primary expression")


def _parse_unary_expression(stream: TokenStream) -> model.Expression:
    if stream.accept(lexer.TokenType.LOGIC_NOT):
        return model.logic_not(_parse_unary_expression(stream))
    return _parse_primary_expression(stream)


def parse_expression(
    stream: TokenStream, *, min_precedence: int = 0
) -> model.Expression:
    left = _parse_unary_expression(stream)
    while _PRECEDENCE.get(stream.current_token.token_type, -1) >= min_precedence:
        operator = stream.consume()
        precedence = _PRECEDENCE[operator.token_type]
        if operator.token_type is lexer.TokenType.QUESTIONMARK:
            consequence = parse_expression(stream)
            stream.expect(":")
            alternative = parse_expression(stream, min_precedence=precedence)
            left = model.ite(left, consequence, alternative)
        else:
            if operator.token_type not in _RIGHT_ASSOCIATIVE:
                precedence += 1
            right = parse_expression(stream, min_precedence=precedence)
            left = _BINARY_CONSTRUCTORS[operator.token_type](left, right)
    return left


@dataclasses.dataclass(frozen=True)
class _DeclarationInfo:
    name: str
    typ: types.Type
    value: t.Optional[model.Expression] = None


def _parse_identifier_declaration(stream: TokenStream) -> _DeclarationInfo:
    name = stream.expect(lexer.TokenType.IDENTIFIER).text
    stream.expect(":")
    typ = parse_type(stream)
    if stream.accept(":="):
        return _DeclarationInfo(name, typ, parse_expression(stream))
    else:
        return _DeclarationInfo(name, typ)


def _parse_constant_declaration(stream: TokenStream) -> model.ConstantDeclaration:
    stream.expect("constant")
    info = _parse_identifier_declaration(stream)
    stream.accept(lexer.TokenType.STRING)  # TODO: comment
    return model.ConstantDeclaration(info.name, info.typ, info.value)


def _parse_variable_declaration(stream: TokenStream) -> model.VariableDeclaration:
    is_transient = bool(stream.accept("transient"))
    stream.expect("variable")
    info = _parse_identifier_declaration(stream)
    stream.accept(lexer.TokenType.STRING)  # TODO: comment
    return model.VariableDeclaration(
        info.name, info.typ, is_transient=is_transient, initial_value=info.value,
    )


def _parse_assignment(stream: TokenStream) -> model.Assignment:
    name = stream.expect(lexer.TokenType.IDENTIFIER).text
    stream.expect(":=")
    value = parse_expression(stream)
    return model.Assignment(effects.Identifier(name), value=value)


def _parse_location(stream: TokenStream, automaton: model.Automaton) -> model.Location:
    is_initial = bool(stream.accept("initial"))
    stream.expect("location")
    name = stream.expect(lexer.TokenType.IDENTIFIER).text
    progress_invariant: t.Optional[model.Expression] = None
    transient_values: t.Set[model.Assignment] = set()
    if stream.accept(":"):
        stream.expect(lexer.TokenType.INDENT)
        while not stream.accept(lexer.TokenType.DEDENT):
            if stream.accept("invariant"):
                if progress_invariant is not None:
                    raise stream.make_error(f"duplicate definition of invariant")
                progress_invariant = parse_expression(stream)
            elif stream.check(lexer.TokenType.IDENTIFIER):
                transient_values.add(_parse_assignment(stream))
            else:
                raise stream.make_error(f"expected location body")
    return automaton.create_location(
        name,
        progress_invariant=progress_invariant,
        transient_values=transient_values,
        initial=is_initial,
    )


def parse_automaton(stream: TokenStream, ctx: model.Context) -> model.Automaton:
    stream.expect("automaton")
    name = stream.expect(lexer.TokenType.IDENTIFIER).text
    stream.expect(":")
    stream.expect(lexer.TokenType.INDENT)
    automaton = ctx.create_automaton(name=name)
    location_map: t.Dict[str, model.Location] = {}
    while not stream.accept(lexer.TokenType.DEDENT):
        if stream.check({"transient", "variable"}):
            declaration = _parse_variable_declaration(stream)
            automaton.scope.declare(declaration)
        elif stream.check({"initial", "location"}):
            location = _parse_location(stream, automaton)
            assert location.name not in location_map and location.name is not None
            location_map[location.name] = location
        elif stream.accept("edge"):
            stream.expect("from")
            location_name = stream.expect(lexer.TokenType.IDENTIFIER).text
            stream.expect(":")
            stream.expect(lexer.TokenType.INDENT)
            action: t.Optional[str] = None
            guard: t.Optional[model.Expression] = None
            rate: t.Optional[model.Expression] = None
            destinations: t.Set[model.Destination] = set()
            while not stream.accept(lexer.TokenType.DEDENT):
                if stream.accept("action"):
                    assert action is None
                    action = stream.expect(lexer.TokenType.IDENTIFIER).text
                elif stream.accept("guard"):
                    assert guard is None
                    guard = parse_expression(stream)
                elif stream.accept("rate"):
                    assert rate is None
                    rate = parse_expression(stream)
                elif stream.accept("to"):
                    target_name = stream.expect(lexer.TokenType.IDENTIFIER).text
                    probability: t.Optional[model.Expression] = None
                    assignments: t.Set[model.Assignment] = set()
                    if stream.accept(":"):
                        stream.expect(lexer.TokenType.INDENT)
                        while not stream.accept(lexer.TokenType.DEDENT):
                            if stream.accept("probability"):
                                assert probability is None
                                probability = parse_expression(stream)
                            elif stream.check(lexer.TokenType.IDENTIFIER):
                                assignments.add(_parse_assignment(stream))
                            else:
                                raise stream.make_error("unexpected token")
                    destinations.add(
                        model.Destination(
                            location=location_map[target_name],
                            probability=probability,
                            assignments=frozenset(assignments),
                        )
                    )
                else:
                    raise stream.make_error("unexpected token")
            automaton.create_edge(
                location_map[location_name],
                frozenset(destinations),
                action=action,
                guard=guard,
                rate=rate,
            )
        else:
            raise stream.make_error("unexpected token")
    return automaton


def _parse_action(stream: TokenStream) -> t.Optional[str]:
    if stream.accept("-"):
        return None
    return stream.expect(lexer.TokenType.IDENTIFIER).text


def _parse_network(stream: TokenStream, ctx: model.Context) -> None:
    stream.expect("network")
    identifier = stream.accept(lexer.TokenType.IDENTIFIER)
    name: t.Optional[str] = None
    if identifier:
        name = identifier.text
    network = ctx.create_network(name=name)
    stream.expect(":")
    stream.expect(lexer.TokenType.INDENT)
    instance_map: t.Dict[str, model.Instance] = {}
    while not stream.accept(lexer.TokenType.DEDENT):
        if stream.accept("restrict"):
            stream.expect("initial")
            network.restrict_initial = parse_expression(stream)
        elif stream.accept("instance"):
            name = stream.expect(lexer.TokenType.IDENTIFIER).text
            assert name not in instance_map
            automaton_name = stream.expect(lexer.TokenType.IDENTIFIER).text
            input_enable: t.Set[str] = set()
            if stream.accept(":"):
                stream.expect(lexer.TokenType.INDENT)
                if stream.accept("input"):
                    stream.expect("enable")
                    input_enable.add(stream.expect(lexer.TokenType.IDENTIFIER).text)
                    while stream.accept(","):
                        input_enable.add(stream.expect(lexer.TokenType.IDENTIFIER).text)
                stream.expect(lexer.TokenType.DEDENT)
            instance_map[name] = ctx.get_automaton_by_name(
                automaton_name
            ).create_instance(input_enable=input_enable)
        elif stream.accept("composition"):
            instances: t.List[model.Instance] = []
            instances.append(
                instance_map[stream.expect(lexer.TokenType.IDENTIFIER).text]
            )
            while stream.accept("|"):
                instances.append(
                    instance_map[stream.expect(lexer.TokenType.IDENTIFIER).text]
                )
            composition = network.create_composition(frozenset(instances))
            if stream.accept(":"):
                stream.expect(lexer.TokenType.INDENT)
                while not stream.accept(lexer.TokenType.DEDENT):
                    stream.expect("synchronize")
                    actions: t.List[t.Optional[str]] = []
                    actions.append(_parse_action(stream))
                    while stream.accept("|"):
                        actions.append(_parse_action(stream))
                    assert len(actions) == len(instances)
                    stream.expect(lexer.TokenType.ARROW)
                    result = _parse_action(stream)
                    vector: t.Dict[model.Instance, str] = {}
                    for index, action in enumerate(actions):
                        if action is not None:
                            vector[instances[index]] = action
                    composition.create_synchronization(vector, result)


def _parse_metadata(stream: TokenStream) -> t.Mapping[str, str]:
    stream.expect("metadata")
    stream.expect(":")
    stream.expect(lexer.TokenType.INDENT)
    fields: t.Dict[str, str] = {}
    while not stream.accept(lexer.TokenType.DEDENT):
        field_name = stream.expect(lexer.TokenType.STRING).match["string"]
        stream.expect(":")
        field_value = stream.expect(lexer.TokenType.STRING).match["string"]
        fields[field_name] = field_value
    return fields


def _parse_moml(stream: TokenStream, ctx: model.Context) -> model.Context:
    automaton_map: t.Dict[str, model.Automaton] = {}
    while True:
        if stream.check({"transient", "variable"}):
            ctx.global_scope.declare(_parse_variable_declaration(stream))
        elif stream.check("constant"):
            ctx.global_scope.declare(_parse_constant_declaration(stream))
        elif stream.check("automaton"):
            automaton = parse_automaton(stream, ctx)
            assert automaton.name not in automaton_map and automaton.name is not None
            automaton_map[automaton.name] = automaton
        elif stream.check("network"):
            _parse_network(stream, ctx)
        elif stream.check(lexer.TokenType.END_OF_FILE):
            break
        elif stream.check("metadata"):
            # TODO: integrate this somehow into the modeling context
            fields = _parse_metadata(stream)  # noqa
        else:
            raise stream.make_error("unexpected token")
    return ctx


def parse_moml(
    stream: TokenStream, *, ctx: t.Optional[model.Context] = None
) -> model.Context:
    if stream.accept("model_type"):
        model_type = model.ModelType[stream.expect(lexer.TokenType.IDENTIFIER).text]
    return _parse_moml(stream, ctx or model.Context(model_type))
