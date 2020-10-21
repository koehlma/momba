# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum

from . import actions, errors, expressions, properties, types

from .automata import Automaton
from .networks import Network

if t.TYPE_CHECKING:
    from . import effects  # noqa: F401


class ModelType(enum.Enum):
    """ Type of the model. """

    LTS = "Labeled Transition System"
    DTMC = "Discrete-Time Markov Chain"
    CTMC = "Continuous-Time Markov Chain"
    MDP = "Markov Decision Process"
    CTMDP = "Continuous-Time Markov Decision Process"
    MA = "Markov Automaton"
    TA = "Timed Automaton"
    PTA = "Probabilistic Timed Automaton"
    STA = "Stochastic Timed Automaton"
    HA = "Hybrid Automaton"
    PHA = "Probabilistic Timed Automaton"
    SHA = "Stochastic Hybrid Automaton"

    full_name: str

    def __init__(self, full_name: str):
        self.full_name = full_name

    @property
    def is_timed(self) -> bool:
        return self in _TIMED_MODEL_TYPES


_TIMED_MODEL_TYPES = {
    ModelType.TA,
    ModelType.PTA,
    ModelType.STA,
    ModelType.HA,
    ModelType.PHA,
    ModelType.SHA,
}


Typed = t.Union[expressions.Expression, "effects.Target"]


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class Declaration:
    identifier: str
    typ: types.Type

    comment: t.Optional[str] = None

    # XXX: this method shall be implemented by all subclasses
    def validate(self, scope: Scope) -> None:
        raise NotImplementedError()

    def is_constant_in(self, scope: Scope) -> bool:
        return False


@d.dataclass(frozen=True)
class VariableDeclaration(Declaration):
    is_transient: t.Optional[bool] = None
    initial_value: t.Optional[expressions.Expression] = None

    def validate(self, scope: Scope) -> None:
        if self.initial_value is not None:
            if not self.typ.is_assignable_from(scope.get_type(self.initial_value)):
                raise errors.InvalidTypeError(
                    "type of initial value is not assignable to variable type"
                )
            if not self.initial_value.is_constant_in(scope):
                raise errors.NotAConstantError(
                    "initial value is required to be a constant"
                )
        # FIXME: check whether wether an initial value is provided if the
        # variable is transient and not provided via value passing


@d.dataclass(frozen=True)
class ConstantDeclaration(Declaration):
    value: t.Optional[expressions.Expression] = None

    @property
    def is_parameter(self) -> bool:
        return self.value is None

    def validate(self, scope: Scope) -> None:
        if self.value is not None:
            if not self.value.is_constant_in(scope):
                raise errors.NotAConstantError(
                    f"value {self.value} of constant declaration is not a constant"
                )
            if not self.typ.is_assignable_from(scope.get_type(self.value)):
                raise errors.InvalidTypeError(
                    "type of constant value is not assignable to constant type"
                )

    def is_constant_in(self, scope: Scope) -> bool:
        return True


@d.dataclass(frozen=True)
class PropertyDefinition:
    name: str
    prop: properties.Property

    comment: t.Optional[str] = None


class Scope:
    ctx: Context
    parent: t.Optional[Scope]

    _declarations: t.Dict[str, Declaration]
    _cache: t.Dict[Typed, types.Type]

    def __init__(self, ctx: Context, parent: t.Optional[Scope] = None):
        self.ctx = ctx
        self.parent = parent
        self._declarations = {}
        self._cache = {}

    @property
    def declarations(self) -> t.AbstractSet[Declaration]:
        return frozenset(self._declarations.values())

    @property
    def variable_declarations(self) -> t.AbstractSet[VariableDeclaration]:
        return frozenset(
            declaration
            for declaration in self._declarations.values()
            if isinstance(declaration, VariableDeclaration)
        )

    @property
    def constant_declarations(self) -> t.AbstractSet[ConstantDeclaration]:
        return frozenset(
            declaration
            for declaration in self._declarations.values()
            if isinstance(declaration, ConstantDeclaration)
        )

    @property
    def clock_declarations(self) -> t.AbstractSet[VariableDeclaration]:
        """
        Returns the set of declarations for clock variables.
        """
        # FIXME: this does not return declarations with a bounded CLOCK type
        return frozenset(
            declaration
            for declaration in self._declarations.values()
            if (
                isinstance(declaration, VariableDeclaration)
                and declaration.typ == types.CLOCK
            )
        )

    def create_child_scope(self) -> Scope:
        return Scope(self.ctx, parent=self)

    def get_type(self, typed: Typed) -> types.Type:
        if typed not in self._cache:
            inferred_type = typed.infer_type(self)
            inferred_type.validate_in(self)
            self._cache[typed] = inferred_type
        return self._cache[typed]

    def is_constant(self, expression: expressions.Expression) -> bool:
        return expression.is_constant_in(self)

    def is_local(self, identifier: str) -> bool:
        return identifier in self._declarations

    def is_declared(self, identifier: str) -> bool:
        if identifier in self._declarations:
            return True
        if self.parent is not None:
            return self.parent.is_declared(identifier)
        return False

    def get_scope(self, identifier: str) -> Scope:
        if identifier in self._declarations:
            return self
        else:
            raise errors.UnboundIdentifierError(
                f"identifier `{identifier}` is unbound in scope {self}"
            )
            return self.parent.get_scope(identifier)

    def lookup(self, identifier: str) -> Declaration:
        try:
            return self._declarations[identifier]
        except KeyError:
            if self.parent is None:
                raise errors.UnboundIdentifierError(
                    f"identifier `{identifier}` is unbound in scope {self}"
                )
            return self.parent.lookup(identifier)

    def add_declaration(self, declaration: Declaration) -> None:
        if declaration.identifier in self._declarations:
            raise errors.InvalidDeclarationError(
                f"identifier `{declaration.identifier}` has already been declared"
            )
        declaration.validate(self)
        self._declarations[declaration.identifier] = declaration

    def declare_variable(
        self,
        identifier: str,
        typ: types.Type,
        *,
        is_transient: t.Optional[bool] = None,
        initial_value: t.Optional[expressions.ValueOrExpression] = None,
        comment: t.Optional[str] = None,
    ) -> None:
        value = None
        if initial_value is not None:
            value = expressions.ensure_expr(initial_value)
        self.add_declaration(
            VariableDeclaration(
                identifier,
                typ,
                is_transient=is_transient,
                initial_value=value,
                comment=comment,
            )
        )

    def declare_constant(
        self,
        identifier: str,
        typ: types.Type,
        *,
        value: t.Optional[expressions.ValueOrExpression] = None,
        comment: t.Optional[str] = None,
    ) -> None:
        if value is None:
            self.add_declaration(
                ConstantDeclaration(identifier, typ, comment=comment, value=None)
            )
        else:
            self.add_declaration(
                ConstantDeclaration(
                    identifier,
                    typ,
                    comment=comment,
                    value=expressions.ensure_expr(value),
                )
            )


class Context:
    """
    Represents a modeling context.

    Parameters:
        model_type: The model type to use for the context.

    Attributes:
        model_type:
            The type of the model, e.g., SHA, PTA or MDP.
        global_scope:
            The scope for global variables and constants.
        actions:
            A set of actions usable in the context.
        networks:
            Automata networks defined in the modeling context.
        properties:
            Properties defined in the modeling context.
    """

    model_type: ModelType
    global_scope: Scope

    _automata: t.Set[Automaton]
    _networks: t.Set[Network]

    _action_types: t.Dict[str, actions.ActionType]
    _named_properties: t.Dict[str, PropertyDefinition]

    _metadata: t.Dict[str, str]

    def __init__(self, model_type: ModelType = ModelType.SHA) -> None:
        self.model_type = model_type
        self.global_scope = Scope(self)
        self._automata = set()
        self._networks = set()
        self._action_types = {}
        self._named_properties = {}
        self._metadata = {}

    @property
    def automata(self) -> t.AbstractSet[Automaton]:
        return self._automata

    @property
    def networks(self) -> t.AbstractSet[Network]:
        return self._networks

    @property
    def metadata(self) -> t.Mapping[str, str]:
        return self._metadata

    @property
    def action_types(self) -> t.Mapping[str, actions.ActionType]:
        return self._action_types

    @property
    def named_properties(self) -> t.Mapping[str, PropertyDefinition]:
        return self._named_properties

    def update_metadata(self, metadata: t.Mapping[str, str]) -> None:
        self._metadata.update(metadata)

    def get_automaton_by_name(self, name: str) -> Automaton:
        for automaton in self._automata:
            if automaton.name == name:
                return automaton
        raise Exception(f"there is no automaton with name {name}")

    def get_network_by_name(self, name: str) -> Network:
        for network in self._networks:
            if network.name == name:
                return network
        raise Exception("there is no network with name")

    def get_property_by_name(self, name: str) -> properties.Property:
        return self._named_properties[name].prop

    def get_action_type_by_name(self, name: str) -> actions.ActionType:
        return self._action_types[name]

    def add_action_type(self, action_type: actions.ActionType) -> None:
        if action_type.name in self._action_types:
            assert action_type is self._action_types[action_type.name]
        self._action_types[action_type.name] = action_type

    def create_action_type(
        self, name: str, *, parameters: t.Sequence[actions.ActionParameter] = ()
    ) -> actions.ActionType:
        if name in self._action_types:
            raise Exception(f"action with name {name!r} already exists")
        action_type = actions.ActionType(name, tuple(parameters))
        self.add_action_type(action_type)
        return action_type

    def create_automaton(self, *, name: t.Optional[str] = None) -> Automaton:
        automaton = Automaton(self, name=name)
        self._automata.add(automaton)
        return automaton

    def create_network(self, *, name: t.Optional[str] = None) -> Network:
        network = Network(self, name=name)
        self._networks.add(network)
        return network

    def define_property(
        self, name: str, prop: properties.Property
    ) -> PropertyDefinition:
        definition = PropertyDefinition(name, prop)
        self._named_properties[name] = definition
        return definition
