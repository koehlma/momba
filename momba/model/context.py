# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <mkoehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import dataclasses
import enum

from . import action, errors, expressions, types
from .automata import Automaton
from .network import Network
from .properties import Property

if t.TYPE_CHECKING:
    # XXX: stupid stuff to make mypy and the linter happy
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


TA_MODEL_TYPES = {
    ModelType.TA,
    ModelType.PTA,
    ModelType.STA,
    ModelType.HA,
    ModelType.PHA,
    ModelType.SHA,
}

Typed = t.Union["expressions.Expression", "effects.Target"]


@dataclasses.dataclass(frozen=True)
class Declaration:
    identifier: str
    typ: types.Type

    comment: t.Optional[str] = None

    def validate(self, scope: Scope) -> None:
        # TODO: check whether type is bounded or basic
        pass

    def is_constant_in(self, scope: Scope) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
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
                raise errors.NotAConstantError("initial value must be a constant")


@dataclasses.dataclass(frozen=True)
class ConstantDeclaration(Declaration):
    """ Constants without values are parameters. """

    value: t.Optional[expressions.Expression] = None

    def validate(self, scope: Scope) -> None:
        if self.value is not None:
            if not self.value.is_constant_in(scope):
                raise errors.NotAConstantError(
                    f"value {self.value} of constant declaration is not a constant"
                )
            if not self.typ.is_assignable_from(scope.get_type(self.value)):
                raise errors.InvalidTypeError(
                    "constant expression is not assignable to constant type"
                )

    def is_constant_in(self, scope: Scope) -> bool:
        return True


class PropertyDefinition:
    _name: t.Optional[str]
    _prop: Property

    def __init__(self, prop: Property, *, name: t.Optional[str] = None) -> None:
        self._name = name
        self._prop = prop

    @property
    def name(self) -> t.Optional[str]:
        return self._name

    @property
    def prop(self) -> Property:
        return self._prop


class Scope:
    ctx: Context
    parent: t.Optional[Scope]

    _declarations: t.Dict[str, Declaration]
    _types: t.Dict[Typed, types.Type]

    def __init__(self, ctx: Context, parent: t.Optional[Scope] = None):
        self.ctx = ctx
        self.parent = parent
        self._declarations = {}
        self._types = {}

    @property
    def declarations(self) -> t.AbstractSet[Declaration]:
        return frozenset(self._declarations.values())

    @property
    def variable_declarations(self) -> t.AbstractSet[VariableDeclaration]:
        return frozenset(
            decl
            for decl in self._declarations.values()
            if isinstance(decl, VariableDeclaration)
        )

    @property
    def constant_declarations(self) -> t.AbstractSet[ConstantDeclaration]:
        return frozenset(
            decl
            for decl in self._declarations.values()
            if isinstance(decl, ConstantDeclaration)
        )

    @property
    def clock_declarations(self) -> t.AbstractSet[VariableDeclaration]:
        """
        Returns the set of declarations for clock variables.
        """
        return frozenset(
            decl
            for decl in self._declarations.values()
            if isinstance(decl, VariableDeclaration) and decl.typ == types.CLOCK
        )

    def create_child_scope(self) -> Scope:
        return Scope(self.ctx, parent=self)

    def get_type(self, typed: Typed) -> types.Type:
        if typed not in self._types:
            inferred_type = typed.infer_type(self)
            inferred_type.validate_in(self)
            self._types[typed] = inferred_type
        return self._types[typed]

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

    def lookup(self, identifier: str) -> Declaration:
        try:
            return self._declarations[identifier]
        except KeyError:
            if self.parent is None:
                raise errors.UnboundIdentifierError(
                    f"identifier {identifier} is unbound in scope {self}"
                )
            return self.parent.lookup(identifier)

    def add_declaration(self, declaration: Declaration) -> None:
        if declaration.identifier in self._declarations:
            raise errors.InvalidDeclarationError(
                f"identifier `{declaration.identifier} has already been declared"
            )
        declaration.validate(self)
        self._declarations[declaration.identifier] = declaration

    def declare(self, identifier: str, typ: types.Type) -> None:
        self.add_declaration(Declaration(identifier, typ))

    def declare_variable(
        self,
        identifier: str,
        typ: types.Type,
        *,
        is_transient: t.Optional[bool] = None,
        initial_value: t.Optional[expressions.Expression] = None,
    ) -> None:
        self.add_declaration(
            VariableDeclaration(
                identifier, typ, is_transient=is_transient, initial_value=initial_value
            )
        )

    def declare_constant(
        self,
        identifier: str,
        typ: types.Type,
        *,
        value: t.Optional[expressions.MaybeExpression] = None,
        comment: t.Optional[str] = None,
    ) -> None:
        """
        Declare a constant in the scope.

        Parameters:
            identifier (str):
                The name of the constant to declare.
            typ (types.Type):
                The type of the constant.
            value:
                The value of the constant. If none is provided, the constant becomes
                a parameter of the model.
            comment:
                An optional comment describing the constant.
        """
        if value is None:
            self.add_declaration(
                ConstantDeclaration(identifier, typ, comment=comment, value=value)
            )
        else:
            self.add_declaration(
                ConstantDeclaration(
                    identifier, typ, comment=comment, value=expressions.convert(value)
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

    _action_types: t.Dict[str, action.ActionType]
    _automata: t.Set[Automaton]
    _networks: t.Set[Network]
    _properties: t.Set[PropertyDefinition]

    _metadata: t.Dict[str, str]

    def __init__(self, model_type: ModelType = ModelType.SHA) -> None:
        self.model_type = model_type
        self.global_scope = Scope(self)
        self._action_types = {}
        self._automata = set()
        self._networks = set()
        self._properties = set()
        self._metadata = {}

    @property
    def automata(self) -> t.AbstractSet[Automaton]:
        """ The automata defined in the modeling context. """
        return self._automata

    @property
    def networks(self) -> t.AbstractSet[Network]:
        return self._networks

    @property
    def properties(self) -> t.AbstractSet[PropertyDefinition]:
        return self._properties

    @property
    def metadata(self) -> t.Mapping[str, str]:
        return self._metadata

    def update_metadata(self, metadata: t.Mapping[str, str]) -> None:
        self._metadata.update(metadata)

    def get_automaton_by_name(self, name: str) -> Automaton:
        for automaton in self._automata:
            if automaton.name == name:
                return automaton
        raise Exception(f"there is no automaton with name {name}")

    def get_action_type_by_name(self, name: str) -> action.ActionType:
        return self._action_types[name]

    def add_action_type(self, action: action.ActionType) -> None:
        if action.name in self._action_types:
            assert action is self._action_types[action.name]
        self._action_types[action.name] = action

    def create_action_type(
        self, name: str, *, parameters: t.Sequence[action.ActionParameter] = ()
    ) -> action.ActionType:
        if name in self._action_types:
            raise Exception(f"action with name {name!r} already exists")
        action_type = action.ActionType(name, tuple(parameters))
        self.add_action_type(action_type)
        return action_type

    @property
    def action_types(self) -> t.ValuesView[action.ActionType]:
        return self._action_types.values()

    def create_automaton(self, *, name: t.Optional[str] = None) -> Automaton:
        automaton = Automaton(self, name=name)
        self._automata.add(automaton)
        return automaton

    def create_network(self, *, name: t.Optional[str] = None) -> Network:
        network = Network(self, name=name)
        self._networks.add(network)
        return network

    def define_property(
        self, prop: Property, *, name: t.Optional[str] = None
    ) -> PropertyDefinition:
        property_definition = PropertyDefinition(name=name, prop=prop)
        self._properties.add(property_definition)
        return property_definition
