# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import collections
import enum

from . import actions, errors, functions, expressions, types

from .automata import Automaton
from .networks import Network


class ModelType(enum.Enum):
    """
    An enum representing different *model types*.

    Attributes
    ----------
    full_name: str
        The full human-readable name of the model type.
    """

    LTS = "Labeled Transition System"
    """ Labeled Transition System """

    DTMC = "Discrete-Time Markov Chain"
    """ Discrete-Time Markov Chain """

    CTMC = "Continuous-Time Markov Chain"
    """ Continuous-Time Markov Chain """

    MDP = "Markov Decision Process"
    """ Markov Decision Process """

    CTMDP = "Continuous-Time Markov Decision Process"
    """ Continuous-Time Markov Decision Process """

    MA = "Markov Automaton"
    """ Markov Automaton """

    TA = "Timed Automaton"
    """ Timed Automaton """

    PTA = "Probabilistic Timed Automaton"
    """ Probabilistic Timed Automaton """

    STA = "Stochastic Timed Automaton"
    """ Stochastic Timed Automaton """

    HA = "Hybrid Automaton"
    """ Hybrid Automaton """

    PHA = "Probabilistic Hybrid Automaton"
    """ Probabilistic Hybrid Automaton """

    SHA = "Stochastic Hybrid Automaton"
    """ Stochastic Hybrid Automaton """

    full_name: str

    def __init__(self, full_name: str):
        self.full_name = full_name

    @property
    def uses_clocks(self) -> bool:
        """
        Returns :obj:`True` if and only if the respective models use real-value clocks.
        """
        return self in _CLOCK_TYPES

    @property
    def is_untimed(self) -> bool:
        """
        Returns :obj:`True` if and only if the model type is *not timed*.

        Untimed model types are :code:`LTS`, :code:`DTMC`, and :code:`MDP`.
        """
        return self in _UNTIMED_TYPES


_CLOCK_TYPES = {
    ModelType.TA,
    ModelType.PTA,
    ModelType.STA,
    ModelType.HA,
    ModelType.PHA,
    ModelType.SHA,
}

_UNTIMED_TYPES = {ModelType.MDP, ModelType.LTS, ModelType.DTMC}


# XXX: this class should be abstract, however, then it would not type-check
# https://github.com/python/mypy/issues/5374
@d.dataclass(frozen=True)
class IdentifierDeclaration:
    """
    Represents a declaration of an identifier.

    Attributes
    ----------
    identifier:
        The declared identifier.
    typ:
        The type of the identifier.
    comment:
        An additional optional comment for the declaration.
    """

    identifier: str
    typ: types.Type

    comment: t.Optional[str] = None

    # XXX: this method shall be implemented by all subclasses
    def validate(self, scope: Scope) -> None:
        """
        Validates that the declaration is valid in the given scope.

        Raises :class:`~errors.ModelingError` if the declaration is invalid.
        """
        raise NotImplementedError()


@d.dataclass(frozen=True)
class VariableDeclaration(IdentifierDeclaration):
    """
    Represents a *variable declaration*.

    Attributes
    ----------
    is_transient:
        Optional boolean flag indicating whether the variable is *transient*.
    initial_value:
        Optional :class:`~momba.model.Expression` providing an initial value for the variable.
    """

    is_transient: t.Optional[bool] = None
    initial_value: t.Optional[expressions.Expression] = None

    def __post_init__(self) -> None:
        # if self.is_transient and self.initial_value is None:
        #     raise errors.ModelingError(
        #         "transient variables must have an initial value", self
        #     )
        pass

    def validate(self, scope: Scope) -> None:
        if self.initial_value is not None:
            value_type = scope.get_type(self.initial_value)
            if not self.typ.is_assignable_from(value_type):
                raise errors.ModelingError(
                    f"type of initial value {value_type} is not "
                    f"assignable to variable type {self.typ}",
                    self,
                )


@d.dataclass(frozen=True)
class ConstantDeclaration(IdentifierDeclaration):
    """
    Represents a *constant declaration*.

    Attributes
    ----------
    value:
        Optional :class:`~momba.model.Expression` specifying the value of the constant.
    """

    value: t.Optional[expressions.Expression] = None

    @property
    def is_parameter(self) -> bool:
        """
        Returns :obj:`True` if and only if the constant is a *parameter*.

        Parameters are constants without a :attr:`value`.
        """
        return self.value is None

    def validate(self, scope: Scope) -> None:
        if self.value is not None:
            value_type = scope.get_type(self.value)
            if not self.typ.is_assignable_from(value_type):
                raise errors.ModelingError(
                    f"type of constant value {value_type} is not "
                    f"assignable to constant type {self.typ}"
                )


@d.dataclass(frozen=True)
class PropertyDefinition:
    """
    Represents a *property definition*.

    Attributes
    ----------
    name:
        The name of the property.
    expression:
        An :class:`~momba.model.Expression` defining the property.
    comment:
        An optional comment describing the property.
    """

    name: str
    expression: expressions.Expression

    comment: t.Optional[str] = None


class Scope:
    """
    Represents a *scope*.

    Attributes
    ----------
    ctx:
        The modeling context associated with the scope.
    parent:
        The parent scope if it exists (:obj:`None` if there is no parent).
    """

    ctx: Context
    parent: t.Optional[Scope]

    _declarations: t.OrderedDict[str, IdentifierDeclaration]
    _functions: t.OrderedDict[str, functions.FunctionDefinition]

    _type_cache: t.Dict[expressions.Expression, types.Type]

    def __init__(self, ctx: Context, parent: t.Optional[Scope] = None):
        self.ctx = ctx
        self.parent = parent
        self._declarations = collections.OrderedDict()
        self._functions = collections.OrderedDict()
        self._type_cache = {}

    def __repr__(self) -> str:
        return f"<Scope parent={self.parent} at 0x{id(self):x}>"

    @property
    def declarations(self) -> t.AbstractSet[IdentifierDeclaration]:
        """
        Variable and constant declarations of the scope.
        """
        return frozenset(self._declarations.values())

    @property
    def variable_declarations(self) -> t.Sequence[VariableDeclaration]:
        """
        Variable declarations of the scope.
        """
        return tuple(
            declaration
            for declaration in self._declarations.values()
            if isinstance(declaration, VariableDeclaration)
        )

    @property
    def constant_declarations(self) -> t.Sequence[ConstantDeclaration]:
        """
        Constant declarations of the scope.
        """
        return tuple(
            declaration
            for declaration in self._declarations.values()
            if isinstance(declaration, ConstantDeclaration)
        )

    @property
    def clock_declarations(self) -> t.AbstractSet[VariableDeclaration]:
        """
        Variable declarations of clock variables of the scope.
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
        """
        Creates a child scope.
        """
        return Scope(self.ctx, parent=self)

    def get_type(self, expr: expressions.Expression) -> types.Type:
        """
        Returns the (inferred) type of the given expression in the scope.
        """
        if expr not in self._type_cache:
            inferred_type = expr.infer_type(self)
            inferred_type.validate_in(self)
            self._type_cache[expr] = inferred_type
        return self._type_cache[expr]

    def get_function(self, name: str) -> functions.FunctionDefinition:
        """
        Retrieves a :class:`FunctionDefinition` by its name.

        Raises :class:`~errors.NotFoundError` if no such definition exists.
        """
        try:
            return self._functions[name]
        except KeyError:
            if self.parent is None:
                raise errors.NotFoundError(f"no function with name {name} found")
            return self.parent.get_function(name)

    def is_local(self, identifier: str) -> bool:
        """
        Checks whether the identifier is locally declared in the scope.
        """
        return identifier in self._declarations

    def is_declared(self, identifier: str) -> bool:
        """
        Checks whether the identifier is declared in the scope.
        """
        if identifier in self._declarations:
            return True
        if self.parent is not None:
            return self.parent.is_declared(identifier)
        return False

    def get_scope(self, identifier: str) -> Scope:
        """
        Retrieves the scope in which the given identifier is declared.

        Raises :class:`~errors.NotFoundError` if no such identifier is declared.
        """
        if identifier in self._declarations:
            return self
        else:
            if self.parent is None:
                raise errors.NotFoundError(
                    f"identifier {identifier!r} is unbound in scope {self!r}"
                )
            return self.parent.get_scope(identifier)

    def lookup(self, identifier: str) -> IdentifierDeclaration:
        """
        Retrieves the declaration for the given identifier.

        Raises :class:`~errors.NotFoundError` if no such identifier is declared.
        """
        try:
            return self._declarations[identifier]
        except KeyError:
            if self.parent is None:
                raise errors.NotFoundError(
                    f"identifier {identifier!r} is unbound in scope {self!r}"
                )
            return self.parent.lookup(identifier)

    def add_declaration(
        self, declaration: IdentifierDeclaration, *, validate: bool = True
    ) -> None:
        """
        Adds an identifier declaration to the scope.

        The flag `validate` specifies whether the declaration should
        be validated within the scope before adding it. In case
        validation fails, a :class:`~errors.ModelingError` is raised.

        Raises :class:`~errors.ModelingError` in case the identifier
        has already been declared.
        """
        if declaration.identifier in self._declarations:
            raise errors.InvalidDeclarationError(
                f"identifier {declaration.identifier!r} has already been declared"
            )
        if validate:
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
        """
        Declares a variable within the scope.

        The parameters are passed to :class:`VariableDeclaration` with
        the exception of `initial_value`. When provided with a value
        which is not an expressions, this function implicitly converts
        the provided value into an expression using :func:`ensure_expr`.

        Raises :class:`~errors.ModelingError` in case the identifier
        has already been declared.
        """
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
        """
        Declares a constant within the scope.

        The parameters are passed to :class:`ConstantDeclaration` with
        the exception of `value`. When provided with a value which is
        not an expressions, this function implicitly converts the
        provided value into an expression using :func:`ensure_expr`.

        Raises :class:`~errors.ModelingError` in case the identifier
        has already been declared.
        """
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

    def define_function(
        self,
        name: str,
        parameters: t.Sequence[functions.FunctionParameter],
        returns: types.Type,
        body: expressions.Expression,
    ) -> functions.FunctionDefinition:
        """
        Defines a function within the scope.

        The parameters are passed to :class:`FunctionDefinition`.

        Raises :class:`~errors.ModelingError` in case an identically
        named function already exists.
        """
        if name in self._functions:
            raise errors.ModelingError(f"a function named {name!r} already exists")
        definition = functions.FunctionDefinition(
            name, tuple(parameters), returns, body
        )
        self._functions[name] = definition
        return definition


class Context:
    """
    Represents a *modeling context*.

    Attributes:
        model_type:
            The :class:`ModelType` of the modeling context.
        global_scope:
            The global :class:`Scope` of the modeling context.
    """

    model_type: ModelType
    global_scope: Scope

    _automata: t.Set[Automaton]
    _networks: t.Set[Network]

    _action_types: t.Dict[str, actions.ActionType]
    _named_properties: t.Dict[str, PropertyDefinition]

    _metadata: t.Dict[str, str]

    def __init__(self, model_type: ModelType) -> None:
        self.model_type = model_type
        self.global_scope = Scope(self)
        self._automata = set()
        self._networks = set()
        self._action_types = {}
        self._named_properties = {}
        self._metadata = {}

    def __repr__(self) -> str:
        return f"<Context model_type={self.model_type} at 0x{id(self):x}>"

    @property
    def automata(self) -> t.AbstractSet[Automaton]:
        """
        The set of automata defined on the modeling context.
        """
        return self._automata

    @property
    def networks(self) -> t.AbstractSet[Network]:
        """
        The set of networks defined on the modeling context.
        """
        return self._networks

    @property
    def metadata(self) -> t.Mapping[str, str]:
        """
        Additional metadata associated with the modeling
        context (e.g., author information).
        """
        return self._metadata

    @property
    def action_types(self) -> t.Mapping[str, actions.ActionType]:
        """
        The action types defined on the modeling context.
        """
        return self._action_types

    @property
    def properties(self) -> t.Mapping[str, PropertyDefinition]:
        """
        The properties defined on the modeling context.
        """
        return self._named_properties

    def update_metadata(self, metadata: t.Mapping[str, str]) -> None:
        """
        Updates the metadata with the provided mapping.
        """
        self._metadata.update(metadata)

    def get_automaton_by_name(self, name: str) -> Automaton:
        """
        Retrieves an automaton by its name.

        Raises :class:`~errors.NotFoundError` if no such automaton exists.
        """
        for automaton in self._automata:
            if automaton.name == name:
                return automaton
        raise errors.NotFoundError(f"there exists no automaton named {name!r}")

    def get_network_by_name(self, name: str) -> Network:
        """
        Retrives a network by its name.

        Raises :class:`~errors.NotFoundError` if no such network exists.
        """
        for network in self._networks:
            if network.name == name:
                return network
        raise errors.NotFoundError(f"there exists no network named {name!r}")

    def get_property_definition_by_name(self, name: str) -> PropertyDefinition:
        """
        Retrieves a property definition by its name.

        Raises :class:`~errors.NotFoundError` if no
        such property definition exists.
        """
        try:
            return self._named_properties[name]
        except KeyError:
            raise errors.NotFoundError(
                f"there exists no property definition named {name!r}"
            )

    def get_action_type_by_name(self, name: str) -> actions.ActionType:
        """
        Retrives an action type by its name.

        Raises :class:`~errors.NotFoundError` if no such action type exists.
        """
        try:
            return self._action_types[name]
        except KeyError:
            raise errors.NotFoundError(f"there exists no action type named {name!r}")

    def _add_action_type(self, action_type: actions.ActionType) -> None:
        """
        Adds an action type to the modeling context.

        Raises :class:`~errors.ModelingError` if an identically
        named action type already exists.
        """
        if action_type.label in self._action_types:
            raise errors.ModelingError(
                f"an action type with name {action_type.label!r} already exists"
            )
        self._action_types[action_type.label] = action_type

    def create_action_type(
        self, name: str, *, parameters: t.Sequence[actions.ActionParameter] = ()
    ) -> actions.ActionType:
        """
        Creates a new action type with the given name and parameters.

        Raises :class:`~errors.ModelingError` if an identically
        named action type already exists.
        """
        if name in self._action_types:
            raise errors.ModelingError(f"action type with name {name!r} already exists")
        action_type = actions.ActionType(name, tuple(parameters))
        self._add_action_type(action_type)
        return action_type

    def create_automaton(self, *, name: t.Optional[str] = None) -> Automaton:
        """
        Creates an automaton with the given optional name and returns it.
        """
        automaton = Automaton(self, name=name)
        self._automata.add(automaton)
        return automaton

    def create_network(self, *, name: t.Optional[str] = None) -> Network:
        """
        Creates a network with the given optional name and returns it.
        """
        network = Network(self, name=name)
        self._networks.add(network)
        return network

    def define_property(
        self, name: str, expression: expressions.Expression
    ) -> PropertyDefinition:
        """
        Defines a property on the modeling context.
        """
        definition = PropertyDefinition(name, expression)
        self._named_properties[name] = definition
        return definition
