# -*- coding:utf-8 -*-
#
# Copyright (C) 2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import fractions
import functools
import itertools
import random

from mxu.maps import FrozenMap

from .. import model

from ..kit import dbm
from ..model import actions, automata, context, effects, expressions, operators, types

from ..utils.distribution import Distribution

from . import evaluation


Binding = FrozenMap[str, evaluation.Value]


@d.dataclass(eq=False)
class PendingDefinition:
    name: str
    value: expressions.Expression
    uses: t.Set[str] = d.field(default_factory=set)


@d.dataclass(eq=False, frozen=True)
class NamespaceBuilder:
    scope: context.Scope
    parent: t.Optional[NamespaceBuilder] = None
    namespace: t.Dict[str, evaluation.Value] = d.field(default_factory=dict)
    defined: t.Set[str] = d.field(default_factory=set)
    deferred: t.Dict[str, t.Set[PendingDefinition]] = d.field(default_factory=dict)

    @functools.cached_property
    def environment(self) -> evaluation.Environment:
        parent: t.Optional[evaluation.Environment] = None
        if self.parent is not None:
            parent = self.parent.environment
        return evaluation.Environment(self.namespace, self.scope, parent)

    @property
    def binding(self) -> Binding:
        assert not self.has_deferred, (
            f"unable to compute initial values due to missing definitions "
            f"{', '.join(map(repr, self.deferred.values()))}"
        )
        return FrozenMap(self.namespace)

    @property
    def has_deferred(self) -> bool:
        return bool(self.deferred)

    def has_value(self, name: str) -> bool:
        if name in self.namespace:
            return True
        if self.parent:
            return self.parent.has_value(name)
        return False

    def create_child(self, scope: context.Scope) -> NamespaceBuilder:
        assert scope.parent is self.scope, "scope is not a child of the builder's scope"
        return NamespaceBuilder(scope, self)

    def define_initial_values(
        self, declarations: t.Iterable[context.Declaration]
    ) -> None:
        for declaration in declarations:
            if declaration.typ == types.CLOCK:
                continue
            if isinstance(declaration, context.VariableDeclaration):
                if declaration.initial_value is not None:
                    self.define(declaration.identifier, declaration.initial_value)
            else:
                assert isinstance(declaration, context.ConstantDeclaration)
                if declaration.value is not None:
                    self.define(declaration.identifier, declaration.value)

    def _check_deferred(self, stack: t.List[PendingDefinition]) -> None:
        while stack:
            top = stack.pop()
            assert not top.uses, "all used names should already be defined"
            self.defined.add(top.name)
            self.namespace[top.name] = evaluation.evaluate(top.value, self.environment)
            for deferred_definition in self.deferred.get(top.name, set()):
                deferred_definition.uses.remove(top.name)
                if not deferred_definition.uses:
                    stack.append(deferred_definition)
            try:
                del self.deferred[top.name]
            except KeyError:
                pass

    def assign(self, name: str, value: evaluation.Value) -> None:
        assert name not in self.defined, f"invalid double definition of name {name!r}"
        assert self.scope.is_local(name), f"{name!r} not local in the builder's scope"
        self.namespace[name] = value
        self.defined.add(name)
        stack: t.List[PendingDefinition] = []
        for pending_definition in self.deferred.get(name, set()):
            pending_definition.uses.discard(name)
            if not pending_definition.uses:
                stack.append(pending_definition)
        self._check_deferred(stack)

    def define(self, name: str, value: expressions.Expression) -> None:
        assert name not in self.defined, f"invalid double definition of name {name!r}"
        assert self.scope.is_local(name), f"{name!r} not local in the builder's scope"
        self.defined.add(name)
        pending_definition = PendingDefinition(name, value)
        for used_name in value.used_names:
            if not self.has_value(used_name.identifier):
                pending_definition.uses.add(used_name.identifier)
                if used_name.identifier not in self.deferred:
                    self.deferred[used_name.identifier] = set()
                self.deferred[used_name.identifier].add(pending_definition)
        if not pending_definition.uses:
            self._check_deferred([pending_definition])


@d.dataclass(frozen=True)
class InstanceState:
    location: automata.Location
    binding: Binding


@d.dataclass(frozen=True)
class GlobalState:
    network: model.Network
    binding: Binding
    instances: FrozenMap[automata.Instance, InstanceState]

    @functools.cached_property
    def global_environment(self) -> evaluation.Environment:
        return evaluation.Environment(self.binding, self.network.ctx.global_scope)

    def get_instance_environment(
        self, instance: automata.Instance
    ) -> evaluation.Environment:
        return evaluation.Environment(
            self.instances[instance].binding,
            instance.automaton.scope,
            self.global_environment,
        )


@d.dataclass(frozen=True)
class ClockVariable:
    identifier: str
    instance: t.Optional[automata.Instance] = None

    def __str__(self) -> str:
        if self.instance is None:
            return self.identifier
        return f"{self.identifier}@{self.instance.automaton.name}"


def _contains_clock_identifier(
    expression: expressions.Expression, scope: context.Scope
) -> bool:
    for used_name in expression.used_names:
        if scope.lookup(used_name.identifier).typ == types.CLOCK:
            return True
    return False


def _extract_clock_constraints(
    expression: expressions.Expression,
    instance: automata.Instance,
    environment: evaluation.Environment,
) -> t.Optional[t.AbstractSet[dbm.Constraint[ClockVariable]]]:
    constraints: t.Set[dbm.Constraint[ClockVariable]] = set()
    conjuncts: t.List[expressions.Expression] = []
    pending: t.List[expressions.Expression] = [expression]
    while pending:
        head = pending.pop()
        if isinstance(head, expressions.Boolean):
            if head.operator is operators.BooleanOperator.AND:
                pending.append(head.left)
                pending.append(head.right)
            else:
                conjuncts.append(head)
        elif isinstance(head, expressions.Comparison):
            if _contains_clock_identifier(head.left, instance.automaton.scope):
                difference = head.left
                operator = head.operator
                bound_expression = head.right
            elif _contains_clock_identifier(head.right, instance.automaton.scope):
                difference = head.right
                operator = head.operator.swap()
                bound_expression = head.left
            else:
                conjuncts.append(head)
                continue
            assert not _contains_clock_identifier(
                bound_expression, instance.automaton.scope
            )
            left: t.Union[dbm.ZeroClock, ClockVariable]
            right: t.Union[dbm.ZeroClock, ClockVariable]
            if isinstance(difference, expressions.Name):
                left = ClockVariable(difference.identifier, instance)
                right = dbm.ZERO_CLOCK
            else:
                assert (
                    isinstance(difference, expressions.ArithmeticBinary)
                    and difference.operator is operators.ArithmeticBinaryOperator.SUB
                    and isinstance(difference.left, expressions.Name)
                    and isinstance(difference.right, expressions.Name)
                )
                left = ClockVariable(difference.left.identifier, instance)
                right = ClockVariable(difference.right.identifier, instance)
            evaluated_bound = evaluation.evaluate(bound_expression, environment)
            assert isinstance(evaluated_bound, evaluation.Numeric)
            if operator.is_less:
                constraints.add(
                    dbm.Constraint(
                        dbm.difference(left, right),
                        dbm.Bound(
                            evaluated_bound.as_fraction,
                            is_strict=operator.is_strict,
                        ),
                    )
                )
            else:
                assert operator.is_greater
                constraints.add(
                    dbm.Constraint(
                        dbm.difference(right, left),
                        dbm.Bound(
                            -evaluated_bound.as_fraction,
                            is_strict=operator.is_strict,
                        ),
                    )
                )
        else:
            conjuncts.append(head)
    if conjuncts:
        result = evaluation.evaluate(expressions.logic_and(*conjuncts), environment)
        if not result.as_bool:
            return None
    return frozenset(constraints)


@d.dataclass(frozen=True)
class Action:
    action_type: actions.ActionType
    arguments: t.Tuple[evaluation.Value, ...]

    def __str__(self) -> str:
        arguments = ", ".join(map(str, self.arguments))
        return f"{self.action_type.name}({arguments})"


@d.dataclass(frozen=True)
class MDPEdge:
    source: GlobalState
    vector: t.Mapping[automata.Instance, automata.Edge]
    action: t.Optional[Action]
    destinations: Distribution[GlobalState]


MDPLocationType = GlobalState
MDPEdgeType = MDPEdge


@d.dataclass(frozen=True)
class PTALocation:
    invariant: t.FrozenSet[dbm.Constraint[ClockVariable]]
    state: GlobalState


@d.dataclass(frozen=True)
class PTADestination:
    reset: t.FrozenSet[ClockVariable]
    location: PTALocation


@d.dataclass(frozen=True)
class PTAEdge:
    source: PTALocation
    action: t.Optional[Action]
    vector: t.Mapping[automata.Instance, automata.Edge]
    guard: t.FrozenSet[dbm.Constraint[ClockVariable]]
    destinations: Distribution[PTADestination]


PTALocationType = PTALocation
PTAEdgeType = PTAEdge


class MombaPTA:
    SUPPORTED_TYPES = frozenset(
        {
            model.ModelType.LTS,
            model.ModelType.MDP,
            model.ModelType.TA,
            model.ModelType.PTA,
        }
    )

    network: model.Network

    _cache: t.Dict[PTALocationType, t.FrozenSet[PTAEdgeType]]

    def __init__(self, network: model.Network) -> None:
        assert network.ctx.model_type in self.SUPPORTED_TYPES
        self.network = network
        self._cache = {}

    @property
    def clock_variables(self) -> t.AbstractSet[ClockVariable]:
        clock_variables: t.Set[ClockVariable] = set()
        for instance in self.network.instances:
            for declaration in instance.automaton.scope.clock_declarations:
                clock_variables.add(ClockVariable(declaration.identifier, instance))
        return frozenset(clock_variables)

    @property
    def initial_locations(self) -> t.AbstractSet[PTALocationType]:
        global_builder = NamespaceBuilder(self.network.ctx.global_scope)
        global_builder.define_initial_values(self.network.ctx.global_scope.declarations)
        assert (
            not self.network.initial_restriction
        ), "initial restrictions are not yet supported"
        invariant: t.Set[dbm.Constraint[ClockVariable]] = set()
        instance_states: t.Dict[automata.Instance, InstanceState] = {}
        for instance in self.network.instances:
            assert (
                not instance.automaton.initial_restriction
            ), "initial restrictions are not yet supported"
            instance_builder = global_builder.create_child(instance.automaton.scope)
            for parameter, argument in zip(
                instance.automaton.parameters, instance.arguments
            ):
                instance_builder.assign(
                    parameter, evaluation.evaluate(argument, global_builder.environment)
                )
            instance_builder.define_initial_values(
                instance.automaton.scope.declarations
            )
            assert (
                len(instance.automaton.initial_locations) == 1
            ), "multiple initial locations are not yet supported"
            (initial_location,) = instance.automaton.initial_locations
            instance_states[instance] = InstanceState(
                initial_location, instance_builder.binding
            )
            if initial_location.progress_invariant:
                instance_invariant = _extract_clock_constraints(
                    initial_location.progress_invariant,
                    instance,
                    instance_builder.environment,
                )
                assert instance_invariant is not None
                invariant |= instance_invariant
        return frozenset(
            {
                PTALocation(
                    invariant=frozenset(invariant),
                    state=GlobalState(
                        self.network, global_builder.binding, FrozenMap(instance_states)
                    ),
                )
            }
        )

    @property
    def edges(self) -> t.AbstractSet[PTAEdgeType]:
        raise NotImplementedError()

    @property
    def locations(self) -> t.AbstractSet[PTALocationType]:
        raise NotImplementedError()

    def get_edges_to(self, destination: PTALocationType) -> t.AbstractSet[PTAEdgeType]:
        raise NotImplementedError()

    def _compute_destination(
        self,
        source: PTALocationType,
        transient_environments: TransientEnvironments,
        destination_vector: DestinationVector,
    ) -> t.Optional[t.Tuple[fractions.Fraction, PTADestination]]:
        target_environments = transient_environments.clone()
        # compute the probability for the respective compound destination
        probability = fractions.Fraction(1)
        for instance, destination in destination_vector.destinations.items():
            if destination.probability is None:
                continue
            probability *= evaluation.evaluate(
                destination.probability,
                target_environments.get_edge_environment(instance),
            ).as_fraction
        # collect the assignments to execute concurrently and clocks to reset
        concurrent_assignments: t.Dict[int, t.Set[DeferredAssignment]] = {}
        reset: t.Set[ClockVariable] = set()
        for instance, destination in destination_vector.destinations.items():
            for assignment in destination.assignments:
                assert isinstance(
                    assignment.target, effects.Name
                ), "non-identifier assignment targets are not supported yet"
                typ = instance.automaton.scope.get_type(assignment.target)
                if typ == types.CLOCK:
                    assert (
                        assignment.index == 0
                    ), "non-zero indices for assignments to clock variables are not supported"
                    value = evaluation.evaluate(
                        assignment.value,
                        target_environments.get_edge_environment(instance),
                    )
                    assert (
                        value.as_int == 0
                    ), "non-zero assignments to clock variables are not supported"
                    reset.add(ClockVariable(assignment.target.identifier, instance))
                else:
                    if assignment.index not in concurrent_assignments:
                        concurrent_assignments[assignment.index] = set()
                    concurrent_assignments[assignment.index].add(
                        DeferredAssignment(
                            instance, assignment.target.identifier, assignment.value
                        )
                    )
        for index in sorted(concurrent_assignments.keys()):
            target_environments.assign_concurrently(concurrent_assignments[index])
        # compute the resulting PTA destination
        instance_states = dict(source.state.instances)
        for instance, destination in destination_vector.destinations.items():
            instance_states[instance] = target_environments.get_instance_state(
                instance, destination.location
            )
        global_state = GlobalState(
            target_environments.network,
            FrozenMap(target_environments.global_namespace),
            FrozenMap.transfer_ownership(instance_states),
        )
        invariant: t.Set[dbm.Constraint[ClockVariable]] = set()
        for instance, state in global_state.instances.items():
            if state.location.progress_invariant is None:
                continue
            instance_invariant = _extract_clock_constraints(
                state.location.progress_invariant,
                instance,
                global_state.get_instance_environment(instance),
            )
            if instance_invariant is None:
                return None
            invariant |= instance_invariant
        pta_destination = PTADestination(
            frozenset(reset), PTALocation(frozenset(invariant), global_state)
        )
        return probability, pta_destination

    def _compute_edge(
        self,
        source: PTALocationType,
        edge_vector: EdgeVector,
        *,
        link: t.Optional[model.Link] = None,
    ) -> t.Optional[PTAEdgeType]:
        transient_environments = TransientEnvironments.from_global_state(
            source.state, edge_vector
        )
        vector_namespace: t.Dict[str, evaluation.Value] = {}
        for slot in edge_vector.slots:
            assert (
                slot.expressions
            ), "guard and read-only arguments are not supported yet"
            last_value: t.Optional[evaluation.Value] = None
            for slot_expression in slot.expressions:
                if slot_expression.instance:
                    slot_environment = transient_environments.get_instance_environment(
                        slot_expression.instance
                    )
                else:
                    slot_environment = transient_environments.global_environment
                value = evaluation.evaluate(
                    slot_expression.expression, slot_environment
                )
                if last_value is not None and value != last_value:
                    return None
            if slot.name is not None:
                vector_namespace[slot.name] = value
            for slot_identifier in slot.targets | slot.aliases:
                transient_environments.assign(
                    slot_identifier.name, value, instance=slot_identifier.instance
                )
        # check whether all guards are true
        guard: t.Set[dbm.Constraint[ClockVariable]] = set()
        for instance, edge in edge_vector.edges.items():
            if edge.guard is None:
                continue
            instance_guard = _extract_clock_constraints(
                edge.guard,
                instance,
                transient_environments.get_edge_environment(instance),
            )
            if instance_guard is None:
                return None
            guard |= instance_guard
        # compute the resulting action
        resulting_action: t.Optional[Action] = None
        if link is not None and edge_vector.result_pattern is not None:
            vector_environment = evaluation.Environment(
                vector_namespace,
                link.construct_scope(transient_environments.network.ctx),
                transient_environments.global_environment,
            )
            arguments: t.List[evaluation.Value] = []
            for argument in edge_vector.result_pattern.arguments:
                if isinstance(argument, actions.WriteArgument):
                    arguments.append(
                        evaluation.evaluate(argument.expression, vector_environment)
                    )
                else:
                    assert isinstance(
                        argument, actions.GuardArgument
                    ), "only write and guard argument allowed in result pattern"
                    arguments.append(vector_namespace[argument.identifier])
            resulting_action = Action(
                edge_vector.result_pattern.action_type, tuple(arguments)
            )
        # compute all destinations
        pta_destinations: t.Dict[PTADestination, fractions.Fraction] = {}
        for instance_destinations in itertools.product(
            *(
                [(instance, destination) for destination in edge.destinations]
                for instance, edge in edge_vector.edges.items()
            )
        ):
            destination_vector = DestinationVector(
                edge_vector, dict(instance_destinations)
            )
            destination = self._compute_destination(
                source, transient_environments, destination_vector
            )
            if destination is None:
                return None
            probability, pta_destination = destination
            pta_destinations[pta_destination] = (
                pta_destinations.get(pta_destination, 0) + probability
            )
        return PTAEdge(
            source=source,
            action=resulting_action,
            vector=FrozenMap.transfer_ownership(edge_vector.edges),  # type: ignore  # FIXME:
            guard=frozenset(guard),
            destinations=Distribution(pta_destinations),
        )

    def get_edges_from(self, source: PTALocationType) -> t.AbstractSet[PTAEdgeType]:
        if source in self._cache:
            return self._cache[source]
        pta_edges: t.Set[PTAEdgeType] = set()
        for instance, state in source.state.instances.items():
            for edge in instance.automaton.get_outgoing_edges(state.location):
                if edge.action_pattern is not None:
                    continue
                pta_edge = self._compute_edge(source, EdgeVector({instance: edge}))
                if pta_edge is not None:
                    pta_edges.add(pta_edge)
        for link in self.network.links:
            assert link.condition is None, "conditional links are not yet supported"
            for instance_edges in itertools.product(
                *(
                    [
                        (instance, edge)
                        for edge in instance.automaton.get_outgoing_edges(
                            source.state.instances[instance].location
                        )
                    ]
                    for instance in link.vector.keys()
                )
            ):
                edge_vector = _compute_edge_vector(
                    link.vector, dict(instance_edges), result_pattern=link.result
                )
                if edge_vector is not None:
                    pta_edge = self._compute_edge(source, edge_vector, link=link)
                    if pta_edge is not None:
                        pta_edges.add(pta_edge)
        self._cache[source] = frozenset(pta_edges)
        return self._cache[source]


class MombaMDP:
    SUPPORTED_TYPES = frozenset({model.ModelType.LTS, model.ModelType.MDP})

    network: model.Network

    _pta: MombaPTA

    def __init__(self, network: model.Network):
        assert network.ctx.model_type in self.SUPPORTED_TYPES
        self.network = network
        self._pta = MombaPTA(network)

    @property
    def initial_locations(self) -> t.AbstractSet[MDPLocationType]:
        return frozenset(location.state for location in self._pta.initial_locations)

    @property
    def edges(self) -> t.AbstractSet[MDPEdgeType]:
        raise NotImplementedError()

    @property
    def locations(self) -> t.AbstractSet[MDPLocationType]:
        raise NotImplementedError()

    def get_edges_from(self, source: MDPLocationType) -> t.AbstractSet[MDPEdgeType]:
        return frozenset(
            MDPEdge(
                source=source,
                action=edge.action,
                vector=edge.vector,
                destinations=Distribution(
                    {
                        destination.location.state: edge.destinations.get_probability(
                            destination
                        )
                        for destination in edge.destinations.support
                    }
                ),
            )
            for edge in self._pta.get_edges_from(PTALocation(frozenset(), source))
        )

    def get_edges_to(self, destination: MDPLocationType) -> t.AbstractSet[MDPEdgeType]:
        raise NotImplementedError()


@d.dataclass(eq=False, frozen=True)
class TransientEnvironments:
    network: model.Network
    edge_vector: EdgeVector
    global_namespace: t.Dict[str, evaluation.Value]
    instance_namespaces: t.Dict[automata.Instance, t.Dict[str, evaluation.Value]]
    edge_namespaces: t.Dict[automata.Instance, t.Dict[str, evaluation.Value]]

    def clone(self) -> TransientEnvironments:
        return TransientEnvironments(
            self.network,
            self.edge_vector,
            dict(self.global_namespace),
            {
                instance: dict(namespace)
                for instance, namespace in self.instance_namespaces.items()
            },
            {
                instance: dict(namespace)
                for instance, namespace in self.edge_namespaces.items()
            },
        )

    @classmethod
    def from_global_state(
        cls, global_state: GlobalState, edge_vector: EdgeVector
    ) -> TransientEnvironments:
        return TransientEnvironments(
            global_state.network,
            edge_vector,
            dict(global_state.binding),
            {
                instance: dict(state.binding)
                for instance, state in global_state.instances.items()
            },
            {instance: {} for instance in edge_vector.edges.keys()},
        )

    @functools.cached_property
    def global_environment(self) -> evaluation.Environment:
        return evaluation.Environment(
            self.global_namespace, self.network.ctx.global_scope
        )

    def get_instance_environment(
        self, instance: automata.Instance
    ) -> evaluation.Environment:
        return evaluation.Environment(
            self.instance_namespaces[instance],
            instance.automaton.scope,
            self.global_environment,
        )

    def get_edge_environment(
        self, instance: automata.Instance
    ) -> evaluation.Environment:
        return evaluation.Environment(
            self.edge_namespaces[instance],
            self.edge_vector.scopes[instance],
            self.get_instance_environment(instance),
        )

    def get_instance_state(
        self, instance: automata.Instance, location: automata.Location
    ) -> InstanceState:
        return InstanceState(
            location,
            FrozenMap.transfer_ownership(
                {
                    name: value
                    for name, value in self.instance_namespaces[instance].items()
                }
            ),
        )

    def assign(
        self,
        name: str,
        value: evaluation.Value,
        *,
        instance: t.Optional[automata.Instance] = None,
    ) -> None:
        if instance is not None:
            if self.edge_vector.scopes[instance].is_local(name):
                self.edge_namespaces[instance][name] = value
                return
            elif instance.automaton.scope.is_local(name):
                self.instance_namespaces[instance][name] = value
                return
        assert self.network.ctx.global_scope.is_local(name)
        self.global_namespace[name] = value

    def assign_concurrently(self, assignments: t.Iterable[DeferredAssignment]) -> None:
        values: t.Dict[DeferredAssignment, evaluation.Value] = {}
        for assignment in assignments:
            values[assignment] = evaluation.evaluate(
                assignment.value,
                self.get_edge_environment(assignment.instance),
            )
        for assignment in assignments:
            self.assign(
                assignment.name, values[assignment], instance=assignment.instance
            )


@d.dataclass(eq=False, frozen=True)
class VectorSlot:
    name: t.Optional[str] = None
    targets: t.Set[SlotIdentifier] = d.field(default_factory=set)
    expressions: t.Set[SlotExpression] = d.field(default_factory=set)
    aliases: t.Set[SlotIdentifier] = d.field(default_factory=set)


@d.dataclass(frozen=True)
class SlotIdentifier:
    name: str
    instance: t.Optional[automata.Instance] = None


@d.dataclass(frozen=True)
class SlotExpression:
    expression: expressions.Expression
    instance: t.Optional[automata.Instance] = None


@d.dataclass(eq=False, frozen=True)
class EdgeVector:
    edges: t.Mapping[automata.Instance, automata.Edge] = d.field(default_factory=dict)
    slots: t.List[VectorSlot] = d.field(default_factory=list)

    result_pattern: t.Optional[actions.ActionPattern] = None

    @functools.cached_property
    def scopes(self) -> t.Mapping[automata.Instance, context.Scope]:
        return {
            instance: edge.create_edge_scope(instance.automaton.scope)
            for instance, edge in self.edges.items()
        }


def _compute_edge_vector(
    vector_patterns: t.Mapping[automata.Instance, actions.ActionPattern],
    instance_edges: t.Mapping[automata.Instance, automata.Edge],
    *,
    result_pattern: t.Optional[actions.ActionPattern] = None,
) -> t.Optional[EdgeVector]:
    vector_slots: t.List[VectorSlot] = []
    named_slots: t.Dict[str, VectorSlot] = {}
    for instance, vector_pattern in vector_patterns.items():
        instance_pattern = instance_edges[instance].action_pattern
        if instance_pattern is None:
            return None
        if vector_pattern.action_type != instance_pattern.action_type:
            return None
        for vector_argument, instance_argument in zip(
            vector_pattern.arguments, instance_pattern.arguments
        ):
            if isinstance(vector_argument, actions.GuardArgument):
                try:
                    slot = named_slots[vector_argument.identifier]
                except KeyError:
                    slot = VectorSlot(vector_argument.identifier)
                    named_slots[vector_argument.identifier] = slot
            else:
                slot = VectorSlot()
            vector_slots.append(slot)
            if isinstance(vector_argument, actions.WriteArgument):
                slot.expressions.add(SlotExpression(vector_argument.expression))
            elif isinstance(vector_argument, actions.ReadArgument):
                slot.targets.add(SlotIdentifier(vector_argument.identifier))
            if isinstance(instance_argument, actions.WriteArgument):
                slot.expressions.add(
                    SlotExpression(instance_argument.expression, instance)
                )
            elif isinstance(instance_argument, actions.ReadArgument):
                slot.targets.add(SlotIdentifier(instance_argument.identifier, instance))
            else:
                assert isinstance(instance_argument, actions.GuardArgument)
                slot.aliases.add(SlotIdentifier(instance_argument.identifier, instance))
    return EdgeVector(instance_edges, vector_slots, result_pattern)


@d.dataclass(eq=False, frozen=True)
class DestinationVector:
    edge_vector: EdgeVector
    destinations: t.Mapping[automata.Instance, automata.Destination] = d.field(
        default_factory=dict
    )


@d.dataclass(eq=False, frozen=True)
class DeferredAssignment:
    instance: automata.Instance
    name: str
    value: expressions.Expression


@d.dataclass(frozen=True)
class PTAOption:
    edge: PTAEdge
    time_lower_bound: dbm.Bound
    time_upper_bound: t.Optional[dbm.Bound] = None


@d.dataclass(frozen=True)
class PTADecision:
    edge: PTAEdge
    time: fractions.Fraction


@d.dataclass(frozen=True)
class ActionTypeOracle:
    weights: t.Mapping[actions.ActionType, int] = d.field(default_factory=dict)

    default_weight: int = 10000

    def __call__(
        self,
        location: PTALocationType,
        valuation: t.Mapping[ClockVariable, fractions.Fraction],
        options: t.AbstractSet[PTAOption],
    ) -> PTADecision:
        weight_sum = sum(
            self.weights.get(option.edge.action.action_type, self.default_weight)
            if option.edge.action is not None
            else self.default_weight
            for option in options
        )
        threshold = random.randint(0, weight_sum)
        total = 0
        for option in options:
            if option.edge.action is None:
                total += self.default_weight
            else:
                total += self.weights.get(
                    option.edge.action.action_type, self.default_weight
                )
            if threshold <= total:
                break
        assert (
            option.time_upper_bound is not None
        ), "infinite time upper bounds not supported by the uniform oracle"
        time = option.time_lower_bound.constant + fractions.Fraction(
            random.random()
        ) * (option.time_upper_bound.constant - option.time_lower_bound.constant)
        return PTADecision(option.edge, time)
