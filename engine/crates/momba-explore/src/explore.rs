//! Algorithms and data structures for state space exploration.
#![allow(dead_code)]

use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

use std::convert::TryInto;

use super::evaluate;
use super::model;
use super::time;

use itertools::Itertools;

type ActionTypeIndex = usize;

type LocationIndex = usize;

type EdgeIndex = usize;

type DestinationIndex = usize;

/// Uniquely identifies an automaton of the model.
type AutomatonReference = usize;

/// Uniquely identifies a location of the model.
#[derive(Clone)]
struct LocationReference {
    automaton: AutomatonReference,
    index: LocationIndex,
}

/// Uniquely identifies an edge of the model.
#[derive(Clone)]
struct EdgeReference {
    location: LocationReference,
    index: EdgeIndex,
}

/// Uniquely identifies a destination of the model.
#[derive(Clone)]
struct DestinationReference {
    edge: EdgeReference,
    index: DestinationIndex,
}

type GlobalEnvironment<'s> = evaluate::Environment<'s, 2>;
type EdgeEnvironment<'s> = evaluate::Environment<'s, 3>;

/// Represents a global state of the automaton network.
///
/// Every state keeps track of the locations the automata of the network are in, the
/// values of the global variables, and a zone describing possible clock valuations.
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub struct State<Z: time::ZoneCompiler> {
    locations: Box<[LocationIndex]>,
    global_store: Box<[model::Value]>,
    transient_store: Box<[model::Value]>,
    zone: Z::Zone,
}

impl<Z: time::ZoneCompiler> State<Z> {
    /// Returns the name of the location the given automata is in.
    pub fn get_location_of<'n>(
        &self,
        network: &'n model::Network,
        automaton_name: &str,
    ) -> Option<&'n String> {
        network
            .automata
            .get_index_of(automaton_name)
            .and_then(|automaton_index| {
                self.locations
                    .get(automaton_index)
                    .and_then(|location_index| {
                        let (_, automaton) = network.automata.get_index(automaton_index).unwrap();
                        automaton
                            .locations
                            .get_index(*location_index)
                            .map(|(location_name, _)| location_name)
                    })
            })
    }

    /// Returns the value of the given global variable.
    pub fn get_global_value(
        &self,
        network: &model::Network,
        identifier: &str,
    ) -> Option<&model::Value> {
        network
            .declarations
            .global_variables
            .get_index_of(identifier)
            .and_then(|index| self.global_store.get(index))
    }

    /// Returns the value of the given transient variable.
    pub fn get_transient_value(
        &self,
        network: &model::Network,
        identifier: &str,
    ) -> Option<&model::Value> {
        network
            .declarations
            .transient_variables
            .get_index_of(identifier)
            .and_then(|index| self.transient_store.get(index))
    }

    /// Returns the clock zone of the state.
    pub fn zone(&self) -> &Z::Zone {
        &self.zone
    }

    fn global_env(&self) -> GlobalEnvironment {
        evaluate::Environment::new([&self.global_store, &self.transient_store])
    }

    fn edge_env<'e, 's: 'e>(&'s self, edge_store: &'e [model::Value]) -> EdgeEnvironment<'e> {
        evaluate::Environment::new([&self.global_store, &self.transient_store, edge_store])
    }
}

pub struct CompiledAssignment {
    target: evaluate::CompiledTargetExpression<3>,
    value: evaluate::CompiledExpression<3>,
}

/// Represents an action.
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub enum Action {
    /// The internal action.
    Internal,
    /// A visible action of a specific type with arguments.
    Visible(VisibleAction),
}

/// Represents a link action.
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub struct VisibleAction {
    action_type: ActionTypeIndex,
    arguments: Box<[model::Value]>,
}

impl VisibleAction {
    /// Returns the name of the action.
    pub fn get_name<'n>(&self, network: &'n model::Network) -> Option<&'n String> {
        network
            .declarations
            .action_types
            .get_index(self.action_type)
            .map(|(action_name, _)| action_name)
    }

    /// Returns a slice representing the arguments.
    pub fn arguments(&self) -> &[model::Value] {
        &self.arguments
    }
}

struct CompiledAutomaton<Z: time::ZoneCompiler> {
    reference: AutomatonReference,
    locations: Vec<CompiledLocation<Z>>,
}

struct CompiledLocation<Z: time::ZoneCompiler> {
    reference: LocationReference,
    invariant: Z::CompiledClockConstraints,
    internal_edges: Vec<CompiledEdge<Z>>,
    visible_edges: Vec<Vec<CompiledVisibleEdge<Z>>>,
}

impl<Z: time::ZoneCompiler> CompiledLocation<Z> {
    fn new(
        network: &model::Network,
        zone_compiler: &Z,
        automaton: &model::Automaton,
        location: &model::Location,
        reference: LocationReference,
        assignment_groups: &IndexSet<usize>,
    ) -> Self {
        let invariant = zone_compiler.compile_constraints(&location.invariant);
        let mut internal_edges = Vec::new();
        let mut visible_edges: Vec<Vec<CompiledVisibleEdge<Z>>> =
            (0..network.declarations.action_types.len())
                .map(|_| Vec::new())
                .collect();
        for (edge_index, edge) in location.edges.iter().enumerate() {
            let edge_reference = EdgeReference {
                location: reference.clone(),
                index: edge_index,
            };
            let compiled_edge = CompiledEdge::new(
                network,
                zone_compiler,
                automaton,
                edge,
                edge_reference,
                assignment_groups,
            );
            match &edge.pattern {
                model::ActionPattern::Internal => internal_edges.push(compiled_edge),
                model::ActionPattern::Link {
                    action_type,
                    arguments,
                } => {
                    let action_type_index = network
                        .declarations
                        .action_types
                        .get_index_of(action_type)
                        .unwrap();
                    visible_edges[action_type_index].push(CompiledVisibleEdge {
                        base: compiled_edge,
                        write_arguments: arguments
                            .iter()
                            .enumerate()
                            .filter_map(|(_argument_index, argument)| match argument {
                                model::PatternArgument::Write { value: _ } => todo!(),
                                _ => None,
                            })
                            .collect(),
                    })
                }
            }
        }
        CompiledLocation {
            reference,
            invariant,
            internal_edges: internal_edges,
            visible_edges: visible_edges,
        }
    }
}

struct CompiledEdge<Z: time::ZoneCompiler> {
    reference: EdgeReference,
    guard: CompiledGuard<Z>,
    destinations: Vec<CompiledDestination<Z>>,
}

struct CompiledVisibleEdge<Z: time::ZoneCompiler> {
    base: CompiledEdge<Z>,
    write_arguments: Box<[(usize, evaluate::CompiledExpression<2>)]>,
}

impl<Z: time::ZoneCompiler> CompiledEdge<Z> {
    fn new(
        network: &model::Network,
        zone_compiler: &Z,
        automaton: &model::Automaton,
        edge: &model::Edge,
        reference: EdgeReference,
        assignment_groups: &IndexSet<usize>,
    ) -> Self {
        let global_scope = network.global_scope();
        let edge_scope = edge.edge_scope(network);
        let guard = CompiledGuard {
            boolean_condition: global_scope.compile(&edge.guard.boolean_condition),
            clock_constraints: zone_compiler.compile_constraints(&edge.guard.clock_constraints),
        };
        let destinations = edge
            .destinations
            .iter()
            .enumerate()
            .map(|(destination_index, destination)| CompiledDestination {
                reference: DestinationReference {
                    edge: reference.clone(),
                    index: destination_index,
                },
                location: automaton
                    .locations
                    .get_index_of(&destination.location)
                    .unwrap(),
                probability: edge_scope.compile(&destination.probability),
                reset: zone_compiler.compile_clock_set(&destination.reset),
                assignments: assignment_groups
                    .iter()
                    .map(|index| {
                        destination
                            .assignments
                            .iter()
                            .filter(|assignment| assignment.index == *index)
                            .map(|assignment| CompiledAssignment {
                                target: edge_scope.compile_target(&assignment.target),
                                value: edge_scope.compile(&assignment.value),
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect();
        CompiledEdge {
            reference,
            guard,
            destinations,
        }
    }

    fn is_enabled(&self, global_env: &evaluate::Environment<2>) -> bool {
        self.guard
            .boolean_condition
            .evaluate(global_env)
            .try_into()
            .unwrap()
    }
}

struct CompiledGuard<Z: time::ZoneCompiler> {
    boolean_condition: evaluate::CompiledExpression<2>,
    clock_constraints: Z::CompiledClockConstraints,
}

struct CompiledDestination<Z: time::ZoneCompiler> {
    reference: DestinationReference,
    location: LocationIndex,
    probability: evaluate::CompiledExpression<3>,
    reset: Z::CompiledClockSet,
    assignments: Box<[Box<[CompiledAssignment]>]>,
}

struct SyncVectorItem {
    automaton: AutomatonReference,
    action_type: ActionTypeIndex,
    slot_mapping: Box<[usize]>,
}

pub enum Transition<'c, Z: time::ZoneCompiler> {
    Internal(InternalTransition<'c, Z>),
    Link(LinkTransition<'c, Z>),
}

pub struct InternalTransition<'c, Z: time::ZoneCompiler> {
    edge: &'c CompiledEdge<Z>,
    zone: Z::Zone,
}

pub struct LinkTransition<'c, Z: time::ZoneCompiler> {
    link: &'c CompiledLink,
    edges: Box<[&'c CompiledVisibleEdge<Z>]>,
    slots: Box<[model::Value]>,
    zone: Z::Zone,
}

impl<'c, Z: time::ZoneCompiler> Transition<'c, Z> {
    pub fn zone(&self) -> &Z::Zone {
        match self {
            Transition::Internal(transition) => &transition.zone,
            Transition::Link(transition) => &transition.zone,
        }
    }
}

impl SyncVectorItem {
    fn argument_to_slot_index(&self, argument_index: usize) -> usize {
        self.slot_mapping[argument_index]
    }

    fn compute_link_edges<'c, Z: time::ZoneCompiler>(
        &'c self,
        global_env: &evaluate::Environment<2>,
        enabled_edges: &Box<[Box<[Box<[&'c CompiledVisibleEdge<Z>]>]>]>,
    ) -> Vec<LinkEdge<Z>> {
        enabled_edges[self.automaton][self.action_type]
            .iter()
            .map(|edge| LinkEdge {
                compiled: edge,
                write_slots: edge
                    .write_arguments
                    .iter()
                    .map(|(argument_index, expression)| {
                        let slot_index = self.argument_to_slot_index(*argument_index);
                        let value = expression.evaluate(global_env);
                        (slot_index, value)
                    })
                    .collect(),
            })
            .collect()
    }
}

struct LinkEdge<'c, Z: time::ZoneCompiler> {
    compiled: &'c CompiledVisibleEdge<Z>,
    write_slots: Box<[(usize, model::Value)]>,
}

pub struct Destination<'c, Z: time::ZoneCompiler> {
    probability: f64,
    destinations: Box<[&'c CompiledDestination<Z>]>,
}

impl<'c, Z: time::ZoneCompiler> Destination<'c, Z> {
    pub fn probability(&self) -> f64 {
        self.probability
    }
}

struct CompiledLink {
    slots_template: Vec<Option<model::Value>>,
    sync_vector: Box<[SyncVectorItem]>,
    // result_action_type: ActionType,
    // result_slot_mapping: LinkSlotMapping,
}

struct CompiledNetwork<Z: time::ZoneCompiler> {
    zone_compiler: Z,
    automata: Box<[CompiledAutomaton<Z>]>,
    links: Box<[CompiledLink]>,
    transient_values: Box<[evaluate::CompiledExpression<1>]>,
    assignment_groups: IndexSet<usize>,
}

impl<Z: time::ZoneCompiler> CompiledNetwork<Z> {
    pub fn new(network: &model::Network) -> Self {
        let zone_compiler = Z::new(network);
        let mut assignment_groups: IndexSet<usize> = network
            .automata
            .values()
            .map(|automaton| automaton.locations.values())
            .flatten()
            .map(|location| location.edges.iter())
            .flatten()
            .map(|edge| edge.destinations.iter())
            .flatten()
            .map(|destination| destination.assignments.iter())
            .flatten()
            .map(|assignment| assignment.index)
            .collect();
        assignment_groups.sort();
        let automata = network
            .automata
            .values()
            .enumerate()
            .map(|(automaton_reference, automaton)| CompiledAutomaton {
                reference: automaton_reference,
                locations: automaton
                    .locations
                    .values()
                    .enumerate()
                    .map(|(location_index, location)| {
                        CompiledLocation::new(
                            network,
                            &zone_compiler,
                            automaton,
                            location,
                            LocationReference {
                                automaton: automaton_reference,
                                index: location_index,
                            },
                            &assignment_groups,
                        )
                    })
                    .collect(),
            })
            .collect();
        let links = network
            .links
            .iter()
            .map(|link| CompiledLink {
                slots_template: vec![None; link.slots.len()],
                sync_vector: link
                    .vector
                    .iter()
                    .map(|(automaton_name, link_pattern)| SyncVectorItem {
                        automaton: network.automata.get_index_of(automaton_name).unwrap(),
                        action_type: network
                            .declarations
                            .action_types
                            .get_index_of(&link_pattern.action_type)
                            .unwrap(),
                        slot_mapping: link_pattern
                            .arguments
                            .iter()
                            .map(|slot_name| link.slots.get_index_of(slot_name).unwrap())
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        let transient_scope = network.transient_scope();
        let transient_values = network
            .declarations
            .transient_variables
            .values()
            .map(|expr| transient_scope.compile(expr))
            .collect();
        CompiledNetwork {
            zone_compiler,
            automata,
            links,
            transient_values,
            assignment_groups,
        }
    }

    fn compute_transition<'c>(
        &self,
        zone: &Z::Zone,
        link: &'c CompiledLink,
        link_edges: &[&LinkEdge<'c, Z>],
    ) -> Option<Transition<'c, Z>> {
        debug_assert_eq!(link.sync_vector.len(), link_edges.len());
        let mut slots = link.slots_template.clone();
        let mut zone = zone.clone();
        for (link_edge, _vector_item) in link_edges.iter().zip(link.sync_vector.iter()) {
            for (slot_index, value) in link_edge.write_slots.iter() {
                match &slots[*slot_index] {
                    None => slots[*slot_index] = Some(value.clone()),
                    Some(other_value) => {
                        if value != other_value {
                            return None;
                        }
                    }
                }
            }
            // TODO: Extract constraints from `link_edge` and apply to `zone`.
            // TODO: Check whether `zone` is empty and return `None` if this is the case.
        }
        slots
            .into_iter()
            .collect::<Option<Box<[_]>>>()
            .map(|slots| {
                Transition::Link(LinkTransition {
                    link: link,
                    edges: link_edges
                        .iter()
                        .map(|link_edge| link_edge.compiled)
                        .collect(),
                    slots,
                    zone,
                })
            })
    }

    fn compute_transient_values(&self, global_store: &[model::Value]) -> Box<[model::Value]> {
        let env = evaluate::Environment::new([global_store]);
        self.transient_values
            .iter()
            .map(|expr| expr.evaluate(&env))
            .collect()
    }
}

pub struct Explorer<Z: time::ZoneCompiler> {
    compiled_network: CompiledNetwork<Z>,
}

impl<T: time::ZoneCompiler> Explorer<T> {
    pub fn new(network: &model::Network) -> Self {
        Explorer {
            compiled_network: CompiledNetwork::new(network),
        }
    }

    pub fn initial_states(&self, network: &model::Network) -> Vec<State<T>> {
        network
            .initial_states
            .iter()
            .map(|state| {
                let global_store: Box<[model::Value]> = network
                    .declarations
                    .global_variables
                    .keys()
                    .map(|identifier| state.values[identifier].clone())
                    .collect();
                let transient_store = self
                    .compiled_network
                    .compute_transient_values(&global_store);
                State {
                    locations: network
                        .automata
                        .iter()
                        .map(|(automaton_name, automaton)| {
                            automaton
                                .locations
                                .get_index_of(&state.locations[automaton_name])
                                .unwrap()
                        })
                        .collect(),
                    global_store: global_store,
                    transient_store: transient_store,
                    zone: self.compiled_network.zone_compiler.create_zero(),
                }
            })
            .collect()
    }

    pub fn transitions<'c>(&'c self, state: &State<T>) -> Vec<Transition<'c, T>> {
        let global_env = state.global_env();
        let enabled_edges = self
            .compiled_network
            .automata
            .iter()
            .zip(state.locations.iter())
            .map(|(automaton, location_index)| {
                automaton.locations[*location_index]
                    .visible_edges
                    .iter()
                    .map(|link_edges| {
                        link_edges
                            .iter()
                            .filter(|edge| edge.base.is_enabled(&global_env))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        self.compiled_network
            .automata
            .iter()
            .map(|automaton| {
                let location = &automaton.locations[state.locations[automaton.reference]];
                location
                    .internal_edges
                    .iter()
                    .filter(|edge| edge.is_enabled(&global_env))
                    .map(|edge| {
                        Transition::Internal(InternalTransition {
                            edge: edge,
                            zone: state.zone.clone(),
                        })
                    })
            })
            .flatten()
            .chain(
                self.compiled_network
                    .links
                    .iter()
                    .map(|link| {
                        link.sync_vector
                            .iter()
                            .map(|vector_item| {
                                vector_item.compute_link_edges(&global_env, &enabled_edges)
                            })
                            .collect::<Vec<_>>()
                            .iter()
                            .multi_cartesian_product()
                            .filter_map(|edges| {
                                self.compiled_network
                                    .compute_transition(&state.zone, link, &edges)
                            })
                            .collect::<Vec<_>>()
                    })
                    .flatten(),
            )
            .collect()
    }

    pub fn destinations<'c>(
        &'c self,
        state: &State<T>,
        transition: &Transition<'c, T>,
    ) -> Vec<Destination<'c, T>> {
        match transition {
            Transition::Internal(InternalTransition { edge, zone: _ }) => {
                let edge_env = state.edge_env(&[]);
                edge.destinations
                    .iter()
                    .map(|destination| Destination {
                        probability: destination
                            .probability
                            .evaluate(&edge_env)
                            .try_into()
                            .unwrap(),
                        destinations: Box::new([destination]),
                    })
                    .collect()
            }
            Transition::Link(LinkTransition {
                link: _,
                edges,
                slots: _,
                zone: _,
            }) => edges
                .iter()
                .map(|edge| edge.base.destinations.iter())
                .multi_cartesian_product()
                .map(|destinations| Destination {
                    probability: destinations.iter().fold(1.0, |probability, destination| {
                        let edge_env = state.edge_env(&[]);
                        probability
                            * destination
                                .probability
                                .evaluate(&edge_env)
                                .unwrap_float64()
                                .into_inner()
                    }),
                    destinations: destinations.into(),
                })
                .collect(),
        }
    }

    pub fn successor<'c>(
        &'c self,
        state: &State<T>,
        transition: &Transition<'c, T>,
        destination: &Destination<T>,
    ) -> State<T> {
        let mut targets = state.global_store.clone();
        for index in 0..self.compiled_network.assignment_groups.len() {
            let global_store = targets.clone();
            for automaton_destination in destination.destinations.iter() {
                let env = EdgeEnvironment::new([&global_store, &state.transient_store, &[]]);
                for assignment in automaton_destination.assignments[index].iter() {
                    let value = assignment.value.evaluate(&env);
                    assignment.target.evaluate(&mut targets, &env).store(value);
                }
            }
        }

        let mut locations = state.locations.clone();
        for automaton_destination in destination.destinations.iter() {
            let automaton_index = automaton_destination.reference.edge.location.automaton;
            locations[automaton_index] = automaton_destination.location;
        }

        let transient_store = self.compiled_network.compute_transient_values(&targets);

        // TODO: Apply `future` to the zone and restrict with invariants of the locations.
        let zone = transition.zone().clone();

        State {
            locations: locations,
            global_store: targets,
            transient_store: transient_store,
            zone: zone,
        }
    }
}
