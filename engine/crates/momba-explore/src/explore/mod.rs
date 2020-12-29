//! Algorithms and data structures for state space exploration.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

use super::model;
use super::time;

// pub mod dead;

mod actions;
mod compiled;
mod evaluate;

use compiled::*;

pub use actions::*;

use itertools::Itertools;

/// A *state* of an automata network.
///
/// Every state keeps track of the locations the automata of the network are in, the
/// values of the global variables, and a zone describing possible clock valuations.
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub struct State<Z: time::TimeType> {
    locations: Box<[LocationIndex]>,
    global_store: Box<[model::Value]>,
    transient_store: Box<[model::Value]>,
    zone: Z::Valuations,
}

impl<Z: time::TimeType> State<Z> {
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
    pub fn zone(&self) -> &Z::Valuations {
        &self.zone
    }

    fn global_env(&self) -> GlobalEnvironment {
        evaluate::Environment::new([&self.global_store, &self.transient_store])
    }

    fn edge_env<'e, 's: 'e>(&'s self, edge_store: &'e [model::Value]) -> EdgeEnvironment<'e> {
        evaluate::Environment::new([&self.global_store, &self.transient_store, edge_store])
    }
}

/// A *transition* of an automata network.
pub struct Transition<'c, Z: time::TimeType> {
    pub(crate) edges: Box<[&'c CompiledEdge<Z>]>,
    pub(crate) actions: Box<[Action]>,
    pub(crate) zone: Z::Valuations,
    pub(crate) action: Action,
}

/// Represents a destination of a transition.
pub struct Destination<'c, Z: time::TimeType> {
    pub(crate) probability: f64,
    pub(crate) destinations: Box<[&'c CompiledDestination<Z>]>,
}

impl<'c, Z: time::TimeType> Destination<'c, Z> {
    pub fn probability(&self) -> f64 {
        self.probability
    }
}

/// A state space explorer for a particular automata network.
pub struct Explorer<Z: time::TimeType> {
    pub network: model::Network,
    compiled_network: CompiledNetwork<Z>,
}

impl<T: time::TimeType> Explorer<T> {
    pub fn new(network: model::Network) -> Self {
        let compiled = CompiledNetwork::new(&network);
        Explorer {
            network,
            compiled_network: compiled,
        }
    }

    pub fn initial_states(&self) -> Vec<State<T>> {
        self.network
            .initial_states
            .iter()
            .map(|state| {
                let global_store: Box<[model::Value]> = self
                    .network
                    .declarations
                    .global_variables
                    .keys()
                    .map(|identifier| state.values[identifier].clone())
                    .collect();
                let transient_store = self
                    .compiled_network
                    .compute_transient_values(&global_store);
                State {
                    locations: self
                        .network
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
                    .map(|edge| Transition {
                        edges: Box::new([edge]),
                        actions: Box::new([Action::Silent]),
                        zone: state.zone().clone(),
                        action: Action::Silent,
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
        transition
            .edges
            .iter()
            .map(|edge| edge.destinations.iter())
            .multi_cartesian_product()
            .map(|destinations| Destination {
                probability: transition.actions.iter().zip(destinations.iter()).fold(
                    1.0,
                    |probability, (action, destination)| {
                        let edge_env = state.edge_env(action.arguments());
                        probability
                            * destination
                                .probability
                                .evaluate(&edge_env)
                                .unwrap_float64()
                                .into_inner()
                    },
                ),
                destinations: destinations.into(),
            })
            .collect()
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
        let zone = transition.zone.clone();

        State {
            locations: locations,
            global_store: targets,
            transient_store: transient_store,
            zone: zone,
        }
    }
}

/// A specialization of [Explorer] for MDPs using [NoClocks][time::NoClocks].
///
/// MDPs do not have any real-valued clocks.
pub type MDPExplorer = Explorer<time::NoClocks>;
