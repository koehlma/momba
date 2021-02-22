//! Algorithms and data structures for state space exploration.

use serde::{Deserialize, Serialize};

use itertools::Itertools;

use super::model;
use super::time;

mod actions;
mod compiled;
mod evaluate;

pub mod external;

use compiled::*;

pub use actions::*;

/// A *state* of an automaton network.
///
/// Every state keeps track of the locations the automata of the network are in, the
/// values of the global variables, and a potentially infinite set of clock valuations
/// using the respective [TimeType][time::TimeType].
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub struct State<V> {
    locations: Box<[LocationIndex]>,
    global_store: Box<[model::Value]>,
    transient_store: Box<[model::Value]>,
    valuations: V,
}

impl<V> State<V> {
    /// Returns the name of the location the automaton with the provided name is in.
    ///
    /// Panics in case the state has not been produced by the provided explorer or there
    /// is no automaton with the provided name in the automaton network.
    pub fn get_location_of<'e, T: time::TimeType>(
        &self,
        explorer: &'e Explorer<T>,
        automaton_name: &str,
    ) -> &'e String {
        explorer
            .network
            .automata
            .get_index_of(automaton_name)
            .and_then(|automaton_index| {
                self.locations
                    .get(automaton_index)
                    .and_then(|location_index| {
                        let (_, automaton) = explorer
                            .network
                            .automata
                            .get_index(automaton_index)
                            .unwrap();
                        automaton
                            .locations
                            .get_index(*location_index)
                            .map(|(location_name, _)| location_name)
                    })
            })
            .expect("Invalid automaton name or explorer passed to `State::get_location_of`.")
    }

    /// Returns the value of the provided global variable.
    ///
    /// Panics in case the state has not been produced by the provided explorer or there
    /// is no global variable with the provided name in the automaton network.
    pub fn get_global_value<T: time::TimeType>(
        &self,
        explorer: &Explorer<T>,
        identifier: &str,
    ) -> &model::Value {
        explorer
            .network
            .declarations
            .global_variables
            .get_index_of(identifier)
            .and_then(|index| self.global_store.get(index))
            .expect("Invalid variable name or explorer passed to `State::get_global_value`.")
    }

    /// Returns the value of the provided transient variable.
    ///
    /// Panics in case the state has not been produced by the provided explorer or there
    /// is no transient variable with the provided name in the automaton network.
    pub fn get_transient_value(&self, network: &model::Network, identifier: &str) -> &model::Value {
        network
            .declarations
            .transient_variables
            .get_index_of(identifier)
            .and_then(|index| self.transient_store.get(index))
            .expect("Invalid variable name or explorer passed to `State::get_transient_value`.")
    }

    /// Returns the clock valuations associated with the state.
    pub fn valuations(&self) -> &V {
        &self.valuations
    }

    /// Constructs a global evaluation environment from the state.
    fn global_env(&self) -> GlobalEnvironment {
        evaluate::Environment::new([&self.global_store, &self.transient_store])
    }

    /// Constructs an edge evaluation environment from the state and edge store.
    fn edge_env<'e, 's: 'e>(&'s self, edge_store: &'e [model::Value]) -> EdgeEnvironment<'e> {
        evaluate::Environment::new([&self.global_store, &self.transient_store, edge_store])
    }

    fn future<T: time::TimeType>(
        state: State<T::Valuations>,
        compiled_network: &CompiledNetwork<T>,
    ) -> State<T::Valuations> {
        let env = evaluate::Environment::new([&state.global_store, &state.transient_store]);

        let mut valuations = compiled_network.zone_compiler.future(state.valuations);

        for (automaton_index, location_index) in state.locations.iter().enumerate() {
            let location = &compiled_network.automata[automaton_index].locations[*location_index];
            valuations = location
                .invariant
                .iter()
                .fold(valuations, |valuations, constraint| {
                    compiled_network.zone_compiler.constrain(
                        valuations,
                        &constraint.difference,
                        constraint.is_strict,
                        constraint.bound.evaluate(&env),
                    )
                })
        }

        State {
            locations: state.locations,
            global_store: state.global_store,
            transient_store: state.transient_store,
            valuations: valuations,
        }
    }
}

/// A *transition* of an automaton network.
pub struct Transition<'e, T: time::TimeType> {
    pub(crate) edges: Box<[&'e CompiledEdge<T>]>,
    pub(crate) actions: Box<[Action]>,
    pub(crate) valuations: T::Valuations,
    pub(crate) action: Action,
}

impl<'e, T: time::TimeType> Transition<'e, T> {
    /// Returns a slice of actions the participating automata perform.
    pub fn local_actions(&self) -> &[Action] {
        &self.actions
    }

    /// Returns a vector of indices of the participating automata.
    pub fn edges(&self) -> Vec<model::EdgeReference> {
        self.edges
            .iter()
            .map(|edge| edge.reference.clone())
            .collect()
    }

    /// Returns the clock valuations for which the transition is performed.
    pub fn valuations(&self) -> &T::Valuations {
        &self.valuations
    }

    /// Returns the resulting action.
    pub fn result_action(&self) -> &Action {
        &self.action
    }

    /// Replaces the valuations of the transition.
    pub fn replace_valuations(self, valuations: T::Valuations) -> Self {
        Transition {
            edges: self.edges,
            actions: self.actions,
            valuations,
            action: self.action,
        }
    }

    pub fn edge_references(&self) -> Vec<model::EdgeReference> {
        self.edges
            .iter()
            .map(|edge| edge.reference.clone())
            .collect()
    }
}

/// A *destination* of a transition.
pub struct Destination<'c, T: time::TimeType> {
    pub(crate) probability: f64,
    pub(crate) destinations: Box<[&'c CompiledDestination<T>]>,
}

impl<'c, T: time::TimeType> Destination<'c, T> {
    /// Returns the probability of the destination.
    pub fn probability(&self) -> f64 {
        self.probability
    }
}

/// A state space explorer for a particular automaton network.
pub struct Explorer<T: time::TimeType> {
    pub network: model::Network,
    pub(crate) compiled_network: CompiledNetwork<T>,
}

impl<T: time::TimeType> Explorer<T> {
    /// Constructs a new state space explorer from the provided network.
    pub fn new(network: model::Network) -> Self {
        let compiled = CompiledNetwork::new(&network);
        Explorer {
            network,
            compiled_network: compiled,
        }
    }

    /// Returns a vector of initial states of the network.
    pub fn initial_states(&self) -> Vec<State<T::Valuations>> {
        let global_scope = self.network.global_scope();
        self.network
            .initial_states
            .iter()
            .map(|state| {
                let locations = self
                    .network
                    .automata
                    .iter()
                    .map(|(automaton_name, automaton)| {
                        automaton
                            .locations
                            .get_index_of(&state.locations[automaton_name])
                            .unwrap()
                    })
                    .collect();
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
                let global_env = evaluate::Environment::new([&global_store, &transient_store]);
                // FIXME: explore the future of this state
                let valuations = self
                    .compiled_network
                    .zone_compiler
                    .create_valuations(
                        state
                            .zone
                            .iter()
                            .map(|constraint| {
                                CompiledClockConstraint::compile(
                                    &self.compiled_network.zone_compiler,
                                    constraint,
                                    &global_scope,
                                )
                                .evaluate(&global_env)
                            })
                            .collect(),
                    )
                    .unwrap();
                State::<T::Valuations>::future(
                    State {
                        locations,
                        global_store,
                        transient_store,
                        valuations,
                    },
                    &self.compiled_network,
                )
            })
            .collect()
    }

    /// Returns a vector of outgoing transitions of the given state.
    pub fn transitions<'c>(&'c self, state: &State<T::Valuations>) -> Vec<Transition<'c, T>> {
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
            .enumerate()
            .map(|(index, automaton)| {
                let location = &automaton.locations[state.locations[index]];
                location.internal_edges.iter().filter_map(|edge| {
                    if !edge.is_enabled(&global_env) {
                        None
                    } else {
                        // We may want to improve the efficiency of this function in the future.
                        //
                        // Instead of applying each constraint individually applying them in bulk
                        // makes canonicalization more efficient for clock zones.
                        let valuations = edge.guard.clock_constraints.iter().fold(
                            state.valuations.clone(),
                            |valuations, constraint| {
                                self.compiled_network.zone_compiler.constrain(
                                    valuations,
                                    &constraint.difference,
                                    constraint.is_strict,
                                    constraint.bound.evaluate(&global_env),
                                )
                            },
                        );
                        if self.compiled_network.zone_compiler.is_empty(&valuations) {
                            None
                        } else {
                            Some(Transition {
                                edges: Box::new([edge]),
                                actions: Box::new([Action::Silent]),
                                valuations,
                                action: Action::Silent,
                            })
                        }
                    }
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
                                self.compiled_network.compute_transition(
                                    &state.valuations,
                                    &global_env,
                                    link,
                                    &edges,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .flatten(),
            )
            .collect()
    }

    /// Returns a vector of destinations for a given transition.
    ///
    /// Panics if the transition has not been generated from the provided state.
    pub fn destinations<'c>(
        &'c self,
        state: &State<T::Valuations>,
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

    /// Returns the successor state reached via a destination.
    ///
    /// Panics if the destination has not been generated from the provided state and transition.
    pub fn successor<'c>(
        &'c self,
        state: &State<T::Valuations>,
        transition: &Transition<'c, T>,
        destination: &Destination<T>,
    ) -> State<T::Valuations> {
        let mut targets = vec![
            model::Value::Vector(state.global_store.clone().into()),
            model::Value::Vector(state.transient_store.clone().into()),
        ];
        for index in 0..self.compiled_network.assignment_groups.len() {
            let global_store = targets[0].clone();
            for (action, automaton_destination) in transition
                .actions
                .iter()
                .zip(destination.destinations.iter())
            {
                let env = EdgeEnvironment::new([
                    &global_store.unwrap_vector(),
                    &state.transient_store,
                    action.arguments(),
                ]);
                for assignment in automaton_destination.assignments[index].iter() {
                    let value = assignment.value.evaluate(&env);
                    assignment.target.evaluate(&mut targets, &env).store(value);
                }
            }
        }

        let mut locations = state.locations.clone();
        let mut valuations = transition.valuations().clone();
        for automaton_destination in destination.destinations.iter() {
            let automaton_index = automaton_destination.automaton_index;
            locations[automaton_index] = automaton_destination.location;
            valuations = self
                .compiled_network
                .zone_compiler
                .reset(valuations, &automaton_destination.reset);
        }

        let transient_store = self
            .compiled_network
            .compute_transient_values(&targets[0].unwrap_vector());

        let state = State::<T::Valuations>::future(
            State {
                locations,
                global_store: targets[0].unwrap_vector().clone().into(),
                transient_store,
                valuations,
            },
            &self.compiled_network,
        );

        // The truth of the guard should imply the truth of the invariants.
        assert!(!self
            .compiled_network
            .zone_compiler
            .is_empty(&state.valuations));

        state
    }

    pub fn externalize_transition(&self, transition: &Transition<T>) -> external::Transition<T> {
        external::Transition {
            edge_vector: transition
                .edges
                .iter()
                .map(|edge| edge.reference.clone())
                .collect(),
            action_vector: transition.actions.clone(),
            action: transition.action.clone(),
            valuations: self
                .compiled_network
                .zone_compiler
                .externalize(transition.valuations.clone()),
        }
    }
}

/// A specialization of [Explorer] for MDPs using [NoClocks][time::NoClocks].
///
/// MDPs do not have any real-valued clocks.
pub type MDPExplorer = Explorer<time::NoClocks>;
