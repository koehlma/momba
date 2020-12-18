use serde::{Deserialize, Serialize};

use super::compiled::*;
use super::model::*;
use super::values::*;

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct BareState {
    pub(crate) values: Box<[Value]>,
    pub(crate) locations: Box<[usize]>,
}

impl BareState {
    pub fn new(network: &Network, state: &State) -> Result<Self, String> {
        Ok(BareState {
            values: network
                .variables
                .keys()
                .map(|identifier| {
                    state
                        .values
                        .get(identifier)
                        .map(|value| value.clone())
                        .ok_or_else(|| format!("Missing value for variable `{}`.", identifier))
                })
                .collect::<Result<Box<[Value]>, _>>()?,
            locations: network
                .automata
                .iter()
                .map(|(automaton_name, automaton)| {
                    automaton
                        .locations
                        .get_index_of(
                            state.locations.get(automaton_name).ok_or_else(|| {
                                format!("No automaton named `{}`", automaton_name)
                            })?,
                        )
                        .ok_or_else(|| {
                            format!(
                                "No does have no location named `{}`",
                                state.locations.get(automaton_name).unwrap()
                            )
                        })
                })
                .collect::<Result<Box<[usize]>, _>>()?,
        })
    }

    pub fn state(&self, network: &Network) -> State {
        State {
            values: self
                .values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    (
                        network.variables.get_index(index).unwrap().0.clone(),
                        value.clone(),
                    )
                })
                .collect(),
            locations: self
                .locations
                .iter()
                .enumerate()
                .map(|(index, location_index)| {
                    let (automaton_name, automaton) = network.automata.get_index(index).unwrap();
                    (
                        automaton_name.clone(),
                        automaton
                            .locations
                            .get_index(*location_index)
                            .unwrap()
                            .0
                            .clone(),
                    )
                })
                .collect(),
        }
    }

    pub fn into_compiled(self, compiled_network: &CompiledNetwork) -> CompiledState {
        CompiledState {
            network: compiled_network,
            values: self.values,
            locations: self.locations,
        }
    }
}

impl Into<BareState> for CompiledState<'_> {
    fn into(self) -> BareState {
        BareState {
            values: self.values,
            locations: self.locations,
        }
    }
}

pub fn initial_states(compiled_network: &CompiledNetwork) -> Result<Vec<BareState>, String> {
    compiled_network
        .network
        .initial
        .iter()
        .map(|state| BareState::new(&compiled_network.network, state))
        .collect()
}
