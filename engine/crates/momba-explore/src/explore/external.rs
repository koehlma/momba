//! Data structures for communicating with external tools.

use serde::{Deserialize, Serialize};
use time::TimeType;

use crate::model;
use crate::time;

use super::actions::Action;

/// Represents a *joint transition* of an automaton network.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Transition<T: time::TimeType> {
    /// The edges of the participating automata instances.
    pub(crate) edge_vector: Box<[model::EdgeReference]>,
    /// The actions with with the respective automata participate.
    pub(crate) action_vector: Box<[Action]>,
    /// The action resulting from synchronization.
    pub(crate) action: Action,
    /// The clock valuations valid for the transition.
    pub(crate) valuations: T::External,
}

// impl<T: time::TimeType> Transition<T> {
//     /// Returns a JSON string representing the transition.
//     pub fn json(&self) -> String {
//         serde_json::to_string(self).unwrap()
//     }
// }
