//! Data structures for representing automata networks.

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};

use super::actions::*;
use super::expressions::*;
use super::references::*;
use super::types::*;
use super::values::*;

/// Represents a network of automata.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Network {
    pub declarations: Declarations,
    pub automata: IndexMap<String, Automaton>,
    pub links: Vec<Link>,
    pub initial_states: Vec<State>,
}

impl Network {
    pub fn get_index_of_global_variable(&self, identifier: &str) -> Option<usize> {
        self.declarations.global_variables.get_index_of(identifier)
    }

    pub fn get_index_of_transient_variable(&self, identifier: &str) -> Option<usize> {
        self.declarations
            .transient_variables
            .get_index_of(identifier)
    }

    pub fn get_automaton(&self, reference: &AutomatonReference) -> &Automaton {
        self.automata.get(&reference.name).unwrap()
    }

    pub fn get_location(&self, reference: &LocationReference) -> &Location {
        self.get_automaton(&reference.automaton)
            .locations
            .get(&reference.name)
            .unwrap()
    }

    pub fn get_edge(&self, reference: &EdgeReference) -> &Edge {
        self.get_location(&reference.location)
            .edges
            .get(reference.index)
            .unwrap()
    }

    pub fn get_destination(&self, reference: &DestinationReference) -> &Destination {
        self.get_edge(&reference.edge)
            .destinations
            .get(reference.index)
            .unwrap()
    }
}

/// The index of an action label relative to [Declarations].
pub type LabelIndex = usize;

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Declarations {
    pub global_variables: IndexMap<String, Type>,
    pub transient_variables: IndexMap<String, Expression>,
    pub clock_variables: IndexSet<String>,
    pub action_labels: IndexMap<String, Vec<Type>>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Automaton {
    pub locations: IndexMap<String, Location>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Location {
    pub invariant: IndexSet<ClockConstraint>,
    pub edges: Vec<Edge>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ClockConstraint {
    pub left: Clock,
    pub right: Clock,
    pub is_strict: bool,
    pub bound: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum Clock {
    Zero,
    Variable { identifier: String },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Edge {
    pub number: usize,
    pub pattern: ActionPattern,
    pub guard: Guard,
    pub destinations: Vec<Destination>,
    pub observations: Vec<Observation>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Observation {
    /// The label of the action type.
    pub label: String,
    /// The arguments of the action.
    pub arguments: Vec<Expression>,
    /// The probability with which the observation is observed.
    pub probability: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Guard {
    pub boolean_condition: Expression,
    pub clock_constraints: IndexSet<ClockConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Destination {
    pub location: String,
    pub probability: Expression,
    pub assignments: Vec<Assignment>,
    pub reset: IndexSet<Clock>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct State {
    pub values: IndexMap<String, Value>,
    pub locations: IndexMap<String, String>,
    pub zone: IndexSet<ClockConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Link {
    pub slots: IndexSet<String>,
    pub vector: IndexMap<String, LinkPattern>,
    pub result: LinkResult,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum LinkResult {
    Silent,
    Labeled(LinkPattern),
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct LinkPattern {
    pub action_type: String,
    pub arguments: Vec<String>,
}
