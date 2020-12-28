//! Data structures for representing automata networks.

use std::collections::{HashMap, HashSet};

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};

use super::expressions::*;
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
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Declarations {
    pub global_variables: IndexMap<String, Type>,
    pub transient_variables: IndexMap<String, Expression>,
    pub clock_variables: IndexSet<String>,
    pub action_types: IndexMap<String, Vec<Type>>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Automaton {
    pub locations: IndexMap<String, Location>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Location {
    pub invariant: HashSet<ClockConstraint>,
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
    pub pattern: ActionPattern,
    pub guard: Guard,
    pub destinations: Vec<Destination>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum ActionPattern {
    Internal,
    Link {
        action_type: String,
        arguments: Vec<PatternArgument>,
    },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "direction")]
pub enum PatternArgument {
    Write { value: Expression },
    Read { identifier: String },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Guard {
    pub boolean_condition: Expression,
    pub clock_constraints: HashSet<ClockConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Destination {
    pub location: String,
    pub probability: Expression,
    pub assignments: Vec<Assignment>,
    pub reset: HashSet<Clock>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct State {
    pub values: HashMap<String, Value>,
    pub locations: HashMap<String, String>,
    pub zone: HashSet<ClockConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Link {
    pub slots: IndexSet<String>,
    pub vector: IndexMap<String, LinkPattern>,
    pub result: LinkResult,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct LinkPattern {
    pub action_type: String,
    pub arguments: Vec<String>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum LinkResult {
    Internal,
    Pattern(LinkPattern),
}
