use serde::{Deserialize, Serialize};

use super::model::*;

/// An *action* which is either [*silent*][Action::Silent] or [*labeled*][Action::Labeled].
///
/// An action is either the *silent action* or a *labeled action* instantiated with
/// a sequence of values coined *arguments*.
/// While action labels are generally represented as strings, we internally store
/// the index into the [action label declarations][Declarations] of the [Network].
/// Hence, the [label index][Action::label_index] is only meaningful with respect to
/// a particular instance of [Network].
/// The string action label can be retrieved via [LabeledAction::label] providing
/// the network the action belongs to.
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub enum Action {
    /// The silent action.
    Silent,
    /// A labeled action.
    Labeled(LabeledAction),
}

const EMPTY_ARGUMENTS: [Value; 0] = [];

impl Action {
    /// Creates a new labeled action.
    pub(crate) fn new(label: LabelIndex, arguments: Box<[Value]>) -> Self {
        Action::Labeled(LabeledAction::new(label, arguments))
    }

    /// Returns `true` if the action is silent.
    pub fn is_silent(&self) -> bool {
        matches!(self, Action::Silent)
    }

    /// Returns `true` if the action is labeled.
    pub fn is_labeled(&self) -> bool {
        matches!(self, Action::Labeled(_))
    }

    /// Returns the index of the action's label or [None][None] if the action is silent.
    pub fn label_index(&self) -> Option<LabelIndex> {
        match self {
            Action::Silent => None,
            Action::Labeled(labeled) => Some(labeled.label),
        }
    }

    /// Returns a slice representing the arguments of the action.
    ///
    /// For the silent action the slice is empty.
    pub fn arguments(&self) -> &[Value] {
        match self {
            Action::Silent => &EMPTY_ARGUMENTS,
            Action::Labeled(labeled) => labeled.arguments(),
        }
    }
}

/// Label and arguments associated with a [labeled action](Action::Labeled).
#[derive(Serialize, Deserialize, Clone, Hash, Eq, PartialEq, Debug)]
pub struct LabeledAction {
    /// The label of the action.
    pub(crate) label: LabelIndex,
    /// The arguments of the action.
    pub(crate) arguments: Box<[Value]>,
}

impl LabeledAction {
    /// Crates a new labeled action.
    pub(crate) fn new(label: LabelIndex, arguments: Box<[Value]>) -> Self {
        LabeledAction { label, arguments }
    }

    pub fn new_with_network(network: &Network, label: &str, arguments: Box<[Value]>) -> Self {
        LabeledAction {
            label: network
                .declarations
                .action_labels
                .get_index_of(label)
                .unwrap(),
            arguments,
        }
    }

    /// Returns the index of the action's label.
    pub fn label_index(&self) -> LabelIndex {
        self.label
    }

    /// Retrieves the name of the action's label from the network.
    pub fn label<'n>(&self, network: &'n Network) -> Option<&'n String> {
        network
            .declarations
            .action_labels
            .get_index(self.label)
            .map(|(action_name, _)| action_name)
    }

    /// Returns a slice representing the arguments of the action.
    pub fn arguments(&self) -> &[Value] {
        &self.arguments
    }
}

impl Into<Action> for LabeledAction {
    fn into(self) -> Action {
        Action::Labeled(self)
    }
}
