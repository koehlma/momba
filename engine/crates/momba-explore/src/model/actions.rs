use serde::{Deserialize, Serialize};

use super::expressions::*;

/// Represents an *action pattern*.
///
/// Action patterns enable value passing.
/// A pattern is either *silent* or *labeled*.
/// Analogously to labeled actions, labeled patterns have a sequence of arguments
/// represented by [PatternArgument].
/// A pattern argument has a *direction* which is either *read* or *write*.
/// A read argument means that the pattern is able to receive a value via the
/// corresponding argument of an action.
/// A write argument means that the pattern will send a value via the corresponding
/// argument of an action.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "kind")]
pub enum ActionPattern {
    /// The silent pattern.
    Silent,
    /// A labeled pattern.
    Labeled(LabeledPattern),
}

/// Represents a labeled pattern.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct LabeledPattern {
    /// The label of the pattern.
    pub label: String,
    /// The pattern arguments.
    pub arguments: Vec<PatternArgument>,
}

/// Represents a pattern argument.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "direction")]
pub enum PatternArgument {
    Write { value: Expression },
    Read { identifier: String },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct WriteArgument {
    pub value: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct ReadArgument {
    pub identifier: String,
}
