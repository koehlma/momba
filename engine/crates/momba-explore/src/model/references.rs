use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct AutomatonReference {
    pub(crate) name: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct LocationReference {
    pub(crate) automaton: AutomatonReference,
    pub(crate) name: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct EdgeReference {
    pub(crate) location: LocationReference,
    pub(crate) index: usize,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct DestinationReference {
    pub(crate) edge: EdgeReference,
    pub(crate) index: usize,
}
