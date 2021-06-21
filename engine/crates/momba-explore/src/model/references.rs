use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct AutomatonReference {
    pub name: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct LocationReference {
    pub automaton: AutomatonReference,
    pub name: String,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct EdgeReference {
    pub location: LocationReference,
    pub index: usize,
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct DestinationReference {
    pub edge: EdgeReference,
    pub index: usize,
}
