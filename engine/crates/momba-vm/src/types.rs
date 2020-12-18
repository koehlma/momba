use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "type")]
pub enum Type {
    Int64,
    Float64,
    Bool,
    Vector { element_type: Box<Type> },
    Unknown,
}
