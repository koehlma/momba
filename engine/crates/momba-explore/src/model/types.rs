//! Data structures for representing types.

use serde::{Deserialize, Serialize};

/// Possible data types of MombaIR values.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", tag = "type")]
pub enum Type {
    /// A 64-bit signed integer.
    Int64,
    /// A double-precision IEEE 754 non-NaN float.
    Float64,
    /// A boolean.
    Bool,
    /// A vector/array of values.
    Vector {
        /// The type of the elements of the vector.
        element_type: Box<Type>,
    },
    /// Indicates that the type is unknown.
    Unknown,
}
