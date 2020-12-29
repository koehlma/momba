//! Data structures for representing automata networks.
//!
//! This module define the structure of *Momba Intermediate Representation* (MombaIR) models.
//! The structure is defined directly in Rust using [Serde](https://serde.rs) and `derive`.
//! As a result, any format supported by Serde can be used to store MombaIR models.
//! Usually, however, MombaIR models are stored using the JSON format.
//!
//! MombaIR has been inspired by the [JANI](https://jani-spec.org) model interchange format.
//! In comparison to JANI, it gives up some higher-level convenience modeling features in favor of
//! simplicity and being more low-level.
//! MombaIR is not intended to be used directly for modeling.
//! Instead a higher-level modeling formalism such as JANI should be used which is then
//! translated to MombaIR.

mod actions;
mod expressions;
mod network;
mod types;
mod values;

pub use actions::*;
pub use expressions::*;
pub use network::*;
pub use types::*;
pub use values::*;
