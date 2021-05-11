//! State space exploration engine for PTAs and MDPs augmented with variables.
//!
//! This crate provides the necessary functionality for efficiently exploring the
//! state space of *Probabilistic Timed Automata* (PTAs) and *Markov Decision Processes*
//! (MDPs) augmented with variables.
//!
//! *Momba Explore* uses its own model representation defined in the module [model][model]
//! leveraging [Serde](https://serde.rs).
//! Hence, models can be loaded from any format supported by Serde.
//! [JANI models](https://jani-spec.org) can be loaded by first translating them using
//! [Momba](https://github.com/koehlma/momba).

mod explore;

pub mod model;
//pub mod simulate;
pub mod time;

pub use explore::*;
