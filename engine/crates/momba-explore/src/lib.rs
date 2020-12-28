#![feature(min_const_generics)]

//! State space exploration engine for PTAs and MDPs with variables.
//!
//! This crate provides the necessary functionality for efficiently exploring the
//! state space of *Probabilistic Timed Automata* (PTAs) and *Markov Decision Processes*
//! (MDPs) augmented with variables.
//!
//!
//!  represented in *Momba's Intermediate Representation* (MombaIR).
//! MombaIR models are automata networks of the respective type of automata running
//! in parallel.
//!
//! For the purpose of state space exploration, a model is compiled into a more efficient
//! internal representation.

mod evaluate;

pub mod explore;
pub mod model;
pub mod time;
