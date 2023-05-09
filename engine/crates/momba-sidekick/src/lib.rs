//! A library that provides Statistical Model Checking for DTMC
//! and Deep Statistical Model Checking for MDP.
//!
//!
//! ## Example
//!
//! ```rust
//! use clock_zones::*;
//!
//! // create a DBM with three clock variables using `i64` as bound type
//! let mut zone: Dbm<i64> = Dbm::new_zero(3);
//!
//! // applies the *future operator* to the zone removing all upper bounds
//! zone.future();
//!
//! // the lower bound of the first variable is still `0` but there is no upper bound
//! assert_eq!(zone.get_lower_bound(Clock::variable(0)), Some(0));
//! assert_eq!(zone.get_upper_bound(Clock::variable(0)), None);
//! ```
//!
//! [discrete-time Markov chains]: https://en.wikipedia.org/wiki/Discrete-time_Markov_chain
//! [marcov decision process]: https://en.wikipedia.org/wiki/Markov_decision_process
//!
//! [name 1]:
//! [name 2]:
//!
//! [1]: ref 1
//! [2]: ref 2

/*
mod bounds;
mod clocks;
mod constants;
mod zones;

pub mod storage;

pub use clocks::{AnyClock, Clock, Variable};

pub use bounds::*;
pub use constants::*;
pub use zones::*;
*/
