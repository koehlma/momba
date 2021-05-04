//! A library for handling *[clock zones]* as they appear in the context of
//! *[timed automata]*.
//!
//! [Timed automata] have been pioneered by [Rajeev Alur] and [David Dill] in
//! 1994 to model real-time systems [[1]].
//! Timed automata extend finite automata with real-valued *clocks*.
//! This crate provides an implementation of the *[difference bound matrix]*
//! (DBM) data structure to efficiently represent [clock zones].
//! The implementation is mostly based on [[2]].
//!
//!
//! ## Architecture
//!
//! The trait [Zone] provides a general abstraction for clock zones.
//! The struct [Dbm] is the heart of this crate and implements the DBM data structure
//! using a variable *[bound type][Bound]* and *[storage layout][storage::Layout]*.
//! The storage layout determines how the bounds are stored while the bound type
//! determines the data structure used to store the individual bounds.
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
//! [clock zones]: https://en.wikipedia.org/wiki/Difference_bound_matrix#Zone
//! [timed automata]: https://en.wikipedia.org/wiki/Timed_automaton
//! [difference bound matrix]: https://en.wikipedia.org/wiki/Difference_bound_matrix
//!
//! [Rajeev Alur]: https://www.cis.upenn.edu/~alur/
//! [David Dill]: https://profiles.stanford.edu/david-dill
//!
//! [1]: https://www.cis.upenn.edu/~alur/TCS94.pdf
//! [2]: https://doi.org/10.1007/978-3-540-27755-2_3

mod bounds;
mod clocks;
mod constants;
mod zones;

pub mod storage;

pub use clocks::{AnyClock, Clock, Variable};

pub use bounds::*;
pub use constants::*;
pub use zones::*;
