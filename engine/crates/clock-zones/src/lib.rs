//! A library for handling clock zones as they appear in the context of
//! [*Timed Automata*](https://link.springer.com/chapter/10.1007/BFb0031987).

pub mod zones;
pub use zones::*;

pub mod bounds;
pub use bounds::*;

pub mod constants;
pub use constants::*;
