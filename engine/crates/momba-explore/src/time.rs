//! Algorithms and data structures for representing time.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::model;

/// An interface for dealing with different ways of representing time.
pub trait TimeType {
    /// Type used to represent potentially infinite sets of clock valuations.
    type Valuations: Clone;

    type CompiledClockConstraints;
    type CompiledClockSet;

    fn new(network: &model::Network) -> Self;

    fn compile_constraints(
        &self,
        constraints: &HashSet<model::ClockConstraint>,
    ) -> Self::CompiledClockConstraints;

    fn compile_clock_set(&self, clock_set: &HashSet<model::Clock>) -> Self::CompiledClockSet;

    fn create_zero(&self) -> Self::Valuations;

    fn constrain(&self, zone: &mut Self::Valuations, constraints: &Self::CompiledClockConstraints);

    fn is_empty(&self, zone: &Self::Valuations) -> bool;
}

/// A time representation not supporting any real-valued clocks.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct NoClocks();

impl TimeType for NoClocks {
    type Valuations = ();

    type CompiledClockConstraints = ();
    type CompiledClockSet = ();

    fn new(_: &model::Network) -> Self {
        NoClocks {}
    }

    fn compile_constraints(
        &self,
        constraints: &HashSet<model::ClockConstraint>,
    ) -> Self::CompiledClockConstraints {
        if constraints.len() > 0 {
            panic!("Clock constraints found!");
        }
        ()
    }

    fn compile_clock_set(&self, clock_set: &HashSet<model::Clock>) -> Self::CompiledClockSet {
        if clock_set.len() > 0 {
            panic!("Clocks found!");
        }
        ()
    }

    fn create_zero(&self) -> Self::Valuations {
        ()
    }

    fn constrain(
        &self,
        _zone: &mut Self::Valuations,
        _constraints: &Self::CompiledClockConstraints,
    ) {
    }

    fn is_empty(&self, _zone: &Self::Valuations) -> bool {
        false
    }
}
