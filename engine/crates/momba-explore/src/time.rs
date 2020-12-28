//! Algorithms and data structures for representing time.

use std::collections::HashSet;

use super::model;

pub trait ZoneCompiler {
    type Zone: Clone;

    type CompiledClockConstraints;
    type CompiledClockSet;

    fn new(network: &model::Network) -> Self;

    fn compile_constraints(
        &self,
        constraints: &HashSet<model::ClockConstraint>,
    ) -> Self::CompiledClockConstraints;

    fn compile_clock_set(&self, clock_set: &HashSet<model::Clock>) -> Self::CompiledClockSet;

    fn create_zero(&self) -> Self::Zone;

    fn constrain(&self, zone: &mut Self::Zone, constraints: &Self::CompiledClockConstraints);

    fn is_empty(&self, zone: &Self::Zone) -> bool;
}

impl ZoneCompiler for () {
    type Zone = ();

    type CompiledClockConstraints = ();
    type CompiledClockSet = ();

    fn new(_: &model::Network) -> Self {
        ()
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

    fn create_zero(&self) -> Self::Zone {
        ()
    }

    fn constrain(&self, _zone: &mut Self::Zone, _constraints: &Self::CompiledClockConstraints) {}

    fn is_empty(&self, _zone: &Self::Zone) -> bool {
        false
    }
}
