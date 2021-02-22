//! Algorithms and data structures for representing time.

use std::collections::HashSet;

use num_traits::cast::FromPrimitive;

use indexmap::IndexSet;

use serde::{Deserialize, Serialize};

use clock_zones;
use clock_zones::Zone;

use crate::Explorer;

use super::model;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Constraint<T: TimeType> {
    pub(crate) difference: T::CompiledDifference,
    pub(crate) is_strict: bool,
    pub(crate) bound: model::Value,
}

/// An interface for dealing with different ways of representing time.
pub trait TimeType: Sized {
    /// Type used to represent potentially infinite sets of clock valuations.
    type Valuations: Eq + PartialEq + std::hash::Hash + Clone;

    /// Type used to represent partially infinite sets of clock valuations externally.
    type External: Eq + PartialEq + std::hash::Hash + Clone;

    /// Type used to represent the difference between two clocks.
    type CompiledDifference: Clone;

    /// Type used to represent a compiled set of clocks.
    type CompiledClocks: Clone;

    /// Crates a new instance of [TimeType] for the given network.
    fn new(network: &model::Network) -> Result<Self, String>;

    /// Takes two clocks and returns a compiled difference between `left` and `right`.
    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference;

    /// Takes a set of clocks and returns a compiled set of clocks.
    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks;

    /// Checks the provided set of valuations is empty.
    fn is_empty(&self, valuations: &Self::Valuations) -> bool;

    /// Creates a set of valuations based on the given constraints.
    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String>;

    /// Constrain a set of valuations with the given constraint.
    fn constrain(
        &self,
        valuations: Self::Valuations,
        difference: &Self::CompiledDifference,
        is_strict: bool,
        bound: model::Value,
    ) -> Self::Valuations;

    /// Resets the clocks of the given set to 0.
    fn reset(
        &self,
        valuations: Self::Valuations,
        clocks: &Self::CompiledClocks,
    ) -> Self::Valuations;

    /// Extrapolates the future of the given valuations.
    fn future(&self, valuations: Self::Valuations) -> Self::Valuations;

    fn externalize(&self, valuations: Self::Valuations) -> Self::External;
}

/// A time representation not supporting any real-valued clocks.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct NoClocks();

impl TimeType for NoClocks {
    type Valuations = ();

    type External = ();

    type CompiledDifference = ();

    type CompiledClocks = ();

    fn new(network: &model::Network) -> Result<Self, String> {
        if network.declarations.clock_variables.len() > 0 {
            Err("time type `NoClocks` does not allow any clocks".to_string())
        } else {
            Ok(NoClocks())
        }
    }

    fn compile_difference(
        &self,
        _left: &model::Clock,
        _right: &model::Clock,
    ) -> Self::CompiledDifference {
        panic!("time type `NoClocks` does not allow any clocks")
    }

    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks {
        if clocks.len() > 0 {
            panic!("time type `NoClocks` does not allow any clocks")
        }
    }

    fn is_empty(&self, _valuations: &Self::Valuations) -> bool {
        false
    }

    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String> {
        if constraints.len() > 0 {
            Err("time type `NoClocks` does not allow any clocks".to_string())
        } else {
            Ok(())
        }
    }

    fn constrain(
        &self,
        _valuations: Self::Valuations,
        _difference: &Self::CompiledDifference,
        _is_strict: bool,
        _bound: model::Value,
    ) -> Self::Valuations {
        panic!("time type `NoClocks` does not allow any clocks")
    }

    fn reset(
        &self,
        _valuations: Self::Valuations,
        _clocks: &Self::CompiledClocks,
    ) -> Self::Valuations {
        ()
    }

    fn future(&self, _valuations: Self::Valuations) -> Self::Valuations {
        ()
    }

    fn externalize(&self, valuations: Self::Valuations) -> Self::External {
        valuations
    }
}

/// A time representation using [f64] clock zones.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Float64Zone {
    clock_variables: IndexSet<String>,
}

impl Float64Zone {
    fn clock_to_index(&self, clock: &model::Clock) -> usize {
        match clock {
            model::Clock::Zero => 0,
            model::Clock::Variable { identifier } => {
                self.clock_variables.get_index_of(identifier).unwrap() + 1
            }
        }
    }

    fn apply_constraint(
        &self,
        zone: &mut <Self as TimeType>::Valuations,
        constraint: Constraint<Self>,
    ) {
        let bound = match constraint.bound {
            model::Value::Int64(value) => ordered_float::NotNan::from_i64(value).unwrap(),
            model::Value::Float64(value) => value,
            _ => panic!("unable to convert {:?} to clock bound", constraint.bound),
        };
        if constraint.is_strict {
            zone.add_constraint(clock_zones::Constraint::new_diff_lt(
                constraint.difference.0,
                constraint.difference.1,
                bound,
            ));
        } else {
            zone.add_constraint(clock_zones::Constraint::new_diff_le(
                constraint.difference.0,
                constraint.difference.1,
                bound,
            ));
        }
    }
}

impl TimeType for Float64Zone {
    type Valuations = clock_zones::DBM<clock_zones::ConstantBound<ordered_float::NotNan<f64>>>;

    type External = Self::Valuations;

    type CompiledDifference = (usize, usize);

    type CompiledClocks = Vec<usize>;

    fn new(network: &model::Network) -> Result<Self, String> {
        Ok(Float64Zone {
            clock_variables: network.declarations.clock_variables.clone(),
        })
    }

    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference {
        (self.clock_to_index(left), self.clock_to_index(right))
    }

    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks {
        clocks
            .iter()
            .map(|clock| self.clock_to_index(clock))
            .collect()
    }

    fn is_empty(&self, valuations: &Self::Valuations) -> bool {
        valuations.is_empty()
    }

    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String> {
        let mut valuations = Self::Valuations::new_unconstrained(self.clock_variables.len());
        for constraint in constraints {
            self.apply_constraint(&mut valuations, constraint);
        }
        Ok(valuations)
    }

    fn constrain(
        &self,
        mut valuations: Self::Valuations,
        difference: &Self::CompiledDifference,
        is_strict: bool,
        bound: model::Value,
    ) -> Self::Valuations {
        self.apply_constraint(
            &mut valuations,
            Constraint {
                difference: difference.clone(),
                is_strict,
                bound,
            },
        );
        valuations
    }

    fn reset(
        &self,
        mut valuations: Self::Valuations,
        clocks: &Self::CompiledClocks,
    ) -> Self::Valuations {
        for clock in clocks {
            valuations.reset(*clock, ordered_float::NotNan::new(0.0).unwrap());
        }
        valuations
    }

    /// Extrapolates the future of the given valuations.
    fn future(&self, mut valuations: Self::Valuations) -> Self::Valuations {
        valuations.future();
        valuations
    }

    fn externalize(&self, valuations: Self::Valuations) -> Self::External {
        valuations
    }
}

/// A time representation using [f64] clock zones.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct GlobalTime {
    clock_variables: IndexSet<String>,
    global_clock: usize,
}

impl GlobalTime {
    fn clock_to_index(&self, clock: &model::Clock) -> usize {
        match clock {
            model::Clock::Zero => 0,
            model::Clock::Variable { identifier } => {
                self.clock_variables.get_index_of(identifier).unwrap() + 1
            }
        }
    }

    fn apply_constraint(
        &self,
        valuations: &mut <Self as TimeType>::Valuations,
        constraint: Constraint<Self>,
    ) {
        let bound = match constraint.bound {
            model::Value::Int64(value) => ordered_float::NotNan::from_i64(value).unwrap(),
            model::Value::Float64(value) => value,
            _ => panic!("unable to convert {:?} to clock bound", constraint.bound),
        };
        if constraint.is_strict {
            valuations
                .zone
                .add_constraint(clock_zones::Constraint::new_diff_lt(
                    constraint.difference.0,
                    constraint.difference.1,
                    bound,
                ));
        } else {
            valuations
                .zone
                .add_constraint(clock_zones::Constraint::new_diff_le(
                    constraint.difference.0,
                    constraint.difference.1,
                    bound,
                ));
        }
    }
}
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct GlobalValuations {
    zone: clock_zones::DBM<clock_zones::ConstantBound<ordered_float::NotNan<f64>>>,
    global_clock: usize,
}

impl GlobalValuations {
    fn new_unconstrained(num_clocks: usize) -> Self {
        GlobalValuations {
            zone: clock_zones::DBM::new_unconstrained(num_clocks),
            global_clock: num_clocks,
        }
    }

    pub fn global_time_lower_bound(&self) -> Option<ordered_float::NotNan<f64>> {
        self.zone.get_lower_bound(self.global_clock)
    }

    pub fn global_time_upper_bound(&self) -> Option<ordered_float::NotNan<f64>> {
        self.zone.get_upper_bound(self.global_clock)
    }

    pub fn set_global_time(&mut self, value: ordered_float::NotNan<f64>) {
        self.zone
            .add_constraint(clock_zones::Constraint::new_le(self.global_clock, value));
        self.zone
            .add_constraint(clock_zones::Constraint::new_ge(self.global_clock, value));
    }
}

impl TimeType for GlobalTime {
    type Valuations = GlobalValuations;

    type External = Self::Valuations;

    type CompiledDifference = (usize, usize);

    type CompiledClocks = Vec<usize>;

    fn new(network: &model::Network) -> Result<Self, String> {
        Ok(GlobalTime {
            clock_variables: network.declarations.clock_variables.clone(),
            global_clock: network.declarations.clock_variables.len() + 1,
        })
    }

    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference {
        (self.clock_to_index(left), self.clock_to_index(right))
    }

    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks {
        clocks
            .iter()
            .map(|clock| self.clock_to_index(clock))
            .collect()
    }

    fn is_empty(&self, valuations: &Self::Valuations) -> bool {
        valuations.zone.is_empty()
    }

    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String> {
        let mut valuations = Self::Valuations::new_unconstrained(self.clock_variables.len() + 1);
        for constraint in constraints {
            self.apply_constraint(&mut valuations, constraint);
        }
        valuations
            .zone
            .reset(self.global_clock, ordered_float::NotNan::new(0.0).unwrap());
        Ok(valuations)
    }

    fn constrain(
        &self,
        mut valuations: Self::Valuations,
        difference: &Self::CompiledDifference,
        is_strict: bool,
        bound: model::Value,
    ) -> Self::Valuations {
        self.apply_constraint(
            &mut valuations,
            Constraint {
                difference: difference.clone(),
                is_strict,
                bound,
            },
        );
        valuations
    }

    fn reset(
        &self,
        mut valuations: Self::Valuations,
        clocks: &Self::CompiledClocks,
    ) -> Self::Valuations {
        for clock in clocks {
            valuations
                .zone
                .reset(*clock, ordered_float::NotNan::new(0.0).unwrap());
        }
        valuations
    }

    /// Extrapolates the future of the given valuations.
    fn future(&self, mut valuations: Self::Valuations) -> Self::Valuations {
        valuations.zone.future();
        valuations
    }

    fn externalize(&self, valuations: Self::Valuations) -> Self::External {
        valuations
    }
}
