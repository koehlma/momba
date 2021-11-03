//! Algorithms and data structures for representing time.

use std::{
    convert::{TryFrom, TryInto},
    env::var,
    fmt::Debug,
};

use num_traits::cast::FromPrimitive;

use indexmap::IndexSet;

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use clock_zones::{Bound, Clock, Variable, Zone};

use super::model;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Constraint<T: Time> {
    pub(crate) difference: T::CompiledDifference,
    pub(crate) is_strict: bool,
    pub(crate) bound: model::Value,
}

/// An interface for dealing with different ways of representing time.
pub trait Time: Sized + Sync + Send + Debug {
    /// Type used to represent potentially infinite sets of clock valuations.
    type Valuations: Eq + PartialEq + std::hash::Hash + Clone + Sync + Send + Debug;

    /// Type used to represent the difference between two clocks.
    type CompiledDifference: Clone + Sync + Send;

    /// Type used to represent a compiled set of clocks.
    type CompiledClocks: Clone + Sync + Send;

    /// Crates a new instance of [TimeType] for the given network.
    fn new(network: &model::Network) -> Result<Self, String>;

    /// Takes two clocks and returns a compiled difference between `left` and `right`.
    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference;

    /// Takes a set of clocks and returns a compiled set of clocks.
    fn compile_clocks(&self, clocks: &IndexSet<model::Clock>) -> Self::CompiledClocks;

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
}

/// A time representation not supporting any real-valued clocks.
#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
pub struct NoClocks();

impl Time for NoClocks {
    type Valuations = ();

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

    fn compile_clocks(&self, clocks: &IndexSet<model::Clock>) -> Self::CompiledClocks {
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
}

/// A time representation using [f64] clock zones.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Float64Zone {
    variables: IndexSet<String>,
}

impl Float64Zone {
    fn model_to_zone_clock(&self, clock: &model::Clock) -> clock_zones::Clock {
        match clock {
            model::Clock::Zero => clock_zones::Clock::ZERO,
            model::Clock::Variable { identifier } => clock_zones::Clock::variable(
                self.variables
                    .get_index_of(identifier)
                    .unwrap_or_else(|| panic!("clock `{}` not found", identifier)),
            )
            .into(),
        }
    }

    fn apply_constraint(
        &self,
        zone: &mut <Self as Time>::Valuations,
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

impl Time for Float64Zone {
    type Valuations = clock_zones::ZoneF64;

    type CompiledDifference = (clock_zones::Clock, clock_zones::Clock);

    type CompiledClocks = Vec<clock_zones::Clock>;

    fn new(network: &model::Network) -> Result<Self, String> {
        Ok(Float64Zone {
            variables: network.declarations.clock_variables.clone(),
        })
    }

    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference {
        (
            self.model_to_zone_clock(left),
            self.model_to_zone_clock(right),
        )
    }

    fn compile_clocks(&self, clocks: &IndexSet<model::Clock>) -> Self::CompiledClocks {
        clocks
            .iter()
            .map(|clock| self.model_to_zone_clock(clock))
            .collect()
    }

    fn is_empty(&self, valuations: &Self::Valuations) -> bool {
        valuations.is_empty()
    }

    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String> {
        let mut valuations = Self::Valuations::new_zero(self.variables.len());
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
            valuations.reset(
                clock_zones::Variable::try_from(*clock).unwrap(),
                ordered_float::NotNan::new(0.0).unwrap(),
            );
        }
        valuations
    }

    /// Extrapolates the future of the given valuations.
    fn future(&self, mut valuations: Self::Valuations) -> Self::Valuations {
        valuations.future();
        valuations
    }
}
