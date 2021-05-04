//! Algorithms and data structures for representing time.

use std::{
    collections::HashSet,
    convert::{TryFrom, TryInto},
    env::var,
};

use num_traits::cast::FromPrimitive;

use indexmap::IndexSet;

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use clock_zones::{Bound, Clock, Variable, Zone};

use super::model;

#[derive(Clone, Eq, PartialEq, Hash, Serialize, Deserialize, Debug)]
pub struct ExternalZone {
    pub constraints: Vec<ExternalConstraint>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Hash, Debug)]
pub struct ExternalConstraint {
    pub left: model::Clock,
    pub right: model::Clock,
    pub is_strict: bool,
    pub bound: NotNan<f64>,
}

fn to_clock(clock: clock_zones::Clock, variables: &IndexSet<String>) -> model::Clock {
    clock
        .try_into()
        .map(|variable: clock_zones::Variable| model::Clock::Variable {
            identifier: variables
                .get_index(variable.number())
                .expect("there should be a name for the clock variable")
                .clone(),
        })
        .unwrap_or(model::Clock::Zero)
}

fn from_clock(clock: &model::Clock, variables: &IndexSet<String>) -> clock_zones::Clock {
    match clock {
        model::Clock::Zero => clock_zones::Clock::ZERO,
        model::Clock::Variable { identifier } => clock_zones::Clock::variable(
            variables
                .get_index_of(identifier)
                .expect("theres should be a clock variable with the provided name"),
        )
        .into(),
    }
}

fn externalize_zone<B: Bound, Z: Zone<B>>(zone: &Z, variables: &IndexSet<String>) -> ExternalZone
where
    B::Constant: Into<NotNan<f64>>,
{
    let mut constraints = Vec::new();
    for left in clock_zones::clocks(zone) {
        for right in clock_zones::clocks(zone) {
            let bound = zone.get_bound(left, right);
            if !bound.is_unbounded() {
                constraints.push(ExternalConstraint {
                    left: to_clock(left, variables),
                    right: to_clock(right, variables),
                    is_strict: bound.is_strict(),
                    bound: bound.constant().unwrap().into(),
                })
            }
        }
    }
    ExternalZone { constraints }
}

fn internalize_zone<B: Bound, Z: Zone<B>>(
    externalized: &ExternalZone,
    variables: &IndexSet<String>,
) -> Z
where
    B::Constant: From<NotNan<f64>>,
{
    let mut zone = Z::new_unconstrained(variables.len());
    for constraint in &externalized.constraints {
        zone.add_constraint(clock_zones::Constraint::new(
            from_clock(&constraint.left, variables),
            from_clock(&constraint.right, variables),
            clock_zones::Bound::new(constraint.is_strict, constraint.bound.into()),
        ))
    }
    zone
}

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
    type External: Serialize + Eq + PartialEq + std::hash::Hash + Clone;

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

    fn externalize(&self, valuations: &Self::Valuations) -> Self::External;

    fn internalize(&self, externalized: &Self::External) -> Self::Valuations;
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

    fn externalize(&self, valuations: &Self::Valuations) -> Self::External {
        valuations.clone()
    }

    fn internalize(&self, externalized: &Self::External) -> Self::Valuations {
        externalized.clone()
    }
}

/// A time representation using [f64] clock zones.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Float64Zone {
    clock_variables: IndexSet<String>,
}

impl Float64Zone {
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
    type Valuations = clock_zones::ZoneF64;

    type External = ExternalZone;

    type CompiledDifference = (clock_zones::Clock, clock_zones::Clock);

    type CompiledClocks = Vec<clock_zones::Clock>;

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
        (
            from_clock(left, &self.clock_variables),
            from_clock(right, &self.clock_variables),
        )
    }

    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks {
        clocks
            .iter()
            .map(|clock| from_clock(clock, &self.clock_variables))
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

    fn externalize(&self, valuations: &Self::Valuations) -> Self::External {
        externalize_zone(valuations, &self.clock_variables)
    }

    fn internalize(&self, externalized: &Self::External) -> Self::Valuations {
        internalize_zone(externalized, &self.clock_variables)
    }
}

/// A time representation using [f64] clock zones.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct GlobalTime {
    clock_variables: IndexSet<String>,
    global_clock: clock_zones::Variable,
}

impl GlobalTime {
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
    zone: clock_zones::Dbm<clock_zones::ConstantBound<ordered_float::NotNan<f64>>>,
    global_clock: clock_zones::Variable,
}

impl GlobalValuations {
    fn new_unconstrained(num_variables: usize) -> Self {
        GlobalValuations {
            zone: clock_zones::Dbm::new_unconstrained(num_variables),
            global_clock: clock_zones::Clock::variable(num_variables),
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

    type External = ExternalZone;

    type CompiledDifference = (clock_zones::Clock, clock_zones::Clock);

    type CompiledClocks = Vec<clock_zones::Clock>;

    fn new(network: &model::Network) -> Result<Self, String> {
        let mut variables = network.declarations.clock_variables.clone();
        variables.insert("__momba_explore_global".to_owned());
        Ok(GlobalTime {
            global_clock: clock_zones::Clock::variable(variables.len() - 1),
            clock_variables: variables,
        })
    }

    fn compile_difference(
        &self,
        left: &model::Clock,
        right: &model::Clock,
    ) -> Self::CompiledDifference {
        (
            from_clock(left, &self.clock_variables),
            from_clock(right, &self.clock_variables),
        )
    }

    fn compile_clocks(&self, clocks: &HashSet<model::Clock>) -> Self::CompiledClocks {
        clocks
            .iter()
            .map(|clock| from_clock(clock, &self.clock_variables))
            .collect()
    }

    fn is_empty(&self, valuations: &Self::Valuations) -> bool {
        valuations.zone.is_empty()
    }

    fn create_valuations(
        &self,
        constraints: Vec<Constraint<Self>>,
    ) -> Result<Self::Valuations, String> {
        let mut valuations = Self::Valuations::new_unconstrained(self.clock_variables.len());
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
            valuations.zone.reset(
                clock_zones::Variable::try_from(*clock).unwrap(),
                ordered_float::NotNan::new(0.0).unwrap(),
            );
        }
        valuations
    }

    /// Extrapolates the future of the given valuations.
    fn future(&self, mut valuations: Self::Valuations) -> Self::Valuations {
        valuations.zone.future();
        valuations
    }

    fn externalize(&self, valuations: &Self::Valuations) -> Self::External {
        externalize_zone(&valuations.zone, &self.clock_variables)
    }

    fn internalize(&self, externalized: &Self::External) -> Self::Valuations {
        let zone: clock_zones::Dbm<clock_zones::ConstantBound<ordered_float::NotNan<f64>>> =
            internalize_zone(externalized, &self.clock_variables);
        GlobalValuations {
            global_clock: clock_zones::Clock::variable(zone.num_variables() - 1),
            zone,
        }
    }
}
