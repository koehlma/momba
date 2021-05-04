use std::cmp::min;
use std::iter::IntoIterator;

use crate::bounds::*;
use crate::clocks::*;
use crate::constants::*;
use crate::storage::*;

/// A *clock constraint* bounding the difference between two clocks.
pub struct Constraint<B: Bound> {
    pub(crate) left: Clock,
    pub(crate) right: Clock,
    pub(crate) bound: B,
}

impl<B: Bound> Constraint<B> {
    /// Returns the left-hand side of the difference.
    pub fn left(&self) -> Clock {
        self.left
    }

    /// Returns the right-hand side of the difference.
    pub fn right(&self) -> Clock {
        self.right
    }

    /// Returns the bound.
    pub fn bound(&self) -> &B {
        &self.bound
    }

    /// Constructs a new constraint.
    pub fn new(left: impl AnyClock, right: impl AnyClock, bound: B) -> Self {
        Self {
            left: left.as_clock(),
            right: right.as_clock(),
            bound,
        }
    }

    /// Constructs a new constraint of the form `clock ≤ constant`.
    pub fn new_le(clock: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: clock.as_clock(),
            right: Clock::ZERO,
            bound: B::new_le(constant),
        }
    }

    /// Constructs a new constraint of the form `clock < constant`.
    pub fn new_lt(clock: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: clock.as_clock(),
            right: Clock::ZERO,
            bound: B::new_lt(constant),
        }
    }

    /// Constructs a new constraint of the form `clock ≥ constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_ge(clock: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: Clock::ZERO,
            right: clock.as_clock(),
            bound: B::new_le(
                constant
                    .checked_neg()
                    .expect("overflow while negating constant"),
            ),
        }
    }

    /// Constructs a new constraint of the form `clock > constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_gt(clock: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: Clock::ZERO,
            right: clock.as_clock(),
            bound: B::new_lt(
                constant
                    .checked_neg()
                    .expect("overflow while negating constant"),
            ),
        }
    }
    /// Constructs a new constraint of the form `left - right ≤ constant`.
    pub fn new_diff_le(left: impl AnyClock, right: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: left.as_clock(),
            right: right.as_clock(),
            bound: B::new_le(constant),
        }
    }

    /// Constructs a new constraint of the form `left - right < constant`.
    pub fn new_diff_lt(left: impl AnyClock, right: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            left: left.as_clock(),
            right: right.as_clock(),
            bound: B::new_lt(constant),
        }
    }

    /// Constructs a new constraint of the form `left - right ≥ constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_diff_ge(left: impl AnyClock, right: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            // the swapped order is intentional
            left: right.as_clock(),
            right: left.as_clock(),
            bound: B::new_le(
                constant
                    .checked_neg()
                    .expect("overflow while negating constant"),
            ),
        }
    }

    /// Constructs a new constraint of the form `left - right > constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_diff_gt(left: impl AnyClock, right: impl AnyClock, constant: B::Constant) -> Self {
        Constraint {
            // the swapped order is intentional
            left: right.as_clock(),
            right: left.as_clock(),
            bound: B::new_lt(
                constant
                    .checked_neg()
                    .expect("overflow while negating constant"),
            ),
        }
    }
}

/// Represents a zone with a specific *[bound type][Bound]*.
pub trait Zone<B: Bound> {
    /// Constructs a new zone without any constraints besides clocks being positive.
    fn new_unconstrained(num_variables: usize) -> Self;
    /// Constructs a new zone where all clocks are set to zero.
    fn new_zero(num_variables: usize) -> Self;

    /// Constructs a new zone from the given iterable of [constraints][Constraint].
    fn with_constraints<U>(num_variables: usize, constraints: U) -> Self
    where
        U: IntoIterator<Item = Constraint<B>>;

    /// Returns the number of clock variables of this zone.
    fn num_variables(&self) -> usize;

    /// Returns the number of clocks of this zone.
    ///
    /// Note: This is always `num_variables + 1` as there is the constant
    /// zero clock plus the clock variables.
    fn num_clocks(&self) -> usize;

    /// Retrieves the difference bound for `left - right`.
    fn get_bound(&self, left: impl AnyClock, right: impl AnyClock) -> &B;

    /// Checks whether the zone is empty.
    fn is_empty(&self) -> bool;

    /// Adds a [constraint][Constraint] to the zone.
    fn add_constraint(&mut self, constraint: Constraint<B>);
    /// Adds all [constraints][Constraint] from the given iterable to the zone.
    fn add_constraints<U>(&mut self, constraints: U)
    where
        U: IntoIterator<Item = Constraint<B>>;

    /// Intersects two zones.
    fn intersect(&mut self, other: &Self);

    /// Computes the *future* of the zone by removing all upper bounds.
    ///
    /// This operation is sometimes also known as *up*.
    fn future(&mut self);
    /// Computes the *past* of the zone by removing all lower bounds.
    ///
    /// This operation is sometimes also known as *down*.
    fn past(&mut self);

    /// Resets the given clock variable to the specified constant.
    fn reset(&mut self, clock: Variable, value: B::Constant);

    /// Checks whether the value of the given clock is unbounded.
    fn is_unbounded(&self, clock: impl AnyClock) -> bool;

    /// Returns the upper bound for the value of the given clock.
    fn get_upper_bound(&self, clock: impl AnyClock) -> Option<B::Constant>;
    /// Returns the lower bound for the value of the given clock.
    fn get_lower_bound(&self, clock: impl AnyClock) -> Option<B::Constant>;

    /// Checks whether the given constraint is satisfied by the zone.
    fn is_satisfied(&self, constraint: Constraint<B>) -> bool;

    /// Checks whether the zone includes the other zone.
    fn includes(&self, other: &Self) -> bool;

    /// Creates a resized copy of the zone by adding or removing clocks.
    ///
    /// Added clocks will be unconstrained.
    fn resize(&self, num_variables: usize) -> Self;
}

/// Returns an iterator over the clocks of a zone.
#[inline(always)]
pub fn clocks<Z: Zone<B>, B: Bound>(zone: &Z) -> impl Iterator<Item = Clock> {
    (0..zone.num_clocks()).map(Clock)
}

/// Returns an iterator over the variables of a zone.
#[inline(always)]
pub fn variables<Z: Zone<B>, B: Bound>(zone: &Z) -> impl Iterator<Item = Variable> {
    (1..zone.num_clocks()).map(Variable)
}

/// An implementation of [Zone] as *difference bound matrix*.
///
/// Uses [LinearLayout] as the default [storage layout][Layout].
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Dbm<B: Bound, L: Layout<B> = LinearLayout<B>> {
    /// The dimension of the matrix.
    dimension: usize,
    /// The internal representation using the given layout.
    layout: L,

    _phantom_bound: std::marker::PhantomData<B>,
}

impl<B: Bound, L: Layout<B>> Dbm<B, L> {
    fn new(num_variables: usize, default: B) -> Self {
        let dimension = num_variables + 1;
        let mut layout = L::new(num_variables, default);
        unsafe {
            // This is safe because we know that the clock indices are
            // within the dimensions of the DBM.
            layout.set_unchecked(Clock::ZERO, Clock::ZERO, B::le_zero());
            for index in 1..dimension {
                let clock = Clock::from_index(index);
                layout.set_unchecked(Clock::ZERO, clock, B::le_zero());
                layout.set_unchecked(clock, clock, B::le_zero());
            }
        }
        Dbm {
            dimension,
            layout,
            _phantom_bound: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn check_clock(&self, clock: impl AnyClock) {
        assert!(clock.into_index() < self.dimension, "invalid clock index");
    }

    /// Canonicalize the DBM given that a particular clock has been touched.
    ///
    /// This method is unsafe because it does not check wether the index of the
    /// touched clock is within the dimensions of the DBM.
    ///
    /// Providing a clock which is not within the dimensions
    /// of the DBM is undefined behavior.
    #[inline(always)]
    unsafe fn canonicalize_touched(&mut self, touched: Clock) {
        for index in 0..self.dimension {
            let left = Clock::from_index(index);
            for index in 0..self.dimension {
                let right = Clock::from_index(index);
                let bound = self
                    .layout
                    .get(left, touched)
                    .add(self.layout.get_unchecked(touched, right))
                    .expect("overflow while adding bounds");
                if bound.is_tighter_than(self.layout.get(left, right)) {
                    self.layout.set_unchecked(left, right, bound);
                }
            }
        }
    }

    fn canonicalize(&mut self) {
        for index in 0..self.dimension {
            let touched = Clock::from_index(index);
            // This is safe because the touched clock is within
            // the dimensions of the DBM.
            unsafe { self.canonicalize_touched(touched) };
        }
    }
}

impl<B: Bound, L: Layout<B>> Zone<B> for Dbm<B, L> {
    fn new_unconstrained(num_variables: usize) -> Self {
        Dbm::new(num_variables, B::unbounded())
    }
    fn new_zero(num_variables: usize) -> Self {
        Dbm::new(num_variables, B::le_zero())
    }

    fn with_constraints<U>(num_variables: usize, constraints: U) -> Self
    where
        U: IntoIterator<Item = Constraint<B>>,
    {
        let mut zone = Self::new_unconstrained(num_variables);
        for constraint in constraints {
            zone.layout
                .set(constraint.left, constraint.right, constraint.bound.clone());
        }
        zone.canonicalize();
        zone
    }

    #[inline(always)]
    fn num_variables(&self) -> usize {
        self.dimension - 1
    }

    #[inline(always)]
    fn num_clocks(&self) -> usize {
        self.dimension
    }

    #[inline(always)]
    fn get_bound(&self, left: impl AnyClock, right: impl AnyClock) -> &B {
        self.layout.get(left, right)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        // This is safe because the zero clock is always present.
        unsafe { self.layout.get_unchecked(Clock::ZERO, Clock::ZERO) }
            .is_tighter_than(&B::le_zero())
    }

    fn add_constraint(&mut self, constraint: Constraint<B>) {
        self.check_clock(constraint.left);
        self.check_clock(constraint.right);
        let bound = self.layout.get(constraint.left, constraint.right);
        if constraint.bound.is_tighter_than(&bound) {
            unsafe {
                // This is safe because we just checked the clocks.
                self.layout
                    .set_unchecked(constraint.left, constraint.right, constraint.bound);
                self.canonicalize_touched(constraint.left);
                self.canonicalize_touched(constraint.right);
            }
        }
    }

    fn add_constraints<U>(&mut self, constraints: U)
    where
        U: IntoIterator<Item = Constraint<B>>,
    {
        for constraint in constraints {
            self.add_constraint(constraint);
        }
    }

    fn intersect(&mut self, other: &Self) {
        assert_eq!(
            self.dimension, other.dimension,
            "unable to intersect zones of different dimension"
        );
        for index in 0..self.dimension {
            let left = Clock::from_index(index);
            for index in 0..self.dimension {
                let right = Clock::from_index(index);
                unsafe {
                    // This is safe because we know that the clock indices are
                    // within the dimensions of the DBMs.
                    if other
                        .layout
                        .get_unchecked(left, right)
                        .is_tighter_than(&self.layout.get(left, right))
                    {
                        self.layout.set_unchecked(
                            left,
                            right,
                            other.layout.get(left, right).clone(),
                        );
                    }
                }
            }
        }
        self.canonicalize();
    }

    fn future(&mut self) {
        for index in 1..self.dimension {
            let left = Clock::from_index(index);
            // This is safe because we know that the clock index is
            // within the dimensions of the DBMs.
            unsafe { self.layout.set_unchecked(left, Clock::ZERO, B::unbounded()) };
        }
    }
    fn past(&mut self) {
        for index in 1..self.dimension {
            let right = Clock::from_index(index);
            self.layout.set(Clock::ZERO, right, B::le_zero());
            for index in 1..self.dimension {
                let left = Clock::from_index(index);
                if self
                    .layout
                    .get(left, right)
                    .is_tighter_than(self.layout.get(Clock::ZERO, right))
                {
                    self.layout
                        .set(Clock::ZERO, right, self.layout.get(left, right).clone());
                }
            }
        }
    }

    fn reset(&mut self, clock: Variable, value: B::Constant) {
        let le_pos_value = B::new_le(value.clone());
        let le_neg_value = B::new_le(value.checked_neg().unwrap());
        self.check_clock(clock);
        for index in 0..self.dimension {
            let other = Clock::from_index(index);
            // This is safe because the index of `other` is guaranteed to be
            // within the dimensions of the DBM and we checked `clock`.
            unsafe {
                self.layout.set_unchecked(
                    clock,
                    other,
                    self.layout
                        .get_unchecked(Clock::ZERO, other)
                        .add(&le_pos_value)
                        .unwrap(),
                );
                self.layout.set(
                    other,
                    clock,
                    self.layout
                        .get_unchecked(other, Clock::ZERO)
                        .add(&le_neg_value)
                        .unwrap(),
                );
            }
        }
    }

    fn is_unbounded(&self, clock: impl AnyClock) -> bool {
        self.layout.get(clock, Clock::ZERO).is_unbounded()
    }

    fn get_upper_bound(&self, clock: impl AnyClock) -> Option<B::Constant> {
        self.layout.get(clock, Clock::ZERO).constant()
    }

    fn get_lower_bound(&self, clock: impl AnyClock) -> Option<B::Constant> {
        Some(
            self.layout
                .get(Clock::ZERO, clock)
                .constant()?
                .checked_neg()
                .unwrap(),
        )
    }

    fn is_satisfied(&self, constraint: Constraint<B>) -> bool {
        !constraint
            .bound
            .is_tighter_than(self.layout.get(constraint.left, constraint.right))
    }

    fn includes(&self, other: &Self) -> bool {
        for index in 0..self.dimension {
            let left = Clock::from_index(index);
            for index in 0..self.dimension {
                let right = Clock::from_index(index);
                unsafe {
                    // This is safe because we know that the indices are within
                    // the dimensions of the DBM.
                    if self
                        .layout
                        .get_unchecked(left, right)
                        .is_tighter_than(other.layout.get(left, right))
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn resize(&self, num_variables: usize) -> Self {
        let mut other = Self::new_unconstrained(num_variables);
        for index in 0..min(self.dimension, other.dimension) {
            let left = Clock::from_index(index);
            for index in 0..min(self.dimension, other.dimension) {
                let right = Clock::from_index(index);
                // This is safe because we know that the clock indices
                // are within the dimensions of both DBMs.
                unsafe {
                    let bound = self.layout.get_unchecked(left, right);
                    other.layout.set_unchecked(left, right, bound.clone());
                }
            }
        }
        other.canonicalize();
        other
    }
}

/// A 32-bit signed integer zone.
pub type ZoneI32 = Dbm<i32>;

/// A 64-bit signed integer zone.
pub type ZoneI64 = Dbm<i64>;

/// A 32-bit floating-point zone.
#[cfg(feature = "float")]
pub type ZoneF32 = Dbm<ConstantBound<ordered_float::NotNan<f32>>>;

/// A 64-bit floating-point zone.
#[cfg(feature = "float")]
pub type ZoneF64 = Dbm<ConstantBound<ordered_float::NotNan<f64>>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let mut zone: Dbm<i64> = Dbm::new_zero(3);
        let x = Clock::variable(0);
        let y = Clock::variable(1);

        assert_eq!(zone.get_lower_bound(x), Some(0));
        assert_eq!(zone.get_upper_bound(x), Some(0));
        zone.future();
        assert_eq!(zone.get_lower_bound(x), Some(0));
        assert_eq!(zone.get_upper_bound(x), None);
        assert!(zone.is_unbounded(x));
        let mut copy = zone.clone();
        zone.add_constraint(Constraint::new_lt(x, 4));
        assert!(!zone.is_unbounded(x));
        assert_eq!(zone.get_lower_bound(x), Some(0));
        assert_eq!(zone.get_upper_bound(x), Some(4));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_le(x, 4));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_lt(x, 3));
        assert!(!copy.includes(&zone));
        assert!(zone.includes(&copy));
        zone.intersect(&copy);
        assert!(zone.includes(&copy));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_diff_lt(x, y, 4));
        copy.add_constraint(Constraint::new_diff_gt(x, y, 5));
        assert!(copy.is_empty());
    }
}
