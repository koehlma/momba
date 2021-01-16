use std::cmp::min;
use std::iter::IntoIterator;

use crate::bounds::*;
use crate::constants::*;
use crate::storage::*;

/// Represents a *clock*.
///
/// Note that `0` is the designated *zero clock* whose value is always `0`.
pub type Clock = usize;

/// A *clock constraint* bounding the difference of two clocks.
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

    /// Constructs a new constraint of the form `clock ≤ constant`.
    pub fn new_le(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: clock,
            right: 0,
            bound: B::new_le(constant),
        }
    }

    /// Constructs a new constraint of the form `clock < constant`.
    pub fn new_lt(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: clock,
            right: 0,
            bound: B::new_lt(constant),
        }
    }

    /// Constructs a new constraint of the form `clock ≥ constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_ge(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: 0,
            right: clock,
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
    pub fn new_gt(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: 0,
            right: clock,
            bound: B::new_lt(
                constant
                    .checked_neg()
                    .expect("overflow while negating constant"),
            ),
        }
    }
    /// Constructs a new constraint of the form `left - right ≤ constant`.
    pub fn new_diff_le(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left,
            right,
            bound: B::new_le(constant),
        }
    }

    /// Constructs a new constraint of the form `left - right < constant`.
    pub fn new_diff_lt(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left,
            right,
            bound: B::new_lt(constant),
        }
    }

    /// Constructs a new constraint of the form `left - right ≥ constant`.
    ///
    /// Panics in case the constant cannot be negated without an overflow.
    pub fn new_diff_ge(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            // the swapped order is intentional
            left: right,
            right: left,
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
    pub fn new_diff_gt(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            // the swapped order is intentional
            left: right,
            right: left,
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
    fn new_unconstrained(num_clocks: usize) -> Self;
    /// Constructs a new zone where all clocks are set to zero.
    fn new_zero(num_clocks: usize) -> Self;

    /// Constructs a new zone from the given iterable of [constraints][Constraint].
    fn with_constraints<U>(num_clocks: usize, constraints: U) -> Self
    where
        U: IntoIterator<Item = Constraint<B>>;

    /// Returns the number of clocks of this zone.
    fn num_clocks(&self) -> usize;

    /// Retrieves the difference bound for `left - right`.
    fn get_bound(&self, left: Clock, right: Clock) -> &B;

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

    /// Resets the given clock to the specified constant.
    fn reset(&mut self, clock: Clock, value: B::Constant);

    /// Checks whether the value of the given clock is unbounded.
    fn is_unbounded(&self, clock: Clock) -> bool;

    /// Returns the upper bound for the value of the given clock.
    fn get_upper_bound(&self, clock: Clock) -> Option<B::Constant>;
    /// Returns the lower bound for the value of the given clock.
    fn get_lower_bound(&self, clock: Clock) -> Option<B::Constant>;

    /// Checks whether the given constraint is satisfied by the zone.
    fn is_satisfied(&self, constraint: Constraint<B>) -> bool;

    /// Checks whether the zone includes the other zone.
    fn includes(&self, other: &Self) -> bool;

    /// Creates a resized copy of the zone by adding or removing clocks.
    ///
    /// Added clocks will be unconstrained.
    fn resize(&self, num_clocks: usize) -> Self;
}

/// An implementation of [Zone] as *difference bound matrix*.
///
/// Uses [LinearLayout] as the default [storage layout][Layout].
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DBM<B: Bound, L: Layout<B> = LinearLayout<B>> {
    /// The dimension of the matrix.
    dimension: usize,
    /// The internal representation using the given layout.
    layout: L,

    _phantom_bound: std::marker::PhantomData<B>,
}

impl<B: Bound, L: Layout<B>> DBM<B, L> {
    fn new(num_clocks: usize, default: B) -> Self {
        let dimension = num_clocks + 1;
        let mut layout = L::new(num_clocks, default);
        layout.set(0, 0, B::le_zero());
        for clock in 1..dimension {
            layout.set(0, clock, B::le_zero());
            layout.set(clock, clock, B::le_zero());
        }
        DBM {
            dimension,
            layout,
            _phantom_bound: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn canonicalize_touched(&mut self, touched: usize) {
        for left in 0..self.dimension {
            for right in 0..self.dimension {
                let bound = self
                    .layout
                    .get(left, touched)
                    .add(self.layout.get(touched, right))
                    .unwrap();
                if bound.is_tighter_than(self.layout.get(left, right)) {
                    self.layout.set(left, right, bound);
                }
            }
        }
    }

    fn canonicalize(&mut self) {
        for touched in 0..self.dimension {
            self.canonicalize_touched(touched);
        }
    }
}

impl<B: Bound, L: Layout<B>> Zone<B> for DBM<B, L> {
    fn new_unconstrained(num_clocks: usize) -> Self {
        DBM::new(num_clocks, B::unbounded())
    }
    fn new_zero(num_clocks: usize) -> Self {
        DBM::new(num_clocks, B::le_zero())
    }

    fn with_constraints<U>(num_clocks: usize, constraints: U) -> Self
    where
        U: IntoIterator<Item = Constraint<B>>,
    {
        let mut zone = Self::new_unconstrained(num_clocks);
        for constraint in constraints {
            zone.layout
                .set(constraint.left, constraint.right, constraint.bound.clone());
        }
        zone.canonicalize();
        zone
    }

    #[inline(always)]
    fn num_clocks(&self) -> usize {
        self.dimension - 1
    }

    #[inline(always)]
    fn get_bound(&self, left: usize, right: usize) -> &B {
        self.layout.get(left, right)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.layout.get(0, 0).is_tighter_than(&B::le_zero())
    }

    fn add_constraint(&mut self, constraint: Constraint<B>) {
        let bound = self.layout.get(constraint.left, constraint.right);
        if constraint.bound.is_tighter_than(&bound) {
            self.layout
                .set(constraint.left, constraint.right, constraint.bound);
            self.canonicalize_touched(constraint.left);
            self.canonicalize_touched(constraint.right);
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
        assert_eq!(self.dimension, other.dimension);
        for left in 0..self.dimension {
            for right in 0..self.dimension {
                if other
                    .layout
                    .get(left, right)
                    .is_tighter_than(&self.layout.get(left, right))
                {
                    self.layout
                        .set(left, right, other.layout.get(left, right).clone());
                }
            }
        }
        self.canonicalize();
    }

    fn future(&mut self) {
        for left in 1..self.dimension {
            self.layout.set(left, 0, B::unbounded());
        }
    }
    fn past(&mut self) {
        for right in 1..self.dimension {
            self.layout.set(0, right, B::le_zero());
            for row in 1..self.dimension {
                if self
                    .layout
                    .get(row, right)
                    .is_tighter_than(self.layout.get(0, right))
                {
                    self.layout
                        .set(0, right, self.layout.get(row, right).clone());
                }
            }
        }
    }

    fn reset(&mut self, clock: Clock, value: B::Constant) {
        assert!(clock > 0);
        let le_pos_value = B::new_le(value.clone());
        let le_neg_value = B::new_le(value.checked_neg().unwrap());
        for other in 0..self.dimension {
            self.layout.set(
                clock,
                other,
                self.layout.get(0, other).add(&le_pos_value).unwrap(),
            );
            self.layout.set(
                other,
                clock,
                self.layout.get(other, 0).add(&le_neg_value).unwrap(),
            );
        }
    }

    fn is_unbounded(&self, clock: Clock) -> bool {
        self.layout.get(clock, 0).is_unbounded()
    }

    fn get_upper_bound(&self, clock: Clock) -> Option<B::Constant> {
        self.layout.get(clock, 0).constant()
    }

    fn get_lower_bound(&self, clock: Clock) -> Option<B::Constant> {
        Some(self.layout.get(0, clock).constant()?.checked_neg().unwrap())
    }

    fn is_satisfied(&self, constraint: Constraint<B>) -> bool {
        !constraint
            .bound
            .is_tighter_than(self.layout.get(constraint.left, constraint.right))
    }

    fn includes(&self, other: &Self) -> bool {
        for left in 0..self.dimension {
            for right in 0..self.dimension {
                if self
                    .layout
                    .get(left, right)
                    .is_tighter_than(other.layout.get(left, right))
                {
                    return false;
                }
            }
        }
        true
    }

    fn resize(&self, num_clocks: usize) -> Self {
        let mut other = Self::new_unconstrained(num_clocks);
        for left in 0..min(self.dimension, other.dimension) {
            for right in 0..min(self.dimension, other.dimension) {
                let bound = self.layout.get(left, right);
                other.layout.set(left, right, bound.clone());
            }
        }
        other.canonicalize();
        other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let mut zone: DBM<i64> = DBM::new_zero(3);
        assert_eq!(zone.get_lower_bound(1), Some(0));
        assert_eq!(zone.get_upper_bound(1), Some(0));
        zone.future();
        assert_eq!(zone.get_lower_bound(1), Some(0));
        assert_eq!(zone.get_upper_bound(1), None);
        assert!(zone.is_unbounded(1));
        let mut copy = zone.clone();
        zone.add_constraint(Constraint::new_lt(1, 4));
        assert!(!zone.is_unbounded(1));
        assert_eq!(zone.get_lower_bound(1), Some(0));
        assert_eq!(zone.get_upper_bound(1), Some(4));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_le(1, 4));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_lt(1, 3));
        assert!(!copy.includes(&zone));
        assert!(zone.includes(&copy));
        zone.intersect(&copy);
        assert!(zone.includes(&copy));
        assert!(copy.includes(&zone));
        copy.add_constraint(Constraint::new_diff_lt(1, 2, 4));
        copy.add_constraint(Constraint::new_diff_gt(1, 2, 5));
        assert!(copy.is_empty());
    }
}
