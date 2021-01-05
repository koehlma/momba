use std::iter::IntoIterator;

use crate::bounds::*;
use crate::constants::*;

type Clock = usize;

pub struct Constraint<B: Bound> {
    pub left: Clock,
    pub right: Clock,
    pub bound: B,
}

impl<B: Bound> Constraint<B> {
    pub fn new_le(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: clock,
            right: 0,
            bound: B::new_le(constant),
        }
    }

    pub fn new_lt(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: clock,
            right: 0,
            bound: B::new_lt(constant),
        }
    }

    pub fn new_ge(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: 0,
            right: clock,
            bound: B::new_le(constant.checked_neg().unwrap()),
        }
    }

    pub fn new_gt(clock: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: 0,
            right: clock,
            bound: B::new_lt(constant.checked_neg().unwrap()),
        }
    }

    pub fn new_diff_le(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left,
            right,
            bound: B::new_le(constant),
        }
    }

    pub fn new_diff_lt(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left,
            right,
            bound: B::new_lt(constant),
        }
    }

    pub fn new_diff_ge(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: right,
            right: left,
            bound: B::new_le(constant.checked_neg().unwrap()),
        }
    }

    pub fn new_diff_gt(left: Clock, right: Clock, constant: B::Constant) -> Self {
        Constraint {
            left: right,
            right: left,
            bound: B::new_lt(constant.checked_neg().unwrap()),
        }
    }
}

pub trait Zone<B: Bound> {
    fn new_unconstrained(num_clocks: usize) -> Self;
    fn new_zero(num_clocks: usize) -> Self;

    fn with_constraints<U>(num_clocks: usize, constraints: U) -> Self
    where
        U: IntoIterator<Item = Constraint<B>>;

    fn num_clocks(self) -> usize;

    fn get_bound(&self, left: Clock, right: Clock) -> &B;

    fn is_empty(&self) -> bool;

    fn add_constraint(&mut self, constraint: Constraint<B>);
    fn add_constraints<U>(&mut self, constraints: U)
    where
        U: IntoIterator<Item = Constraint<B>>;

    fn intersect(&mut self, other: &Self);

    fn future(&mut self);
    fn past(&mut self);

    fn reset(&mut self, clock: Clock, value: B::Constant);

    fn is_unbounded(&self, clock: Clock) -> bool;

    fn get_upper_bound(&self, clock: Clock) -> Option<B::Constant>;
    fn get_lower_bound(&self, clock: Clock) -> Option<B::Constant>;

    fn is_satisfied(&self, constraint: Constraint<B>) -> bool;

    fn includes(&self, other: &Self) -> bool;
}

pub trait DBMLayout<B: Bound> {
    fn new(num_clocks: usize, default: B) -> Self;

    fn set(&mut self, left: Clock, right: Clock, bound: B);
    fn get(&self, left: Clock, right: Clock) -> &B;
}

impl<B: Bound> DBMLayout<B> for Vec<Vec<B>> {
    #[inline(always)]
    fn new(num_clocks: usize, default: B) -> Self {
        let dimension = num_clocks + 1;
        let mut matrix = Vec::with_capacity(dimension);
        for row in 0..dimension {
            matrix.push(Vec::with_capacity(dimension));
            matrix[row].resize(dimension, default.clone());
        }
        matrix
    }

    #[inline(always)]
    fn set(&mut self, left: Clock, right: Clock, bound: B) {
        self[left][right] = bound;
    }
    #[inline(always)]
    fn get(&self, left: Clock, right: Clock) -> &B {
        &self[left][right]
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DBM<B: Bound, L: DBMLayout<B> = Vec<Vec<B>>> {
    dimension: usize,
    layout: L,
    le_zero: B,
    unbounded: B,
}

impl<B: Bound, L: DBMLayout<B>> DBM<B, L> {
    fn new(num_clocks: usize, default: B) -> Self {
        let dimension = num_clocks + 1;
        let mut layout = L::new(num_clocks, default);
        let le_zero = B::le_zero();
        layout.set(0, 0, le_zero.clone());
        for clock in 1..dimension {
            layout.set(0, clock, le_zero.clone());
            layout.set(clock, clock, le_zero.clone());
        }
        DBM {
            dimension,
            layout,
            le_zero,
            unbounded: B::unbounded(),
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

impl<B: Bound, L: DBMLayout<B>> Zone<B> for DBM<B, L> {
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
    fn num_clocks(self) -> usize {
        self.dimension - 1
    }

    #[inline(always)]
    fn get_bound(&self, left: usize, right: usize) -> &B {
        self.layout.get(left, right)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.layout.get(0, 0).is_tighter_than(&self.le_zero)
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
            self.layout.set(left, 0, self.unbounded.clone());
        }
    }
    fn past(&mut self) {
        for right in 1..self.dimension {
            self.layout.set(0, right, self.le_zero.clone());
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
