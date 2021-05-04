//! Different [storage layouts][Layout] for DBMs.

use super::*;

/// Represents a storage layout for [Dbm].
pub trait Layout<B: Bound> {
    /// Initializes the storage for `num_variables` clock variables using
    /// `init` as initial bound for all differences.
    fn new(num_variables: usize, init: B) -> Self;

    /// Sets the bound for the clock difference `left - right`.
    fn set(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B);

    /// Sets the bound for the clock difference `left - right`.
    ///
    /// # Safety
    /// Assumes that the indices of each clock are strictly less than `num_variables`.
    /// Causes undefined behavior otherwise.
    unsafe fn set_unchecked(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B);

    /// Retrieves the bound for the clock difference `left - right`.
    fn get(&self, left: impl AnyClock, right: impl AnyClock) -> &B;

    /// Retrieves the bound for the clock difference `left - right`.
    ///
    /// # Safety
    /// Assumes that the indices of each clock are strictly less than `num_variables`.
    /// Causes undefined behavior otherwise.
    unsafe fn get_unchecked(&self, left: impl AnyClock, right: impl AnyClock) -> &B;
}

/// A [storage layout](Layout) using a one-dimensional array.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct LinearLayout<B: Bound> {
    dimension: usize,
    bounds: Box<[B]>,
}

impl<B: Bound> LinearLayout<B> {
    /// Computes the index where the bound for `left - right` is stored.
    #[inline(always)]
    fn index_of(&self, left: impl AnyClock, right: impl AnyClock) -> usize {
        left.into_index() * self.dimension + right.into_index()
    }
}

impl<B: Bound> Layout<B> for LinearLayout<B> {
    #[inline(always)]
    fn new(num_variables: usize, default: B) -> Self {
        let dimension = num_variables + 1;
        LinearLayout {
            dimension,
            bounds: vec![default; dimension * dimension].into(),
        }
    }

    #[inline(always)]
    fn set(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B) {
        let index = self.index_of(left, right);
        self.bounds[index] = bound;
    }

    #[inline(always)]
    unsafe fn set_unchecked(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B) {
        let index = self.index_of(left, right);
        *self.bounds.get_unchecked_mut(index) = bound;
    }

    #[inline(always)]
    fn get(&self, left: impl AnyClock, right: impl AnyClock) -> &B {
        &self.bounds[self.index_of(left, right)]
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, left: impl AnyClock, right: impl AnyClock) -> &B {
        &self.bounds.get_unchecked(self.index_of(left, right))
    }
}

/// A [storage layout](Layout) using a two-dimensional array.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
#[repr(transparent)]
pub struct MatrixLayout<B: Bound> {
    matrix: Box<[Box<[B]>]>,
}

impl<B: Bound> Layout<B> for MatrixLayout<B> {
    #[inline(always)]
    fn new(num_variables: usize, default: B) -> Self {
        let dimension = num_variables + 1;
        MatrixLayout {
            matrix: vec![vec![default; dimension].into(); dimension].into(),
        }
    }

    #[inline(always)]
    fn set(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B) {
        self.matrix[left.into_index()][right.into_index()] = bound;
    }

    #[inline(always)]
    unsafe fn set_unchecked(&mut self, left: impl AnyClock, right: impl AnyClock, bound: B) {
        *self
            .matrix
            .get_unchecked_mut(left.into_index())
            .get_unchecked_mut(right.into_index()) = bound;
    }

    #[inline(always)]
    fn get(&self, left: impl AnyClock, right: impl AnyClock) -> &B {
        &self.matrix[left.into_index()][right.into_index()]
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, left: impl AnyClock, right: impl AnyClock) -> &B {
        &self
            .matrix
            .get_unchecked(left.into_index())
            .get_unchecked(right.into_index())
    }
}
