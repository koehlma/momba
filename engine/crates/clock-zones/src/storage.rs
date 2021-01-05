use super::*;

/// Represents a storage layout for [DBM].
pub trait Layout<B: Bound> {
    /// Initializes the storage for `num_clock` clocks using `init` as
    /// initial bound for all differences.
    fn new(num_clocks: usize, init: B) -> Self;

    /// Sets the bound for the clock difference `left - right`.
    fn set(&mut self, left: Clock, right: Clock, bound: B);

    /// Retrieves the bound for the clock difference `left - right`.
    fn get(&self, left: Clock, right: Clock) -> &B;
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
    fn index_of(&self, left: Clock, right: Clock) -> usize {
        left * self.dimension + right
    }
}

impl<B: Bound> Layout<B> for LinearLayout<B> {
    #[inline(always)]
    fn new(num_clocks: usize, default: B) -> Self {
        let dimension = num_clocks + 1;
        LinearLayout {
            dimension,
            bounds: vec![default; dimension * dimension].into(),
        }
    }

    #[inline(always)]
    fn set(&mut self, left: Clock, right: Clock, bound: B) {
        let index = self.index_of(left, right);
        self.bounds[index] = bound;
    }
    #[inline(always)]
    fn get(&self, left: Clock, right: Clock) -> &B {
        &self.bounds[self.index_of(left, right)]
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
    fn new(num_clocks: usize, default: B) -> Self {
        let dimension = num_clocks + 1;
        MatrixLayout {
            matrix: vec![vec![default; dimension].into(); dimension].into(),
        }
    }

    #[inline(always)]
    fn set(&mut self, left: Clock, right: Clock, bound: B) {
        self.matrix[left][right] = bound;
    }

    #[inline(always)]
    fn get(&self, left: Clock, right: Clock) -> &B {
        &self.matrix[left][right]
    }
}
