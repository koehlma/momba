//! Provides a type [`Transposed`] inverting all transitions of a TS.

use super::traits::*;

/// A transposed TS.
#[derive(Debug, Clone, Copy)]
pub struct Transposed<TS>(TS);

impl<TS> Transposed<TS> {
    /// Creates a new transposed TS.
    pub fn new(ts: TS) -> Self {
        Self(ts)
    }

    /// Returns the original inner TS.
    pub fn into_inner(self) -> TS {
        self.0
    }
}

impl<TS: BaseTs> BaseTs for Transposed<TS> {
    type StateId = TS::StateId;

    type State = TS::State;

    fn get_label(&self, id: &Self::StateId) -> &Self::State {
        self.0.get_label(id)
    }
}

impl<TS: States> States for Transposed<TS> {
    type StatesIter<'iter> = TS::StatesIter<'iter>
        where
            Self: 'iter;

    fn states(&self) -> Self::StatesIter<'_> {
        self.0.states()
    }

    fn num_states(&self) -> usize {
        self.0.num_states()
    }
}

impl<TS: Predecessors> Successors for Transposed<TS> {
    type SuccessorsIter<'iter> = TS::PredecessorsIter<'iter>
        where
            Self: 'iter;

    fn successors(&self, state: &Self::StateId) -> Self::SuccessorsIter<'_> {
        self.0.predecessors(state)
    }
}

impl<TS: Successors> Predecessors for Transposed<TS> {
    type PredecessorsIter<'iter> = TS::SuccessorsIter<'iter>
        where
            Self: 'iter;

    fn predecessors(&self, state: &Self::StateId) -> Self::PredecessorsIter<'_> {
        self.0.successors(state)
    }
}

impl<TS: MakeDenseStateSet> MakeDenseStateSet for Transposed<TS> {
    type DenseStateSet = TS::DenseStateSet;

    fn make_dense_state_set(&self) -> Self::DenseStateSet {
        self.0.make_dense_state_set()
    }
}

impl<TS: MakeSparseStateSet> MakeSparseStateSet for Transposed<TS> {
    type SparseStateSet = TS::SparseStateSet;

    fn make_sparse_state_set(&self) -> Self::SparseStateSet {
        self.0.make_sparse_state_set()
    }
}

impl<TS: MakeDenseStateMap<T>, T> MakeDenseStateMap<T> for Transposed<TS> {
    type DenseStateMap = TS::DenseStateMap;

    fn make_dense_state_map(&self) -> Self::DenseStateMap {
        self.0.make_dense_state_map()
    }
}

impl<TS: MakeSparseStateMap<T>, T> MakeSparseStateMap<T> for Transposed<TS> {
    type SparseStateMap = TS::SparseStateMap;

    fn make_sparse_state_map(&self) -> Self::SparseStateMap {
        self.0.make_sparse_state_map()
    }
}
