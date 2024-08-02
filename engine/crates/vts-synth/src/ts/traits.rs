//! Abstractions for explicit-state transition systems.

use std::{fmt::Debug, hash::Hash};

/// Base trait for transition systems.
pub trait BaseTs {
    /// Type for unique identifiers of the states of the TS.
    type StateId: Debug + Clone + Eq + Hash;

    /// State type of the TS.
    type State;

    /// Retrieves the state with the given id.
    fn get_label(&self, id: &Self::StateId) -> &Self::State;
}

impl<'ts, TS: BaseTs> BaseTs for &'ts TS {
    type StateId = TS::StateId;

    type State = TS::State;

    #[inline(always)]
    fn get_label(&self, id: &Self::StateId) -> &Self::State {
        (*self).get_label(id)
    }
}

/// A copyable reference to a TS.
pub trait TsRef: BaseTs + Copy {}

impl<TS: BaseTs + Copy> TsRef for TS {}

/// A TS supporting iteration over all states.
pub trait States: BaseTs {
    /// Iterator type over all states of the TS.
    type StatesIter<'iter>: 'iter + Iterator<Item = Self::StateId>
    where
        Self: 'iter;

    /// Iterator over all states of the TS.
    fn states(&self) -> Self::StatesIter<'_>;

    /// The number of states of the TS.
    fn num_states(&self) -> usize {
        self.states().count()
    }
}

impl<'ts, TS: States> States for &'ts TS {
    type StatesIter<'iter> = TS::StatesIter<'iter> where Self: 'iter;

    #[inline(always)]
    fn states(&self) -> Self::StatesIter<'_> {
        (*self).states()
    }

    #[inline(always)]
    fn num_states(&self) -> usize {
        (*self).num_states()
    }
}

/// A TS with _initial states_.
pub trait InitialStates: BaseTs {
    /// Iterator type over the initial states of the TS.
    type InitialStatesIter<'iter>: 'iter + Iterator<Item = Self::StateId>
    where
        Self: 'iter;

    /// Iterator over the initial states of the TS.
    fn initial_states(&self) -> Self::InitialStatesIter<'_>;

    /// Checks whether the state is an initial state.
    fn is_initial(&self, state: &Self::StateId) -> bool;

    /// The number of initial states of the TS.
    fn num_initial_states(&self) -> usize {
        self.initial_states().count()
    }
}

impl<'ts, TS: InitialStates> InitialStates for &'ts TS {
    type InitialStatesIter<'iter> = TS::InitialStatesIter<'iter>
        where
            Self: 'iter;

    #[inline(always)]
    fn initial_states(&self) -> Self::InitialStatesIter<'_> {
        (*self).initial_states()
    }

    #[inline(always)]
    fn is_initial(&self, state: &Self::StateId) -> bool {
        (*self).is_initial(state)
    }

    #[inline(always)]
    fn num_initial_states(&self) -> usize {
        (*self).num_initial_states()
    }
}

/// A TS supporting iteration over all transitions.
pub trait Transitions: BaseTs {
    type Transition<'trans>
    where
        Self: 'trans;

    /// Iterator type over all transitions of the TS.
    type TransitionsIter<'iter>: 'iter + Iterator<Item = Self::Transition<'iter>>
    where
        Self: 'iter;

    /// Iterator over all transitions of the TS.
    fn transitions(&self) -> Self::TransitionsIter<'_>;

    /// The number of transitions of the TS.
    fn num_transitions(&self) -> usize {
        self.transitions().count()
    }
}

impl<'ts, TS: Transitions> Transitions for &'ts TS {
    type Transition<'trans> = TS::Transition<'trans>
    where
        Self: 'trans;

    type TransitionsIter<'iter> = TS::TransitionsIter<'iter>
    where
        Self: 'iter;

    #[inline(always)]
    fn transitions(&self) -> Self::TransitionsIter<'_> {
        (*self).transitions()
    }

    #[inline(always)]
    fn num_transitions(&self) -> usize {
        (*self).num_transitions()
    }
}

/// A TS supporting iteration over the _successor states_ of a state.
pub trait Successors: BaseTs {
    /// Iterator type over the successor states of a state.
    type SuccessorsIter<'iter>: 'iter + Iterator<Item = Self::StateId>
    where
        Self: 'iter;

    /// Iterator over the successor states of a state.
    fn successors(&self, state: &Self::StateId) -> Self::SuccessorsIter<'_>;
}

impl<'ts, TS: Successors> Successors for &'ts TS {
    type SuccessorsIter<'iter> = TS::SuccessorsIter<'iter>
        where
            Self: 'iter;

    #[inline(always)]
    fn successors(&self, state: &Self::StateId) -> Self::SuccessorsIter<'_> {
        (*self).successors(state)
    }
}

/// A TS supporting iteration over the _predecessor states_ of a state.
pub trait Predecessors: BaseTs {
    /// Iterator type over the predecessor states of a state.
    type PredecessorsIter<'iter>: 'iter + Iterator<Item = Self::StateId>
    where
        Self: 'iter;

    /// Iterator over the predecessor states of a state.
    fn predecessors(&self, state: &Self::StateId) -> Self::PredecessorsIter<'_>;
}

impl<'ts, TS: Predecessors> Predecessors for &'ts TS {
    type PredecessorsIter<'iter> = TS::PredecessorsIter<'iter>
        where
            Self: 'iter;

    #[inline(always)]
    fn predecessors(&self, state: &Self::StateId) -> Self::PredecessorsIter<'_> {
        (*self).predecessors(state)
    }
}

/// A set of states of a TS.
pub trait StateSet<S> {
    /// Inserts a state into the set.
    ///
    /// Returns `true` if the state has been inserted, i.e., was not already in the
    /// set.
    fn insert(&mut self, state: S) -> bool;

    /// Removes a state from the set.
    ///
    /// Returns `true` if the state has been removed, i.e., was contained in the set.
    fn remove(&mut self, state: &S) -> bool;

    /// Checks whether a state is contained in the set.
    fn contains(&self, state: &S) -> bool;

    /// Clears the set.
    fn clear(&mut self);

    type Iter<'iter>: 'iter + Iterator<Item = S>
    where
        Self: 'iter;

    fn iter(&self) -> Self::Iter<'_>;
}

/// A TS supporting _dense state sets_.
pub trait MakeDenseStateSet: BaseTs {
    /// Type of dense state sets.
    type DenseStateSet: StateSet<Self::StateId>;

    /// Creates an empty dense state set.
    fn make_dense_state_set(&self) -> Self::DenseStateSet;
}

impl<'ts, TS: MakeDenseStateSet> MakeDenseStateSet for &'ts TS {
    type DenseStateSet = TS::DenseStateSet;

    fn make_dense_state_set(&self) -> Self::DenseStateSet {
        (*self).make_dense_state_set()
    }
}

/// A TS supporting _sparse state sets_.
pub trait MakeSparseStateSet: BaseTs {
    /// Type of sparse state sets.
    type SparseStateSet: StateSet<Self::StateId>;

    /// Crates an empty sparse state set.
    fn make_sparse_state_set(&self) -> Self::SparseStateSet;
}

impl<'ts, TS: MakeSparseStateSet> MakeSparseStateSet for &'ts TS {
    type SparseStateSet = TS::SparseStateSet;

    fn make_sparse_state_set(&self) -> Self::SparseStateSet {
        (*self).make_sparse_state_set()
    }
}

/// A map from states of a TS to other values.
pub trait StateMap<S, V> {
    /// Inserts a value for a state.
    fn insert(&mut self, state: S, value: V);

    fn insert_default(&mut self, state: S)
    where
        V: Default,
    {
        self.insert(state, V::default())
    }

    /// Removes the value for a state.
    fn remove(&mut self, state: &S) -> Option<V>;

    /// Returns the value for a state.
    fn get(&self, state: &S) -> Option<&V>;

    /// Returns the value for a state.
    fn get_mut(&mut self, state: &S) -> Option<&mut V>;

    fn get_mut_or_default(&mut self, state: &S) -> &mut V
    where
        S: Clone,
        V: Default,
    {
        if !self.contains(state) {
            self.insert(state.clone(), V::default());
        }
        self.get_mut(state).unwrap()
    }

    /// Checks whether the map contains a value for a state.
    fn contains(&self, state: &S) -> bool;
}

/// A TS supporting _dense state maps_.
pub trait MakeDenseStateMap<V>: BaseTs {
    /// Type of dense state maps.
    type DenseStateMap: StateMap<Self::StateId, V>;

    /// Creates an empty dense state map.
    fn make_dense_state_map(&self) -> Self::DenseStateMap;
}

impl<'ts, T, TS: MakeDenseStateMap<T>> MakeDenseStateMap<T> for &'ts TS {
    type DenseStateMap = TS::DenseStateMap;

    fn make_dense_state_map(&self) -> Self::DenseStateMap {
        (*self).make_dense_state_map()
    }
}

/// A TS supporting _sparse state maps_.
pub trait MakeSparseStateMap<V>: BaseTs {
    /// Type of sparse state maps.
    type SparseStateMap: StateMap<Self::StateId, V>;

    /// Creates an empty sparse state map.
    fn make_sparse_state_map(&self) -> Self::SparseStateMap;
}

impl<'ts, T, TS: MakeSparseStateMap<T>> MakeSparseStateMap<T> for &'ts TS {
    type SparseStateMap = TS::SparseStateMap;

    fn make_sparse_state_map(&self) -> Self::SparseStateMap {
        (*self).make_sparse_state_map()
    }
}
