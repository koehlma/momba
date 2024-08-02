use std::hash::Hash;

/// Base trait for transition systems.
pub trait BaseTs {
    /// Uniquely identifies a state of the transition system.
    type StateId: Copy + Eq + Ord + Hash;

    /// Uniquely identifies a transition of the transition system.
    type TransitionId: Copy + Eq + Ord + Hash;
}

impl<'ts, TS: BaseTs> BaseTs for &'ts TS {
    type StateId = TS::StateId;

    type TransitionId = TS::TransitionId;
}

/// A copyable reference to a TS.
pub trait TsRef: BaseTs + Copy {}

impl<TS: BaseTs + Copy> TsRef for TS {}

/// A TS supporting iteration oval all states.
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

/// A TS supporting iteration oval all transitions.
pub trait Transitions: BaseTs {
    /// Iterator type over all transitions of the TS.
    type TransitionsIter<'iter>: 'iter + Iterator<Item = Self::TransitionId>
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
    type TransitionsIter<'iter> = TS::TransitionsIter<'iter> where Self: 'iter;

    #[inline(always)]
    fn transitions(&self) -> Self::TransitionsIter<'_> {
        (*self).transitions()
    }

    #[inline(always)]
    fn num_transitions(&self) -> usize {
        (*self).num_transitions()
    }
}
