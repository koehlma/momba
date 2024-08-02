//! Efficient representation of explicit-state transition systems.

use std::{
    fmt::Debug,
    hash::{BuildHasher, Hash, Hasher},
    ops::{Index, Range},
    sync::atomic::{self},
};

use bit_set::BitSet;
use hashbrown::{hash_map::DefaultHashBuilder, raw::RawTable, HashMap, HashSet};

use self::traits::*;

pub mod algorithms;
pub mod output;
pub mod traits;
pub mod transposed;
pub mod types;

const TS_DATA_ID_SHIFT: u64 = 48;
const TS_DATA_ID_MASK: u64 = 0xFFFF;
const STATE_IDX_MAX: usize = ((1 << TS_DATA_ID_SHIFT) - 1) & (usize::MAX as u64) as usize;
const STATE_IDX_SHIFT: u64 = 0;
const STATE_IDX_MASK: u64 = !(TS_DATA_ID_MASK << TS_DATA_ID_SHIFT);

/// Returns the ID for the next transition system.
fn next_ts_data_id() -> u16 {
    static ID: atomic::AtomicU16 = atomic::AtomicU16::new(0);
    ID.fetch_add(1, atomic::Ordering::Relaxed)
}

/// Unique identifier of a state in a TS.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId(u64);

impl StateId {
    pub const fn from_parts(ts_id: u16, idx: usize) -> Self {
        debug_assert!(idx <= STATE_IDX_MAX as usize);
        Self(((ts_id as u64) << TS_DATA_ID_SHIFT) | ((idx as u64) << STATE_IDX_SHIFT))
    }

    /// The index of the state in the state vector.
    pub fn idx(self) -> usize {
        ((self.0 >> STATE_IDX_SHIFT) & STATE_IDX_MASK) as usize
    }

    /// The id of the transition system the state belongs to.
    fn ts_id(self) -> u16 {
        ((self.0 >> TS_DATA_ID_SHIFT) & TS_DATA_ID_MASK) as u16
    }
}

impl Debug for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateId")
            .field("ts_id", &self.ts_id())
            .field("idx", &self.idx())
            .finish()
    }
}

/// Unique identifier of a transition in a TS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionId(pub(crate) usize);

/// State with additional data stored in a TS.
#[derive(Debug, Clone)]
struct StateEntry<S> {
    /// The actual data of the state.
    state: S,
    /// Range into the outgoing transitions vector.
    outgoing: Range<usize>,
    /// Range into the reverse edges vector.
    reverse: Range<usize>,
}

impl<S> StateEntry<S> {
    /// Creates a new state entry.
    fn new(state: S) -> Self {
        Self {
            state,
            outgoing: 0..0,
            reverse: 0..0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateLabeling<L>(Vec<L>);

impl<L> std::ops::Index<StateId> for StateLabeling<L> {
    type Output = L;

    fn index(&self, index: StateId) -> &Self::Output {
        &self.0[index.idx()]
    }
}

impl<L> std::ops::Index<&StateId> for StateLabeling<L> {
    type Output = L;

    fn index(&self, index: &StateId) -> &Self::Output {
        &self.0[index.idx()]
    }
}

impl<L> std::ops::IndexMut<StateId> for StateLabeling<L> {
    fn index_mut(&mut self, index: StateId) -> &mut Self::Output {
        &mut self.0[index.idx()]
    }
}

impl<L> std::ops::IndexMut<&StateId> for StateLabeling<L> {
    fn index_mut(&mut self, index: &StateId) -> &mut Self::Output {
        &mut self.0[index.idx()]
    }
}

/// A transition of a TS.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transition<L> {
    /// The source state of the transition.
    source: StateId,
    /// The label of the transition.
    label: L,
    /// The target state of the transition.
    target: StateId,
}

impl<L> Transition<L> {
    /// The source state of the transition.
    pub fn source(&self) -> StateId {
        self.source
    }

    /// The label of the transition.
    pub fn action(&self) -> &L {
        &self.label
    }

    /// The target state of the transition.
    pub fn target(&self) -> StateId {
        self.target
    }
}

/// The actual data of a TS.
#[derive(Debug, Clone)]
struct TsData<S, L> {
    /// The id of the TS.
    ts_id: u16,
    /// The states of the TS.
    states: Vec<StateEntry<S>>,
    /// The initial states of the TS.
    initial_states: HashSet<StateId>,
    /// The transitions.
    transitions: Vec<Transition<L>>,
    /// The outgoing transition of the TS.
    outgoing: Vec<TransitionId>,
    /// The reverse edges of the TS.
    reverse: Vec<TransitionId>,
}

impl<S, L> TsData<S, L> {
    fn new() -> Self {
        Self {
            ts_id: next_ts_data_id(),
            states: Vec::new(),
            initial_states: HashSet::new(),
            transitions: Vec::new(),
            outgoing: Vec::new(),
            reverse: Vec::new(),
        }
    }

    /// Turns `self` into [`Ts`] by establishing the necessary invariants.
    ///
    /// **Complexity:** `O(n log n)` where `n` is the number of edges.
    fn into_ts(mut self) -> Ts<S, L> {
        // 1️⃣ Sort the incoming and outgoing edges.
        self.outgoing.sort_by(|x, y| {
            self.transitions[x.0]
                .source
                .idx()
                .cmp(&self.transitions[y.0].source.idx())
        });
        self.reverse.sort_by(|x, y| {
            self.transitions[x.0]
                .target
                .idx()
                .cmp(&self.transitions[y.0].target.idx())
        });

        // println!("{:?}", self.outgoing);
        // println!("{:?}", self.reverse);

        // 2️⃣ Update/initializes the ranges stored with the states.
        let mut last_outgoing = 0;
        let mut last_reverse = 0;
        for (id, state) in self.states.iter_mut().enumerate() {
            // Update outgoing range.
            let start_outgoing = last_outgoing;
            while last_outgoing < self.outgoing.len()
                && self.transitions[self.outgoing[last_outgoing].0]
                    .source
                    .idx()
                    == id
            {
                last_outgoing += 1;
            }
            state.outgoing = start_outgoing..last_outgoing;
            // Update incoming range.
            let start_reverse = last_reverse;
            while last_reverse < self.reverse.len()
                && self.transitions[self.reverse[last_reverse].0].target.idx() == id
            {
                last_reverse += 1;
            }
            state.reverse = start_reverse..last_reverse;
            // Sanity check the updated ranges.
            debug_assert!(self.outgoing[state.outgoing.clone()].iter().all(|edge| self
                .transitions[edge.0]
                .source()
                .idx()
                == id));
            debug_assert!(self.reverse[state.reverse.clone()]
                .iter()
                .all(|edge| self.transitions[edge.0].target().idx() == id));
        }

        // Check that all edges have been considered.
        assert_eq!(last_outgoing, self.outgoing.len());
        assert_eq!(last_reverse, self.reverse.len());

        let ts = Ts {
            data: self,
            empty_state_set: im::HashSet::new(),
        };

        ts.assert_invariants();

        ts
    }
}

impl<S, L> Index<StateId> for TsData<S, L> {
    type Output = StateEntry<S>;

    fn index(&self, index: StateId) -> &Self::Output {
        debug_assert!(
            index.ts_id() == self.ts_id,
            "State does not belong to this TS (state TS = {}, actual TS = {}).",
            index.ts_id(),
            self.ts_id,
        );
        &self.states[index.idx()]
    }
}

/// An explicit-state TS.
#[derive(Debug, Clone)]
pub struct Ts<S, L> {
    /// The data of the TS.
    data: TsData<S, L>,
    /// Empty sparse state set.
    ///
    /// This is needed because we want all sparse state sets with the same states to have
    /// the same hash. This means that the same hasher has to be used, i.e., they all have
    /// to be constructed from the same empty [`im::HashSet`].
    empty_state_set: im::HashSet<StateId>,
}

impl<S, L> Ts<S, L> {
    /// The outgoing transitions of a state.
    pub fn outgoing(&self, state: &StateId) -> impl '_ + Iterator<Item = &Transition<L>> {
        self.data.outgoing[self.data[*state].outgoing.clone()]
            .iter()
            .map(|transition| &self.data.transitions[transition.0])
    }

    /// The incoming transitions of a satte.
    pub fn incoming(&self, state: &StateId) -> impl '_ + Iterator<Item = &Transition<L>> {
        self.data.reverse[self.data[*state].reverse.clone()]
            .iter()
            .map(|transition| &self.data.transitions[transition.0])
    }

    /// The incoming transitions of a satte.
    pub fn incoming_ids(&self, state: &StateId) -> impl '_ + Iterator<Item = TransitionId> {
        self.data.reverse[self.data[*state].reverse.clone()]
            .iter()
            .map(|transition| *transition)
    }

    pub fn get_transition(&self, id: TransitionId) -> &Transition<L> {
        &self.data.transitions[id.0]
    }

    /// Applies a function to the labels of the TS.
    pub fn map_labels<F, U>(&self, mut fun: F) -> Ts<S, U>
    where
        F: FnMut(&L) -> U,
        S: Clone,
    {
        Ts {
            data: TsData {
                ts_id: self.data.ts_id,
                states: self.data.states.clone(),
                initial_states: self.data.initial_states.clone(),
                transitions: self
                    .data
                    .transitions
                    .iter()
                    .map(|transition| Transition {
                        source: transition.source,
                        label: fun(&transition.label),
                        target: transition.target,
                    })
                    .collect(),
                outgoing: self.data.outgoing.clone(),
                reverse: self.data.reverse.clone(),
            },
            empty_state_set: im::HashSet::new(),
        }
    }

    /// Applies a function to the states of the TS.
    pub fn map_states<F, U>(&self, mut fun: F) -> Ts<U, L>
    where
        F: FnMut(StateId, &S) -> U,
        L: Clone,
    {
        Ts {
            data: TsData {
                ts_id: self.data.ts_id,
                states: self
                    .data
                    .states
                    .iter()
                    .enumerate()
                    .map(|(idx, entry)| StateEntry {
                        state: fun(StateId::from_parts(self.data.ts_id, idx), &entry.state),
                        outgoing: entry.outgoing.clone(),
                        reverse: entry.reverse.clone(),
                    })
                    .collect(),
                initial_states: self.data.initial_states.clone(),
                transitions: self.data.transitions.clone(),
                outgoing: self.data.outgoing.clone(),
                reverse: self.data.reverse.clone(),
            },
            empty_state_set: im::HashSet::new(),
        }
    }

    pub fn create_state_labeling<L2>(
        &self,
        mut default: impl FnMut(StateId) -> L2,
    ) -> StateLabeling<L2> {
        let mut labels = Vec::with_capacity(self.num_states());
        for id in self.states() {
            labels.push(default(id));
        }
        StateLabeling(labels)
    }

    pub fn create_default_state_labeling<L2>(&self) -> StateLabeling<L2>
    where
        L2: Default,
    {
        self.create_state_labeling(|_| L2::default())
    }

    pub fn assert_invariants(&self) {
        for state in self.states() {
            for transition in self.outgoing(&state) {
                assert_eq!(transition.source(), state);
            }
            for transition in self.incoming(&state) {
                assert_eq!(transition.target(), state);
            }
        }
    }
}

impl<S, L> BaseTs for Ts<S, L> {
    type StateId = StateId;

    type State = S;

    fn get_label(&self, id: &Self::StateId) -> &Self::State {
        &self.data[*id].state
    }
}

impl<S, L> States for Ts<S, L> {
    type StatesIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn states(&self) -> Self::StatesIter<'_> {
        (0..self.num_states()).map(|idx| StateId::from_parts(self.data.ts_id, idx))
    }

    fn num_states(&self) -> usize {
        self.data.states.len()
    }
}

impl<S, L> InitialStates for Ts<S, L> {
    type InitialStatesIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn initial_states(&self) -> Self::InitialStatesIter<'_> {
        self.data.initial_states.iter().cloned()
    }

    fn is_initial(&self, state: &Self::StateId) -> bool {
        self.data.initial_states.contains(state)
    }

    fn num_initial_states(&self) -> usize {
        self.data.initial_states.len()
    }
}

impl<S, L> Transitions for Ts<S, L> {
    type Transition<'trans> = &'trans Transition<L>
    where
        Self: 'trans;

    type TransitionsIter<'iter> = impl 'iter + Iterator<Item = Self::Transition<'iter>>
    where
        Self: 'iter;

    fn transitions(&self) -> Self::TransitionsIter<'_> {
        self.data.transitions.iter()
    }

    fn num_transitions(&self) -> usize {
        self.data.transitions.len()
    }
}

impl<S, L> Successors for Ts<S, L> {
    type SuccessorsIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn successors(&self, state: &Self::StateId) -> Self::SuccessorsIter<'_> {
        self.data.outgoing[self.data[*state].outgoing.clone()]
            .iter()
            .map(|transition| self.data.transitions[transition.0].target())
    }
}

impl<S, L> Predecessors for Ts<S, L> {
    type PredecessorsIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn predecessors(&self, state: &Self::StateId) -> Self::PredecessorsIter<'_> {
        self.data.reverse[self.data[*state].reverse.clone()]
            .iter()
            .map(|transition| self.data.transitions[transition.0].source())
    }
}

impl<S, L> MakeDenseStateSet for Ts<S, L> {
    type DenseStateSet = DenseStateSet;

    fn make_dense_state_set(&self) -> Self::DenseStateSet {
        DenseStateSet {
            ts_id: self.data.ts_id,
            set: BitSet::with_capacity(self.data.states.len()),
        }
    }
}

impl<S, L> MakeSparseStateSet for Ts<S, L> {
    type SparseStateSet = SparseStateSet;

    fn make_sparse_state_set(&self) -> Self::SparseStateSet {
        SparseStateSet(self.empty_state_set.clone())
    }
}

impl<S, L, V> MakeDenseStateMap<V> for Ts<S, L> {
    type DenseStateMap = DenseStateMap<V>;

    fn make_dense_state_map(&self) -> Self::DenseStateMap {
        let mut entries = Vec::with_capacity(self.data.states.len());
        for _ in 0..self.data.states.len() {
            entries.push(DenseMapEntry::Vacant);
        }
        DenseStateMap {
            ts_id: self.data.ts_id,
            entries,
        }
    }
}

impl<S, L, V> MakeSparseStateMap<V> for Ts<S, L> {
    type SparseStateMap = SparseStateMap<V>;

    fn make_sparse_state_map(&self) -> Self::SparseStateMap {
        SparseStateMap(HashMap::new())
    }
}

/// A dense set of states of a TS.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DenseStateSet {
    ts_id: u16,
    set: BitSet,
}

impl StateSet<StateId> for DenseStateSet {
    #[inline(always)]
    fn insert(&mut self, state: StateId) -> bool {
        debug_assert!(self.ts_id == state.ts_id());
        self.set.insert(state.idx())
    }

    #[inline(always)]
    fn remove(&mut self, state: &StateId) -> bool {
        debug_assert!(self.ts_id == state.ts_id());
        self.set.remove(state.idx())
    }

    #[inline(always)]
    fn contains(&self, state: &StateId) -> bool {
        debug_assert!(self.ts_id == state.ts_id());
        self.set.contains(state.idx())
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.set.clear()
    }

    type Iter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn iter(&self) -> Self::Iter<'_> {
        self.set
            .iter()
            .map(|idx| StateId::from_parts(self.ts_id, idx))
    }
}

/// A sparse set of states of a TS.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SparseStateSet(im::HashSet<StateId>);

impl StateSet<StateId> for SparseStateSet {
    #[inline(always)]
    fn insert(&mut self, state: StateId) -> bool {
        self.0.insert(state).is_none()
    }

    fn remove(&mut self, state: &StateId) -> bool {
        self.0.remove(state).is_some()
    }

    fn contains(&self, state: &StateId) -> bool {
        self.0.contains(state)
    }

    fn clear(&mut self) {
        self.0.clear()
    }

    type Iter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter().cloned()
    }
}

/// An entry of a dense state map.
#[derive(Debug, Clone)]
enum DenseMapEntry<V> {
    Occupied(V),
    Vacant,
}

/// A dense map from states to values of type `V`.
#[derive(Debug, Clone)]
pub struct DenseStateMap<V> {
    ts_id: u16,
    entries: Vec<DenseMapEntry<V>>,
}

impl<V> StateMap<StateId, V> for DenseStateMap<V> {
    fn insert(&mut self, state: StateId, value: V) {
        debug_assert!(state.ts_id() == self.ts_id);
        self.entries[state.idx()] = DenseMapEntry::Occupied(value);
    }

    fn remove(&mut self, state: &StateId) -> Option<V> {
        debug_assert!(state.ts_id() == self.ts_id);
        let mut value = DenseMapEntry::Vacant;
        std::mem::swap(&mut value, &mut self.entries[state.idx()]);
        match value {
            DenseMapEntry::Occupied(value) => Some(value),
            DenseMapEntry::Vacant => None,
        }
    }

    fn get(&self, state: &StateId) -> Option<&V> {
        debug_assert!(state.ts_id() == self.ts_id);
        match &self.entries[state.idx()] {
            DenseMapEntry::Occupied(value) => Some(value),
            DenseMapEntry::Vacant => None,
        }
    }

    fn contains(&self, state: &StateId) -> bool {
        debug_assert!(state.ts_id() == self.ts_id);
        matches!(self.entries[state.idx()], DenseMapEntry::Occupied(_))
    }

    fn get_mut(&mut self, state: &StateId) -> Option<&mut V> {
        match &mut self.entries[state.idx()] {
            DenseMapEntry::Occupied(value) => Some(value),
            DenseMapEntry::Vacant => None,
        }
    }
}

/// A sparse map from states to values of type `V`.
#[derive(Debug, Clone)]
pub struct SparseStateMap<V>(HashMap<StateId, V>);

impl<V> StateMap<StateId, V> for SparseStateMap<V> {
    fn insert(&mut self, state: StateId, value: V) {
        self.0.insert(state, value);
    }

    fn remove(&mut self, state: &StateId) -> Option<V> {
        self.0.remove(state)
    }

    fn get(&self, state: &StateId) -> Option<&V> {
        self.0.get(state)
    }

    fn contains(&self, state: &StateId) -> bool {
        self.0.contains_key(state)
    }

    fn get_mut(&mut self, state: &StateId) -> Option<&mut V> {
        self.0.get_mut(state)
    }
}

/// Computes the hash of a value.
fn hash_value<V: Hash, HB: BuildHasher>(value: &V, builder: &HB) -> u64 {
    let mut hasher = builder.build_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A builder for transition systems.
#[derive(Clone)]
pub struct TsBuilder<S, L> {
    /// The data of the transition.
    data: TsData<S, L>,
    /// Table used for deduplication of states.
    state_table: RawTable<StateId>,
    /// Table used for deduplication of transitions.
    transition_table: RawTable<TransitionId>,
    /// The hash builder for deduplication.
    hash_builder: DefaultHashBuilder,

    insert_called: usize,
}

impl<S: Debug, L: Debug> Debug for TsBuilder<S, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TsBuilder")
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

impl<S, L> TsBuilder<S, L> {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            data: TsData::new(),
            state_table: RawTable::new(),
            transition_table: RawTable::new(),
            hash_builder: DefaultHashBuilder::default(),
            insert_called: 0,
        }
    }

    /// Mark a state as initial state.
    pub fn mark_initial(&mut self, state: StateId) {
        self.data.initial_states.insert(state);
    }

    /// Looks up the id of the state in the table.
    pub fn lookup_state(&self, state: &S) -> Option<StateId>
    where
        S: Eq + Hash,
    {
        let hash = hash_value(state, &self.hash_builder);
        self.state_table
            .get(hash, |id| &self.data[*id].state == state)
            .cloned()
    }

    /// Inserts a state into the transition system.
    pub fn insert_state(&mut self, state: S) -> StateId
    where
        S: Eq + Hash,
    {
        let hash = hash_value(&state, &self.hash_builder);
        let id = self
            .state_table
            .get(hash, |id| self.data[*id].state == state)
            .cloned();

        match id {
            Some(id) => id,
            None => {
                let id = StateId::from_parts(self.data.ts_id, self.data.states.len());
                self.data.states.push(StateEntry::new(state));
                self.state_table.insert(hash, id, |id| {
                    hash_value(&self.data[*id].state, &self.hash_builder)
                });
                id
            }
        }
    }

    /// Inserts a transition into the transition system.
    pub fn insert_transition(&mut self, source: StateId, label: L, target: StateId)
    where
        L: Eq + Hash,
    {
        let transition = Transition {
            source,
            label,
            target,
        };
        self.insert_called += 1;
        let hash = hash_value(&transition, &self.hash_builder);
        let id = self
            .transition_table
            .get(hash, |id| self.data.transitions[id.0] == transition)
            .cloned();
        if id.is_none() {
            let id = TransitionId(self.data.transitions.len());
            self.data.transitions.push(transition);
            self.data.outgoing.push(id);
            self.data.reverse.push(id);
            self.transition_table.insert(hash, id, |id| {
                hash_value(&self.data.transitions[id.0], &self.hash_builder)
            });
        }
    }

    /// Builds the TS.
    pub fn build(self) -> Ts<S, L> {
        println!("Insert transition called {} times.", self.insert_called);
        self.data.into_ts()
    }
}

impl<S, L> BaseTs for TsBuilder<S, L> {
    type StateId = StateId;

    type State = S;

    fn get_label(&self, id: &Self::StateId) -> &Self::State {
        &self.data[*id].state
    }
}

impl<S, L> States for TsBuilder<S, L> {
    type StatesIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn states(&self) -> Self::StatesIter<'_> {
        (0..self.num_states()).map(|idx| StateId::from_parts(self.data.ts_id, idx))
    }

    fn num_states(&self) -> usize {
        self.data.states.len()
    }
}

impl<S, L> InitialStates for TsBuilder<S, L> {
    type InitialStatesIter<'iter> = impl 'iter + Iterator<Item = StateId>
    where
        Self: 'iter;

    fn initial_states(&self) -> Self::InitialStatesIter<'_> {
        self.data.initial_states.iter().cloned()
    }

    fn is_initial(&self, state: &Self::StateId) -> bool {
        self.data.initial_states.contains(state)
    }

    fn num_initial_states(&self) -> usize {
        self.data.initial_states.len()
    }
}
