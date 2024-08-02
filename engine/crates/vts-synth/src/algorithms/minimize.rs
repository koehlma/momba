use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use indexmap::IndexMap;
use tracing::{info, span, warn, Level};

use crate::ts::{
    traits::{
        BaseTs, InitialStates, MakeDenseStateMap, MakeSparseStateSet, StateMap, StateSet, States,
        Transitions,
    },
    types::{Vts, VtsState},
    SparseStateSet, StateId, TransitionId, TsBuilder,
};

/// An index into an [`IdxVec`].
pub trait Idx: Copy {
    /// Constructs an index from [`usize`].
    fn from_usize(idx: usize) -> Self;

    /// Converts the index to [`usize`].
    fn as_usize(self) -> usize;

    /// Increment the index.
    fn increment(&mut self) {
        *self = Self::from_usize(self.as_usize() + 1)
    }
}

/// A vector with a typed index.
pub struct IdxVec<I, T> {
    data: Vec<T>,
    _phantom_index: PhantomData<fn(&I)>,
}

impl<I, T> IdxVec<I, T> {
    /// Creates an empty [`IdxVec`].
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Converts a [`Vec`] to an [`IdxVec`].
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            _phantom_index: PhantomData,
        }
    }
}

impl<I, T: Clone> Clone for IdxVec<I, T> {
    fn clone(&self) -> Self {
        Self::from_vec(self.data.clone())
    }
}

impl<I, T> Default for IdxVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> IdxVec<I, T> {
    /// Adds a value to the vector.
    pub fn push(&mut self, value: T) -> I {
        let idx = self.next_idx();
        self.data.push(value);
        idx
    }

    /// The index of the last element.
    pub fn last_idx(&self) -> I {
        assert!(self.data.len() > 0);
        I::from_usize(self.data.len() - 1)
    }

    /// A mutable reference to the last element.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.data.last_mut()
    }

    /// The index of the next element.
    pub fn next_idx(&self) -> I {
        I::from_usize(self.data.len())
    }
}

impl<I: Idx, T> std::ops::Index<I> for IdxVec<I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.data[index.as_usize()]
    }
}

impl<I: Idx, T> std::ops::IndexMut<I> for IdxVec<I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index.as_usize()]
    }
}

impl<I: Idx, T> std::ops::Index<std::ops::Range<I>> for IdxVec<I, T> {
    type Output = [T];

    fn index(&self, index: std::ops::Range<I>) -> &Self::Output {
        &self.data[index.start.as_usize()..index.end.as_usize()]
    }
}

macro_rules! new_idx_type {
    ($(#[$meta:meta])* $name:ident($int:ty)) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($int);

        impl Idx for $name {
            fn from_usize(idx: usize) -> Self {
                #[cfg(debug_assertions)]
                return Self(idx.try_into().unwrap());
                #[cfg(not(debug_assertions))]
                return Self(idx as $int);
            }


            fn as_usize(self) -> usize {
                self.0 as usize
            }
        }
    };
}

new_idx_type! {
    /// Uniquely identifies a location in [`Partition`].
    LocationIdx(u32)
}

new_idx_type! {
    /// Uniquely identifies a class of a partition.
    ClsIdx(u32)
}

impl ClsIdx {
    /// Invalid class index.
    pub fn invalid() -> Self {
        Self(u32::MAX)
    }

    /// Returns whether the index is invalid.
    pub fn is_invalid(self) -> bool {
        self.0 == u32::MAX
    }
}

/// A partition of a set.
///
/// Based on *Efficient Minimization of DFAs with Partial Transitions* by Almari and Lehtinen.
///
/// [1]: https://doi.org/10.4230/LIPIcs.STACS.2008.1328
pub struct Partition<E> {
    location_to_element: IdxVec<LocationIdx, E>,
    element_to_location: IdxVec<E, LocationIdx>,
    cls: IdxVec<E, ClsIdx>,
    first: IdxVec<ClsIdx, LocationIdx>,
    marker: IdxVec<ClsIdx, LocationIdx>,
    last: IdxVec<ClsIdx, LocationIdx>,
}

impl<E: Idx> Partition<E> {
    /// Creates a new partition of an empty set.
    pub fn new(capacity: usize) -> Self {
        // let mut last = IdxVec::new();
        // last.push(LocationIdx::from_usize(0));
        Self {
            location_to_element: IdxVec::new(),
            element_to_location: IdxVec::from_vec(vec![LocationIdx::from_usize(0); capacity]),
            cls: IdxVec::from_vec(vec![ClsIdx::invalid(); capacity]),
            first: IdxVec::new(),
            marker: IdxVec::new(),
            last: IdxVec::new(),
        }
    }

    /// Iterator over the classes.
    pub fn iter_classes(&self) -> impl Iterator<Item = ClsIdx> {
        (0..self.first.data.len()).map(ClsIdx::from_usize)
    }

    /// Returns the number of classes.
    pub fn num_classes(&self) -> usize {
        self.first.data.len()
    }

    pub fn last_cls(&self) -> ClsIdx {
        ClsIdx(self.first.data.len() as u32)
    }

    pub fn size(&self, cls: ClsIdx) -> usize {
        self.last[cls].as_usize() - self.first[cls].as_usize()
    }

    /// Creates a new class.
    pub fn create_class(&mut self) -> ClsIdx {
        let first = self.location_to_element.next_idx();
        let cls = self.first.push(first);
        self.last.push(first);
        self.marker.push(first);
        cls
    }

    /// Adds an element to the current class.
    pub fn push(&mut self, element: E) {
        let mut location = self.location_to_element.push(element);
        self.element_to_location[element] = location;
        self.cls[element] = self.first.last_idx();
        location.increment();
        *self.last.last_mut().unwrap() = location;
    }

    /// Returns the class of the element.
    pub fn cls_of(&self, element: E) -> ClsIdx {
        self.cls[element]
    }

    /// Marks an element of a class.
    pub fn mark(&mut self, element: E) {
        let cls = self.cls_of(element);
        let location = self.element_to_location[element];
        let marker = self.marker[cls];
        if location >= marker {
            self.location_to_element[location] = self.location_to_element[marker];
            self.element_to_location[self.location_to_element[location]] = location;
            self.location_to_element[marker] = element;
            self.element_to_location[element] = marker;
            self.marker[cls].increment();
        }
    }

    /// Resets the marker.
    pub fn reset_marker(&mut self, cls: ClsIdx) {
        self.marker[cls] = self.first[cls];
    }

    /// Splits a class based on the marked elements.
    pub fn split(&mut self, cls: ClsIdx) -> Option<ClsIdx> {
        if self.marker[cls] == self.last[cls] {
            self.marker[cls] = self.first[cls];
        }
        if self.marker[cls] == self.first[cls] {
            None
        } else {
            let new_cls = self.first.push(self.first[cls]);
            self.marker.push(self.first[cls]);
            self.last.push(self.marker[cls]);
            self.first[cls] = self.marker[cls];
            let mut location = self.first[new_cls];
            while location < self.last[new_cls] {
                self.cls[self.location_to_element[location]] = new_cls;
                location.increment();
            }
            Some(new_cls)
        }
    }

    /// The elements of the class.
    pub fn elements(&self, cls: ClsIdx) -> &[E] {
        let first = self.first[cls];
        let last = self.last[cls];
        &self.location_to_element[first..last]
    }

    /// The marked elements of the class.
    pub fn marked(&self, cls: ClsIdx) -> &[E] {
        let first = self.first[cls];
        let marker = self.marker[cls];
        &self.location_to_element[first..marker]
    }

    /// The unmarked elements of the class.
    pub fn unmarked(&self, cls: ClsIdx) -> &[E] {
        let marker = self.marker[cls];
        let last = self.last[cls];
        &self.location_to_element[marker..last]
    }

    /// Returns whether a class has marked elements.
    pub fn has_marks(&self, cls: ClsIdx) -> bool {
        self.marker[cls] != self.first[cls]
    }

    #[track_caller]
    pub fn assert_valid(&self) {
        for cls in self.iter_classes() {
            assert!(
                self.first[cls] < self.last[cls],
                "Classes must be non-empty."
            );
        }
    }
}

impl Idx for StateId {
    fn from_usize(idx: usize) -> Self {
        StateId::from_parts(0, idx)
    }

    fn as_usize(self) -> usize {
        self.idx()
    }
}

impl Idx for TransitionId {
    fn from_usize(idx: usize) -> Self {
        TransitionId(idx)
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

pub struct MinimizeFast<'cx, Q, V, A> {
    vts: &'cx Vts<Q, V, A>,
    language_insensitive: bool,
    eliminate_self_loops: bool,
    state_partition: Partition<StateId>,
    transition_partition: Partition<TransitionId>,
    touched_splitters: Vec<ClsIdx>,
    waiting_splitters: Vec<ClsIdx>,
}

impl<'cx, Q, V, A> MinimizeFast<'cx, Q, V, A>
where
    V: Clone + Hash + Eq,
    A: Debug + Clone + Hash + Eq,
{
    pub fn new(vts: &'cx Vts<Q, V, A>) -> Self {
        Self {
            vts,
            language_insensitive: false,
            eliminate_self_loops: false,
            state_partition: Partition::new(vts.num_states()),
            transition_partition: Partition::new(vts.num_transitions()),
            touched_splitters: Vec::new(),
            waiting_splitters: Vec::new(),
        }
    }

    pub fn with_language_insensitive(mut self, language_insensitive: bool) -> Self {
        self.language_insensitive = language_insensitive;
        self.eliminate_self_loops = language_insensitive;
        self
    }

    pub fn with_eliminate_self_loops(mut self, eliminate_self_loops: bool) -> Self {
        self.eliminate_self_loops = eliminate_self_loops;
        self
    }

    #[track_caller]
    fn sanity_check_state_partition(&self) {
        self.state_partition.assert_valid();
        assert_eq!(
            self.state_partition
                .iter_classes()
                .map(|cls| self.state_partition.elements(cls).len())
                .sum::<usize>(),
            self.vts.num_states(),
            "No states of the original VTS must get lost."
        );
    }

    #[track_caller]
    fn sanity_check_transition_partition(&self) {
        self.transition_partition.assert_valid();
        assert_eq!(
            self.transition_partition
                .iter_classes()
                .map(|cls| self.transition_partition.elements(cls).len())
                .sum::<usize>(),
            self.vts.num_transitions(),
            "No transitions of the original VTS must get lost."
        );
    }

    fn transition_cls_target(&self, transition_cls: ClsIdx) -> ClsIdx {
        self.state_partition.cls_of(
            self.vts
                .get_transition(
                    *self
                        .transition_partition
                        .elements(transition_cls)
                        .iter()
                        .next()
                        .unwrap(),
                )
                .target(),
        )
    }

    #[track_caller]
    fn sanity_check_partition_consistency(&self) {
        for transition_cls in self.transition_partition.iter_classes() {
            let target_cls = self.transition_cls_target(transition_cls);
            for transition_id in self.transition_partition.elements(transition_cls) {
                let transition = self.vts.get_transition(*transition_id);
                assert_eq!(self.state_partition.cls_of(transition.target()), target_cls);
            }
        }
    }

    fn emit_partition_info(&self) {
        info!(
            "Partitions have {} state and {} transition classes.",
            self.state_partition.num_classes(),
            self.transition_partition.num_classes()
        );
    }

    fn initialize_partitions(&mut self) {
        self.vts.assert_invariants();

        // Assert that the partitions have not been initialized yet.
        assert!(self.state_partition.num_classes() == 0);
        assert!(self.transition_partition.num_classes() == 0);

        // Initialize state partition.
        let mut verdict_to_states = IndexMap::new();
        for state_id in self.vts.states() {
            let verdict = &self.vts.get_label(&state_id).verdict;
            verdict_to_states
                .entry(verdict)
                .or_insert_with(Vec::new)
                .push(state_id);
        }
        assert_eq!(
            verdict_to_states
                .values()
                .map(|cls_states| cls_states.len())
                .sum::<usize>(),
            self.vts.num_states(),
            "No states of the original VTS must get lost."
        );
        for cls_states in verdict_to_states.values() {
            self.state_partition.create_class();
            for cls_state in cls_states {
                self.state_partition.push(*cls_state);
            }
        }
        assert_eq!(
            verdict_to_states.len(),
            self.state_partition.num_classes(),
            "Number of initial classes is incorrect."
        );
        self.sanity_check_state_partition();

        // Initialize transition partition.
        let mut label_to_transitions = IndexMap::new();
        for state_cls in self.state_partition.iter_classes() {
            label_to_transitions.clear();
            for state in self.state_partition.elements(state_cls) {
                for transition_id in self.vts.incoming_ids(state) {
                    let transition = self.vts.get_transition(transition_id);
                    assert_eq!(transition.target(), *state);
                    label_to_transitions
                        .entry(transition.action())
                        .or_insert_with(Vec::new)
                        .push(transition_id);
                }
            }
            if label_to_transitions.is_empty() {
                continue;
            }
            // if state_cls.as_usize() > 0 {
            //     self.transition_partition.create_class();
            // }
            for cls_transitions in label_to_transitions.values() {
                self.transition_partition.create_class();
                for cls_transition in cls_transitions {
                    self.transition_partition.push(*cls_transition);
                }
            }
        }
        self.emit_partition_info();

        self.sanity_check_transition_partition();
        self.sanity_check_partition_consistency();
    }

    fn split_class(&mut self, state_cls: ClsIdx) {
        if let Some(mut new_state_cls) = self.state_partition.split(state_cls) {
            if self.state_partition.size(state_cls) < self.state_partition.size(new_state_cls) {
                new_state_cls = state_cls;
            }
            let state_cls = new_state_cls;
            for state in self.state_partition.elements(state_cls) {
                for transition_id in self.vts.incoming_ids(state) {
                    let transition_cls = self.transition_partition.cls_of(transition_id);
                    if !self.transition_partition.has_marks(transition_cls) {
                        self.touched_splitters.push(transition_cls);
                    }
                    self.transition_partition.mark(transition_id);
                }
            }
            while let Some(splitter) = self.touched_splitters.pop() {
                if let Some(new_transition_cls) = self.transition_partition.split(splitter) {
                    self.waiting_splitters.push(new_transition_cls);
                }
            }
        }
    }

    fn refine(&mut self) {
        self.waiting_splitters
            .extend(self.transition_partition.iter_classes());
        let mut touched_classes = Vec::new();
        let mut iteration = 0;
        while let Some(splitter) = self.waiting_splitters.pop() {
            // self.sanity_check_partition_consistency();
            if iteration % 10_000 == 0 {
                info!("Waiting splitters {}.", self.waiting_splitters.len());
                self.emit_partition_info();
            }
            iteration += 1;
            let action = self
                .vts
                .get_transition(
                    *self
                        .transition_partition
                        .elements(splitter)
                        .iter()
                        .next()
                        .unwrap(),
                )
                .action();
            for transition_id in self.transition_partition.elements(splitter) {
                let source = self.vts.get_transition(*transition_id).source();
                let source_cls = self.state_partition.cls_of(source);
                if !self.state_partition.has_marks(source_cls) {
                    touched_classes.push(source_cls);
                }
                self.state_partition.mark(source);
            }
            while let Some(state_cls) = touched_classes.pop() {
                let mut do_split = !self.language_insensitive;
                if self.language_insensitive {
                    for unmarked in self.state_partition.unmarked(state_cls) {
                        for transition in self.vts.outgoing(unmarked) {
                            if transition.action() == action {
                                do_split = true;
                                break;
                            }
                        }
                    }
                }
                if do_split {
                    self.split_class(state_cls);
                } else {
                    self.state_partition.reset_marker(state_cls)
                }
            }
        }
    }

    /// Minimizes the given `vts` using a variant of Hopcroft's algorithm[^1].
    ///
    /// If `language_insensitive` is set, the language of the original VTS is not preserved.
    ///
    /// [^1]: John Hopcroft. [An `n log n` Algorithm for Minimizing States in Finite Automata](https://doi.org/10.1016/B978-0-12-417750-5.50022-1). 1971.
    pub fn run(mut self) -> Vts<SparseStateSet, V, A> {
        let _span = span!(Level::INFO, "minimize_fast").entered();

        info!(
            "Minimizing VTS with {} states and {} transitions.",
            self.vts.num_states(),
            self.vts.num_transitions(),
        );

        if self.eliminate_self_loops && !self.language_insensitive {
            warn!("Removing self loops will change the accepted language.");
        }

        self.initialize_partitions();

        self.sanity_check_state_partition();
        self.sanity_check_transition_partition();
        self.sanity_check_partition_consistency();

        info!("Refining equivalence classes.");
        self.refine();

        self.sanity_check_state_partition();
        self.sanity_check_transition_partition();
        self.sanity_check_partition_consistency();

        // 1️⃣ Partition the states based on the verdicts they yield.

        // 3️⃣ Build the minimized VTS using the computed classes.
        info!("Constructing minimized VTS using the computed classes.");
        let mut builder = TsBuilder::new();

        // 3️⃣ Build a lookup tables for equivalence classes.
        let mut state_to_cls = self.vts.make_dense_state_map();
        let mut cls_to_verdict = Vec::new();
        let mut cls_to_set = Vec::new();
        for state_cls in self.state_partition.iter_classes() {
            let mut cls_set = self.vts.make_sparse_state_set();
            for state in self.state_partition.elements(state_cls) {
                cls_set.insert(*state);
                state_to_cls.insert(*state, state_cls);
                // TODO: Check whether there was no class for the state before.
                // debug_assert!(inserted.is_none(), "A state be in a single class.");
            }
            cls_to_set.push(cls_set);
            let cls_verdict = self
                .vts
                .get_label(
                    self.state_partition
                        .elements(state_cls)
                        .iter()
                        .next()
                        .expect("Equivalence classes must not be empty"),
                )
                .verdict
                .clone();
            cls_to_verdict.push(cls_verdict);
        }

        // Add the states to the new VTS.
        let cls_states = self
            .state_partition
            .iter_classes()
            .map(|state_cls| {
                let cls_set = cls_to_set[state_cls.as_usize()].clone();
                let verdict = cls_to_verdict[state_cls.as_usize()].clone();
                let cls_state = builder.insert_state(VtsState::new(cls_set, verdict));
                for state in self.state_partition.elements(state_cls) {
                    if self.vts.is_initial(&state) {
                        builder.mark_initial(cls_state);
                        break;
                    }
                }
                cls_state
            })
            .collect::<Vec<_>>();
        // Add the transitions to the new VTS.
        for (source_cls_idx, source) in cls_states.iter().enumerate() {
            for state in cls_to_set[source_cls_idx].iter() {
                for transition in self.vts.outgoing(&state) {
                    let target_cls_idx = state_to_cls
                        .get(&transition.target())
                        .expect("Every state must belong to a class.")
                        .as_usize();
                    if !self.eliminate_self_loops || source_cls_idx != target_cls_idx {
                        let target = cls_states[target_cls_idx];
                        builder.insert_transition(*source, transition.action().clone(), target);
                    }
                }
            }
        }

        builder.build()
    }
}
