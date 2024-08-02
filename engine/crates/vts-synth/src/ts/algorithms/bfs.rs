//! Breadth-first search.
//!
//! 🛠️ Can we share code between DFS and BFS?

use std::collections::VecDeque;

use crate::ts::traits::*;

/// An TS supporting breadth-first search.
pub trait SupportsBfs: TsRef + MakeDenseStateSet + Successors {}

impl<TS: TsRef + MakeDenseStateSet + Successors> SupportsBfs for TS {}

/// State a breadth-first search.
pub struct Bfs<TS: SupportsBfs> {
    ts: TS,
    visited: TS::DenseStateSet,
    queue: VecDeque<TS::StateId>,
}

impl<TS: SupportsBfs> Bfs<TS> {
    /// Initiates a new breadth-first search starting from the initial states.
    pub fn new(ts: TS) -> Self
    where
        TS: InitialStates,
    {
        Self::new_with(ts, ts.initial_states())
    }

    /// Initiates a new breadth-first search starting from the provided states.
    pub fn new_with<I>(ts: TS, start: I) -> Self
    where
        I: IntoIterator<Item = TS::StateId>,
    {
        Self {
            ts,
            visited: ts.make_dense_state_set(),
            queue: start.into_iter().collect(),
        }
    }

    /// Initiates a new breadth-first search without any states.
    pub fn empty(ts: TS) -> Self {
        Self::new_with(ts, [])
    }

    /// The TS over which the search is performed.
    pub fn ts(&self) -> &TS {
        &self.ts
    }

    /// Pushes a state on the BFS queue.
    pub fn push(&mut self, state: TS::StateId) -> bool {
        if self.visited.insert(state.clone()) {
            self.queue.push_back(state);
            true
        } else {
            false
        }
    }

    /// Indicates whether the given state has been visited.
    pub fn has_been_visited(&self, state: &TS::StateId) -> bool {
        self.visited.contains(&state)
    }

    /// Indicates whether the given state is queued.
    pub fn is_queued(&self, state: &TS::StateId) -> bool {
        self.queue.iter().any(|item| item == state)
    }

    /// Indicates whether the given state has been finalized.
    pub fn has_been_finalized(&self, state: &TS::StateId) -> bool {
        self.has_been_visited(state) && !self.is_queued(state)
    }

    /// Indicates whether the queue is empty.
    pub fn is_queue_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl<TS: SupportsBfs> Iterator for Bfs<TS> {
    type Item = TS::StateId;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(state) = self.queue.pop_front() {
            let ts = self.ts;
            for successor in ts.successors(&state) {
                self.push(successor);
            }
            Some(state)
        } else {
            None
        }
    }
}
