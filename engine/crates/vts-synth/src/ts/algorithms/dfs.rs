//! Depth-first search.

use crate::ts::traits::*;

/// An TS supporting depth-first search.
pub trait SupportsDfs: TsRef + MakeDenseStateSet + Successors {}

impl<TS: TsRef + MakeDenseStateSet + Successors> SupportsDfs for TS {}

/// An item on the stack of a depth-first search.
#[derive(Debug, Clone)]
pub enum StackItem<S> {
    /// Visit the given state.
    Visit(S),
    /// Finalize the given state.
    Finalize(S),
}

impl<S> StackItem<S> {
    /// The transition system state of the stack item.
    pub fn state(&self) -> &S {
        match self {
            StackItem::Visit(state) | StackItem::Finalize(state) => state,
        }
    }
}

/// State a depth-first search.
pub struct Dfs<TS: SupportsDfs> {
    ts: TS,
    visited: TS::DenseStateSet,
    stack: Vec<StackItem<TS::StateId>>,
}

impl<TS: SupportsDfs> Dfs<TS> {
    /// Initiates a new depth-first search starting from the initial states.
    pub fn new(ts: TS) -> Self
    where
        TS: InitialStates,
    {
        Self::new_with(ts, ts.initial_states())
    }

    /// Initiates a new depth-first search starting from the provided states.
    pub fn new_with<I>(ts: TS, start: I) -> Self
    where
        I: IntoIterator<Item = TS::StateId>,
    {
        Self {
            ts,
            visited: ts.make_dense_state_set(),
            stack: start.into_iter().map(StackItem::Visit).collect(),
        }
    }

    /// Initiates a new depth-first search without any states.
    pub fn empty(ts: TS) -> Self {
        Self::new_with(ts, [])
    }

    /// The TS over which the search is performed.
    pub fn ts(&self) -> &TS {
        &self.ts
    }

    /// Pushes a state on the DFS stack.
    pub fn push(&mut self, state: TS::StateId) -> bool {
        if self.visited.contains(&state) {
            false
        } else {
            self.stack.push(StackItem::Visit(state));
            true
        }
    }

    /// Indicates whether the given state has been visited.
    pub fn has_been_visited(&self, state: &TS::StateId) -> bool {
        self.visited.contains(&state)
    }

    /// Indicates whether the given state is on the stack.
    pub fn is_on_stack(&self, state: &TS::StateId) -> bool {
        self.stack.iter().any(|item| item.state() == state)
    }

    /// Indicates whether the stack is empty.
    pub fn is_stack_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Consumes a stack item and advances the search.
    pub fn consume(&mut self) -> Option<StackItem<TS::StateId>> {
        if let Some(item) = self.stack.pop() {
            if let StackItem::Visit(state) = &item {
                if self.visited.insert(state.clone()) {
                    self.stack.push(StackItem::Finalize(state.clone()));
                    let ts = self.ts;
                    for successor in ts.successors(state) {
                        self.push(successor);
                    }
                }
            }
            Some(item)
        } else {
            None
        }
    }

    /// Iterator over states in pre-order.
    pub fn iter_pre_order(&mut self) -> PreOrderIter<'_, TS> {
        PreOrderIter(self)
    }

    /// Iterator over states in pre-order.
    pub fn iter_post_order(&mut self) -> PostOrderIter<'_, TS> {
        PostOrderIter(self)
    }
}

/// Iterator over states in DFS pre-order.
pub struct PreOrderIter<'dfs, TS: SupportsDfs>(&'dfs mut Dfs<TS>);

impl<'dfs, TS: SupportsDfs> Iterator for PreOrderIter<'dfs, TS> {
    type Item = TS::StateId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.0.consume() {
            if let StackItem::Visit(state) = item {
                return Some(state);
            }
        }
        None
    }
}

/// Iterator over states in DFS post-order.
pub struct PostOrderIter<'dfs, TS: SupportsDfs>(&'dfs mut Dfs<TS>);

impl<'dfs, TS: SupportsDfs> Iterator for PostOrderIter<'dfs, TS> {
    type Item = TS::StateId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.0.consume() {
            if let StackItem::Finalize(state) = item {
                return Some(state);
            }
        }
        None
    }
}
