//! Specialized types of transition systems.

use super::Ts;

/// A label of a guarded transition system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VatsLabel<A, G> {
    /// The action of the label.
    pub action: A,
    /// The guard of the label.
    pub guard: G,
}

impl<A, G> VatsLabel<A, G> {
    /// Creates a new GTS label.
    pub fn new(action: A, guard: G) -> Self {
        Self { action, guard }
    }
}

/// A guarded transition system.
pub type Vats<S, A, G> = Ts<S, VatsLabel<A, G>>;

/// A state of a belief transition system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VtsState<Q, B> {
    /// The control state of the state.
    pub control: Q,
    /// The belief state of the state.
    pub verdict: B,
}

impl<Q, B> VtsState<Q, B> {
    /// Creates a new VTS state.
    pub fn new(control: Q, verdict: B) -> Self {
        Self { control, verdict }
    }
}

/// A belief transition system.
pub type Vts<Q, B, L> = Ts<VtsState<Q, B>, L>;
