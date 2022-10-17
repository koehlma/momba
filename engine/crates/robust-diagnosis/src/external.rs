//! Data structures for interfacing with other tools.

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use momba_explore::model;

/// The type used to represent time.
pub type Time = NotNan<f64>;

/// A *labeled action* with arguments.
#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Clone, Debug)]
pub struct LabeledAction {
    /// The label of the action.
    pub label: String,
    /// The arguments of the action.
    pub arguments: Vec<model::Value>,
}

/// An *event* happened at a certain point in time.
#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Clone, Debug)]
pub struct TimedEvent {
    /// The time the event took place.
    pub time: Time,
    /// The labeled action associated with the event.
    pub action: LabeledAction,
}

/// A latency interval captured by a *base latency* and *jitter bound*.
#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Clone, Debug)]
pub struct LatencyInterval {
    /// The base latency.
    pub base_latency: Time,
    /// The jitter bound.
    pub jitter_bound: Time,
}

impl LatencyInterval {
    /// Returns the minimum of the latency interval.
    pub fn min_latency(&self) -> Time {
        self.base_latency - self.jitter_bound
    }

    /// Returns the maximum of the latency interval.
    pub fn max_latency(&self) -> Time {
        self.base_latency + self.jitter_bound
    }
}

/// A timed observation of a labeled action.
#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Clone, Debug)]
pub struct TimedObservation {
    /// The time the observation has arrived.
    pub time: Time,
    /// The action associated with the observation.
    pub action: LabeledAction,
    /// The latency the observation may have.
    pub latency: LatencyInterval,
}

/// Timing imprecisions observations are subject to.
#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Clone, Debug)]
pub struct Imprecisions {
    /// The clock drift parameter ε.
    pub clock_drift: Time,

    /// The minimal latency of any observation.
    pub min_latency: Time,
    /// The maximal latency of any observation.
    pub max_latency: Time,
}
