//! Implementation of the [`Observer`] responsible for managing *observations*.

use indexmap::IndexSet;
use ordered_float::NotNan;

use momba_explore::LabeledAction;

/// A *timed* observation of a specific labeled action.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Observation {
    /// The *observation time*.
    pub time: NotNan<f64>,
    /// The labeled action that has been observed.
    pub action: LabeledAction,

    /// The *base latency* of the observation.
    pub base_latency: NotNan<f64>,
    /// The *jitter bound* of the observation.
    pub jitter_bound: NotNan<f64>,

    /// The *maximal latency* of the observation.
    pub max_latency: NotNan<f64>,
    /// The *minimal latency* of the observation.
    pub min_latency: NotNan<f64>,
}

impl Observation {
    /// Constructs a new [Observation] with the provided parameters.
    pub fn new(
        time: NotNan<f64>,
        action: LabeledAction,
        base_latency: NotNan<f64>,
        jitter_bound: NotNan<f64>,
    ) -> Self {
        Observation {
            time,
            action,
            base_latency,
            jitter_bound,
            max_latency: base_latency + jitter_bound,
            min_latency: base_latency - jitter_bound,
        }
    }
}

/// A closed [NotNan&lt;f64&gt;][NotNan] interval.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct ClosedInterval {
    /// The lower bound of the closed interval.
    pub lower_bound: NotNan<f64>,
    /// The upper bound of the closed interval.
    pub upper_bound: NotNan<f64>,
}

/// Represents timing imprecisions observations are subject to.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Imprecisions {
    /// Clock drift of the observer relative to the system.
    pub clock_drift: NotNan<f64>,

    /// The maximal latency of any observation.
    pub max_latency: NotNan<f64>,
    /// The minimal latency of any observation.
    pub min_latency: NotNan<f64>,

    /// The minimal drift slope.
    pub min_drift_slope: NotNan<f64>,
    /// The maximal drift slope.
    pub max_drift_slope: NotNan<f64>,
}

impl Imprecisions {
    /// Constructs a new [Imprecisions] with the provided parameters.
    pub fn new(
        clock_drift: NotNan<f64>,
        max_latency: NotNan<f64>,
        min_latency: NotNan<f64>,
    ) -> Self {
        Imprecisions {
            clock_drift,
            max_latency,
            min_latency,
            min_drift_slope: NotNan::new(1.0 / (1.0 + clock_drift.into_inner())).unwrap(),
            max_drift_slope: NotNan::new(1.0 + clock_drift.into_inner()).unwrap(),
        }
    }

    /// Checks whether the observation is valid given the imprecisions.
    #[must_use]
    pub fn validate(&self, observation: &Observation) -> bool {
        observation.base_latency > NotNan::new(0.0).unwrap()
            && observation.jitter_bound > NotNan::new(0.0).unwrap()
            && observation.max_latency <= self.max_latency
            && observation.min_latency >= self.min_latency
    }

    pub fn approximate_drift_delta(&self, left: NotNan<f64>, right: NotNan<f64>) -> ClosedInterval {
        if left > right {
            ClosedInterval {
                lower_bound: (left - right) * self.min_drift_slope,
                upper_bound: (left - right) * self.max_drift_slope,
            }
        } else {
            ClosedInterval {
                lower_bound: (left - right) * self.max_drift_slope,
                upper_bound: (left - right) * self.min_drift_slope,
            }
        }
    }

    pub fn approximate_delta(&self, left: &Observation, right: &Observation) -> ClosedInterval {
        let delta = self.approximate_drift_delta(left.time, right.time);
        ClosedInterval {
            lower_bound: delta.lower_bound - left.max_latency + right.min_latency,
            upper_bound: delta.upper_bound - left.min_latency + right.max_latency,
        }
    }

    /// Computes the *difference bound* between both observations.
    pub fn compute_bound(&self, left: &Observation, right: &Observation) -> NotNan<f64> {
        self.approximate_delta(left, right).upper_bound
    }

    /// Computes whether any event corresponding to `observation` is guaranteed to
    /// have happened before any event corresponding to `before`.
    pub fn happens_before(&self, observation: &Observation, before: &Observation) -> bool {
        self.compute_bound(observation, before) < NotNan::new(0.0).unwrap()
    }

    /// Computes whether any event corresponding to `observation` is guaranteed to
    /// have happened after any event corresponding to `after`.
    pub fn happens_after(&self, observation: &Observation, after: &Observation) -> bool {
        self.happens_before(after, observation)
    }
}

/// Uniquely identifies an observation fed into [Observer].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct ObservationIndex(usize);

/// Maintains a set of observations and their relationships.
///
/// We use a transitivity reduced DAG to maintain the relationships between observations.
pub struct Observer {
    pub imprecisions: Imprecisions,

    pub observations: Vec<Observation>,

    before: Vec<IndexSet<ObservationIndex>>,
    after: Vec<IndexSet<ObservationIndex>>,

    roots: IndexSet<ObservationIndex>,
    horizon: IndexSet<ObservationIndex>,
}

impl Observer {
    pub fn new(imprecisions: Imprecisions) -> Self {
        Observer {
            imprecisions,
            observations: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
            roots: IndexSet::new(),
            horizon: IndexSet::new(),
        }
    }

    /// Creates a direct relationship between both observations.
    fn connect(&mut self, before: ObservationIndex, after: ObservationIndex) {
        debug_assert!(self
            .imprecisions
            .happens_before(&self.observations[before.0], &self.observations[after.0]));
        self.after[before.0].insert(after);
        self.before[after.0].insert(before);
        self.horizon.remove(&before);
        self.roots.remove(&after);
    }

    /// Removes a direct relationship between both observations.
    fn disconnect(&mut self, before: ObservationIndex, after: ObservationIndex) {
        debug_assert!(self.after[before.0].remove(&after));
        debug_assert!(self.before[after.0].remove(&before));
    }

    /// Returns the set of observations that might correspond to an event happened
    /// before the events corresponding to all other observations.
    pub fn roots(&self) -> &IndexSet<ObservationIndex> {
        &self.roots
    }

    /// Returns the set of observations that might correspond to an event happened
    /// after the events corresponding to all other observations.
    pub fn horizon(&self) -> &IndexSet<ObservationIndex> {
        &self.horizon
    }

    /// Returns a reference to the observations.
    pub fn get(&self, index: ObservationIndex) -> &Observation {
        &self.observations[index.0]
    }

    /// Returns a transitivity reduced set of observations whose corresponding events
    /// have happened after the event corresponding to `observation`.
    pub fn after(&self, index: ObservationIndex) -> &IndexSet<ObservationIndex> {
        &self.after[index.0]
    }

    /// Returns a transitivity reduced set of observations whose corresponding events
    /// have happened before the event corresponding to `observation`.
    pub fn before(&self, index: ObservationIndex) -> &IndexSet<ObservationIndex> {
        &self.before[index.0]
    }

    /// Inserts an observation into the graph and returns its reference.
    pub fn insort(&mut self, observation: Observation) -> ObservationIndex {
        let index = ObservationIndex(self.observations.len());
        self.observations.push(observation);

        self.before.push(IndexSet::new());
        self.after.push(IndexSet::new());

        let mut visited = self.horizon.clone();

        let mut pending = self.horizon.iter().map(|index| *index).collect::<Vec<_>>();

        // the set of observations happening after `observation`
        let mut observations_after: IndexSet<ObservationIndex> = IndexSet::new();
        // the set of observations happening before `observation`
        let mut observations_before: IndexSet<ObservationIndex> = IndexSet::new();

        while let Some(other_index) = pending.pop() {
            let observation = &self.observations[index.0];
            let other_observation = &self.observations[other_index.0];

            if self
                .imprecisions
                .happens_before(other_observation, observation)
            {
                observations_before.insert(other_index);
            } else {
                if self
                    .imprecisions
                    .happens_after(observation, other_observation)
                {
                    observations_after.insert(other_index);
                }
                for before_other_reference in self.before[other_index.0].iter() {
                    if visited.insert(*before_other_reference) {
                        pending.push(*before_other_reference)
                    }
                }
            }
        }

        // the set of observations happening directly after `observation`
        let mut directly_after: IndexSet<ObservationIndex> = IndexSet::new();
        // the set of observations happening directly before `observation`
        let mut directly_before: IndexSet<ObservationIndex> = IndexSet::new();

        'outer_before: while let Some(before_observation_index) = observations_before.pop() {
            let before_observation = &self.observations[before_observation_index.0];
            let mut not_directly_before = IndexSet::new();
            for other_index in observations_before.iter() {
                let other = &self.observations[other_index.0];
                if self.imprecisions.happens_before(before_observation, other) {
                    observations_before = observations_before
                        .difference(&not_directly_before)
                        .map(|reference| *reference)
                        .collect();
                    continue 'outer_before;
                } else if self.imprecisions.happens_after(other, before_observation) {
                    not_directly_before.insert(*other_index);
                }
            }
            observations_before = observations_before
                .difference(&not_directly_before)
                .map(|reference| *reference)
                .collect();
            directly_before.insert(before_observation_index);
        }

        'outer_after: while let Some(after_observation_index) = observations_after.pop() {
            let after_observation = &self.observations[after_observation_index.0];
            let mut not_directly_after = IndexSet::new();
            for other_index in observations_before.iter() {
                let other = &self.observations[other_index.0];
                if self.imprecisions.happens_after(after_observation, other) {
                    observations_after = observations_after
                        .difference(&not_directly_after)
                        .map(|reference| *reference)
                        .collect();
                    continue 'outer_after;
                } else if self.imprecisions.happens_after(other, after_observation) {
                    not_directly_after.insert(*other_index);
                }
            }
            observations_after = observations_after
                .difference(&not_directly_after)
                .map(|reference| *reference)
                .collect();
            directly_after.insert(after_observation_index);
        }

        if directly_before.is_empty() {
            self.roots.insert(index);
        }
        if directly_after.is_empty() {
            self.horizon.insert(index);
        }

        for before in directly_before {
            self.connect(before, index);
            let mut disconnect: IndexSet<ObservationIndex> = IndexSet::new();
            for after_before in self.after[before.0].iter() {
                if directly_after.contains(after_before) {
                    disconnect.insert(*after_before);
                }
            }
            for after_before in disconnect.into_iter() {
                self.disconnect(before, after_before);
            }
        }

        for after in directly_after {
            self.connect(index, after);
        }

        index
    }
}
