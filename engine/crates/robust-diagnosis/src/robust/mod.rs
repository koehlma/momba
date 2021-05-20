//! An implementation of *robust real-time diagnosis*.

pub mod generate;

pub mod graph;
pub mod merged;
pub mod space;

use std::{collections::VecDeque, hash::Hash, ptr::hash};
use std::{mem, sync::atomic::AtomicUsize};
use std::{rc::Rc, time::Instant};

use hashbrown::{HashMap, HashSet};

use indexmap::IndexSet;

use serde::{de::value, Deserialize, Serialize};

use clock_zones::{clocks, Constraint, Zone};

use ordered_float::NotNan;

use momba_explore::time::Time;
use momba_explore::*;

/// A *timed* observation of a specific labeled action.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Observation {
    /// The *observation time*.
    pub(crate) time: NotNan<f64>,
    /// The labeled action that has been observed.
    pub(crate) action: LabeledAction,

    /// The *base latency* of the observation.
    pub(crate) base_latency: NotNan<f64>,
    /// The *jitter bound* of the observation.
    pub(crate) jitter_bound: NotNan<f64>,

    /// The *maximal latency* of the observation.
    pub(crate) max_latency: NotNan<f64>,
    /// The *minimal latency* of the observation.
    pub(crate) min_latency: NotNan<f64>,
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

impl Observation {
    /// Returns the *observation time*.
    pub fn time(&self) -> NotNan<f64> {
        self.time
    }

    /// Returns the labeled action that has been observed.
    pub fn action(&self) -> &LabeledAction {
        &self.action
    }
}

/// A closed [NotNan&lt;f64&gt;][NotNan] interval.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct ClosedInterval {
    pub(crate) lower_bound: NotNan<f64>,
    pub(crate) upper_bound: NotNan<f64>,
}

/// Represents timing imprecisions observations are subject to.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Imprecisions {
    /// Clock drift of the observer relative to the system.
    pub(crate) clock_drift: NotNan<f64>,

    /// The maximal latency of any observation.
    pub(crate) max_latency: NotNan<f64>,
    /// The minimal latency of any observation.
    pub(crate) min_latency: NotNan<f64>,

    /// The minimal drift slope.
    pub(crate) min_drift_slope: NotNan<f64>,
    /// The maximal drift slope.
    pub(crate) max_drift_slope: NotNan<f64>,
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

    fn approximate_drift_delta(&self, left: NotNan<f64>, right: NotNan<f64>) -> ClosedInterval {
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

    fn approximate_delta(&self, left: &Observation, right: &Observation) -> ClosedInterval {
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
type ObservationReference = usize;

/// Maintains a set of observations and their relationships.
///
/// We use a transitivity reduced DAG to maintain the relationships between observations.
pub struct Observer {
    pub(crate) imprecisions: Imprecisions,

    pub(crate) observations: Vec<Observation>,

    before: Vec<IndexSet<ObservationReference>>,
    after: Vec<IndexSet<ObservationReference>>,

    roots: IndexSet<ObservationReference>,
    horizon: IndexSet<ObservationReference>,
}

impl Observer {
    fn new(imprecisions: Imprecisions) -> Self {
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
    fn connect(&mut self, before: ObservationReference, after: ObservationReference) {
        debug_assert!(self
            .imprecisions
            .happens_before(&self.observations[before], &self.observations[after]));
        self.after[before].insert(after);
        self.before[after].insert(before);
        self.horizon.remove(&before);
        self.roots.remove(&after);
    }

    /// Removes a direct relationship between both observations.
    fn disconnect(&mut self, before: ObservationReference, after: ObservationReference) {
        debug_assert!(self.after[before].remove(&after));
        debug_assert!(self.before[after].remove(&before));
    }

    /// Returns the set of observations that might correspond to an event happened
    /// before the events corresponding to all other observations.
    pub fn roots(&self) -> &IndexSet<ObservationReference> {
        &self.roots
    }

    /// Returns the set of observations that might correspond to an event happened
    /// after the events corresponding to all other observations.
    pub fn horizon(&self) -> &IndexSet<ObservationReference> {
        &self.horizon
    }

    /// Returns a reference to the observations.
    pub fn get(&self, observation: ObservationReference) -> &Observation {
        &self.observations[observation]
    }

    /// Returns a transitivity reduced set of observations whose corresponding events
    /// have happened after the event corresponding to `observation`.
    pub fn after(&self, observation: ObservationReference) -> &IndexSet<ObservationReference> {
        &self.after[observation]
    }

    /// Returns a transitivity reduced set of observations whose corresponding events
    /// have happened before the event corresponding to `observation`.
    pub fn before(&self, observation: ObservationReference) -> &IndexSet<ObservationReference> {
        &self.before[observation]
    }

    /// Inserts an observation into the graph and returns its reference.
    pub fn insort(&mut self, observation: Observation) -> ObservationReference {
        let reference = self.observations.len();
        self.observations.push(observation);

        self.before.push(IndexSet::new());
        self.after.push(IndexSet::new());

        let mut visited = self.horizon.clone();

        let mut pending = self.horizon.iter().map(|index| *index).collect::<Vec<_>>();

        // the set of observations happening after `observation`
        let mut observations_after: IndexSet<ObservationReference> = IndexSet::new();
        // the set of observations happening before `observation`
        let mut observations_before: IndexSet<ObservationReference> = IndexSet::new();

        while let Some(other_reference) = pending.pop() {
            let observation = &self.observations[reference];
            let other_observation = &self.observations[other_reference];

            if self
                .imprecisions
                .happens_before(other_observation, observation)
            {
                observations_before.insert(other_reference);
            } else {
                if self
                    .imprecisions
                    .happens_after(observation, other_observation)
                {
                    observations_after.insert(other_reference);
                }
                for before_other_reference in self.before[other_reference].iter() {
                    if visited.insert(*before_other_reference) {
                        pending.push(*before_other_reference)
                    }
                }
            }
        }

        // the set of observation happening directly after `observation`
        let mut directly_after: IndexSet<ObservationReference> = IndexSet::new();
        // the set of observation happening directly before `observation`
        let mut directly_before: IndexSet<ObservationReference> = IndexSet::new();

        'outer_before: while let Some(before_observation_reference) = observations_before.pop() {
            let before_observation = &self.observations[before_observation_reference];
            let mut not_directly_before = IndexSet::new();
            for other_reference in observations_before.iter() {
                let other = &self.observations[*other_reference];
                if self.imprecisions.happens_before(before_observation, other) {
                    observations_before = observations_before
                        .difference(&not_directly_before)
                        .map(|reference| *reference)
                        .collect();
                    continue 'outer_before;
                } else if self.imprecisions.happens_after(other, before_observation) {
                    not_directly_before.insert(*other_reference);
                }
            }
            observations_before = observations_before
                .difference(&not_directly_before)
                .map(|reference| *reference)
                .collect();
            directly_before.insert(before_observation_reference);
        }

        'outer_after: while let Some(after_observation_reference) = observations_after.pop() {
            let after_observation = &self.observations[after_observation_reference];
            let mut not_directly_after = IndexSet::new();
            for other_reference in observations_before.iter() {
                let other = &self.observations[*other_reference];
                if self.imprecisions.happens_after(after_observation, other) {
                    observations_after = observations_after
                        .difference(&not_directly_after)
                        .map(|reference| *reference)
                        .collect();
                    continue 'outer_after;
                } else if self.imprecisions.happens_after(other, after_observation) {
                    not_directly_after.insert(*other_reference);
                }
            }
            observations_after = observations_after
                .difference(&not_directly_after)
                .map(|reference| *reference)
                .collect();
            directly_after.insert(after_observation_reference);
        }

        if directly_before.is_empty() {
            self.roots.insert(reference);
        }
        if directly_after.is_empty() {
            self.horizon.insert(reference);
        }

        for before in directly_before {
            self.connect(before, reference);
            let mut disconnect: IndexSet<ObservationReference> = IndexSet::new();
            for after_before in self.after[before].iter() {
                if directly_after.contains(after_before) {
                    disconnect.insert(*after_before);
                }
            }
            for after_before in disconnect.into_iter() {
                self.disconnect(before, after_before);
            }
        }

        for after in directly_after {
            self.connect(reference, after);
        }

        reference
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct HistoryItem {
    observation: usize,
    tracking_clock: clock_zones::Variable,
}

/// A state used by a diagnoser.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DiagnosisState {
    state: State<time::Float64Zone>,
    faults: im::HashSet<usize>,

    /// Must not be modified.
    history: VecDeque<HistoryItem>,
}

/// A prefix is a set of observation references.
type Prefix = im::HashSet<ObservationReference>;

#[derive(Clone, Debug)]
pub struct StateSet {
    pub(crate) states: HashSet<Rc<DiagnosisState>>,
    marked: HashMap<Rc<DiagnosisState>, HashSet<ObservationReference>>,
}

/// A *diagnoser* providing a diagnosis from observations.
pub struct Diagnoser {
    pub(crate) observer: Observer,
    pub(crate) explorer: momba_explore::Explorer<time::Float64Zone>,

    pub(crate) explore_counter: AtomicUsize,

    observations: im::HashSet<ObservationReference>,
    local_time: NotNan<f64>,

    prefixes: HashMap<Prefix, StateSet>,

    observable_indices: HashSet<usize>,
    fault_indices: HashSet<usize>,

    //marked: HashMap<Rc<DiagnosisState>, HashSet<usize>>,

    // Pending observations
    pending: im::HashSet<ObservationReference>,

    /// For how many observations should we apply the precise timing check?
    ///
    /// [None] means that we keep an unbounded history in states.
    history_bound: Option<usize>,
}

type CacheKey = (Rc<DiagnosisState>, Option<usize>);

enum StackItem {
    Explore(CacheKey),
    Populate((CacheKey, Vec<CacheKey>)),
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct DiagnosisResult {
    pub consistent: bool,
    pub failure_possible: bool,
    pub failure_necessary: bool,
    pub possible_failures: HashSet<usize>,
    pub necessary_failures: HashSet<usize>,
    pub states: usize,
    pub prefixes: usize,
}

impl Diagnoser {
    pub fn new(
        imprecisions: Imprecisions,
        network: model::Network,
        observable_indices: HashSet<usize>,
        fault_indices: HashSet<usize>,
        history_bound: Option<usize>,
    ) -> Self {
        let mut diagnoser = Diagnoser {
            observer: Observer::new(imprecisions),
            explorer: Explorer::new(network),

            explore_counter: AtomicUsize::new(0),

            observations: im::HashSet::new(),
            local_time: NotNan::new(0.0).unwrap(),

            prefixes: HashMap::new(),

            observable_indices,
            fault_indices,

            //marked: HashMap::new(),
            pending: im::HashSet::new(),

            history_bound,
        };

        //let mut cache = HashMap::new();

        diagnoser.prefixes.insert(
            im::HashSet::new(),
            StateSet {
                states: diagnoser.explore_states(
                    &diagnoser
                        .explorer
                        .initial_states()
                        .into_iter()
                        .map(|initial_state| {
                            Rc::new(DiagnosisState {
                                state: initial_state,
                                faults: im::HashSet::new(),
                                history: VecDeque::new(),
                            })
                        })
                        .collect::<Box<[_]>>(),
                    None,
                ),
                marked: HashMap::new(),
            },
        );

        diagnoser
    }

    pub fn result(&self) -> DiagnosisResult {
        let mut possible_failures = HashSet::new();
        let mut necessary_failures = None;

        let mut failure_necessary = true;

        let mut state_counter = 0;

        let green_observations: im::HashSet<_> = self
            .pending
            .iter()
            .filter(|observation| self.is_green(self.observer.get(**observation)))
            .cloned()
            .collect();

        for (prefix, state_set) in self.prefixes.iter() {
            state_counter += state_set.states.len();

            if !green_observations.is_subset(prefix) {
                continue;
            }

            for state in state_set.states.iter() {
                // println!("{:?}", state.failures);
                possible_failures.extend(state.faults.iter().copied());
                if state.faults.is_empty() {
                    failure_necessary = false;
                }
                match necessary_failures {
                    None => necessary_failures = Some(state.faults.clone()),
                    Some(failures) => {
                        necessary_failures = Some(failures.intersection(state.faults.clone()))
                    }
                }
            }
        }

        DiagnosisResult {
            consistent: !self.prefixes.is_empty(),
            failure_possible: !possible_failures.is_empty(),
            failure_necessary: failure_necessary,
            possible_failures: possible_failures,
            necessary_failures: necessary_failures
                .map_or_else(|| HashSet::new(), |failures| failures.into_iter().collect()),
            states: state_counter,
            prefixes: self.prefixes.len(),
        }
    }

    fn explore_states(
        &self,
        states: &[Rc<DiagnosisState>],
        expected: Option<usize>,
    ) -> HashSet<Rc<DiagnosisState>> {
        let mut stack: Vec<_> = states
            .iter()
            .map(|state| (state.clone(), expected.clone()))
            .collect();
        let mut visited: HashSet<_> = stack.iter().cloned().collect();

        let mut successors = HashSet::new();

        // let mut processed = 0;

        while let Some((state, expected)) = stack.pop() {
            let transitions = self.explorer.transitions(&state.state);

            // processed += 1;
            // if processed % 20000 == 0 {
            //     println!("processed {} states", processed);
            //     if state.history.len() != 0 {
            //         println!("{:?}", expected);
            //         println!("{:?}", state.history);
            //         for item in &state.history {
            //             println!(
            //                 "[{:?} {:?}]",
            //                 state
            //                     .state
            //                     .valuations()
            //                     .get_lower_bound(item.tracking_clock),
            //                 state
            //                     .state
            //                     .valuations()
            //                     .get_upper_bound(item.tracking_clock),
            //             )
            //         }
            //     }
            // }

            let mut should_add = false;

            for mut transition in transitions {
                let action = transition.result_action();
                let is_observable = action
                    .label_index()
                    .map(|index| self.observable_indices.contains(&index))
                    .unwrap_or(false);
                let is_expected = expected
                    .map(|expected| match &action {
                        Action::Labeled(labeled) => self.observer.get(expected).action == *labeled,
                        _ => false,
                    })
                    .unwrap_or(false);

                if is_observable {
                    if expected.is_none() {
                        should_add = true;
                    }
                    if !is_expected {
                        continue;
                    }
                }

                let mut faults = state.faults.clone();
                if let Some(label_index) = action.label_index() {
                    if self.fault_indices.contains(&label_index) {
                        faults.insert(label_index);
                    }
                }

                let mut history = state.history.clone();
                if is_expected && self.history_bound != Some(0) {
                    let (tracking_clock, mut valuations) = match self.history_bound {
                        None => {
                            let valuations = transition.valuations();
                            (
                                clock_zones::Clock::variable(valuations.num_variables()),
                                valuations.resize(valuations.num_clocks() + 1),
                            )
                        }
                        Some(n) => {
                            if history.len() >= n {
                                // recycle existing clock
                                let oldest_item = history.pop_front().unwrap();
                                (oldest_item.tracking_clock, transition.valuations().clone())
                            } else {
                                let valuations = transition.valuations();
                                (
                                    clock_zones::Clock::variable(valuations.num_variables()),
                                    valuations.resize(valuations.num_clocks() + 1),
                                )
                            }
                        }
                    };

                    valuations.reset(tracking_clock, NotNan::new(0.0).unwrap());
                    // println!("reset tracking clock, {}", history.len());
                    // println!("{:?}", valuations);
                    let observation = expected.unwrap();

                    valuations.add_constraints(
                        history
                            .iter()
                            .map(|item| {
                                let delta = self.observer.imprecisions.approximate_delta(
                                    self.observer.get(observation),
                                    self.observer.get(item.observation),
                                );
                                vec![
                                    Constraint::new_diff_le(
                                        tracking_clock,
                                        item.tracking_clock,
                                        -delta.lower_bound,
                                    ),
                                    Constraint::new_diff_le(
                                        item.tracking_clock,
                                        tracking_clock,
                                        delta.upper_bound,
                                    ),
                                ]
                                .into_iter()
                            })
                            .flatten(),
                    );

                    history.push_back(HistoryItem {
                        observation,
                        tracking_clock,
                    });

                    transition = transition.replace_valuations(valuations);
                }

                if !transition.valuations().is_empty() {
                    for destination in self
                        .explorer
                        .destinations(&state.state, &transition)
                        .into_iter()
                    {
                        let successor = (
                            Rc::new(DiagnosisState {
                                state: self.explorer.successor(
                                    &state.state,
                                    &transition,
                                    &destination,
                                ),
                                faults: faults.clone(),
                                history: history.clone(),
                            }),
                            if is_expected { None } else { expected },
                        );

                        if visited.insert(successor.clone()) {
                            stack.push(successor);
                        }
                    }
                }
            }

            if should_add {
                successors.insert(state);
            }
        }

        successors
    }

    // fn explore_states(
    //     &self,
    //     state: Rc<DiagnosisState>,
    //     expected: Option<usize>,
    //     // seen: &mut HashMap<(Rc<DiagnosisState>, Option<usize>), usize>,
    //     // trace: &mut Vec<(
    //     //     Vec<momba_explore::model::EdgeReference>,
    //     //     clock_zones::ZoneF64,
    //     // )>,
    //     cache: &mut HashMap<(Rc<DiagnosisState>, Option<usize>), im::HashSet<Rc<DiagnosisState>>>,
    // ) -> im::HashSet<Rc<DiagnosisState>> {
    //     // This optimization is probably correct but does not conform to what we describe in our paper.
    //     // if expected.is_none() {
    //     //     return im::hashset![state];
    //     // }

    //     let counter = self
    //         .explore_counter
    //         .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    //     if counter % 2000 == 0 {
    //         println!("counter: {}", counter);
    //     }

    //     let cache_key = (state.clone(), expected);

    //     // if let Some(previous) = seen.insert(cache_key.clone(), trace.len()) {
    //     //     println!("{} {}", previous, trace.len());
    //     //     for (action, valuations) in &trace[previous - 1..] {
    //     //         println!("{:?}", action);
    //     //         for left in clock_zones::clocks(valuations) {
    //     //             for right in clock_zones::clocks(valuations) {
    //     //                 println!(
    //     //                     "{:?} - {:?} {:?}",
    //     //                     left,
    //     //                     right,
    //     //                     valuations.get_bound(left, right)
    //     //                 );
    //     //             }
    //     //         }
    //     //     }
    //     //     println!("{:?}", state);
    //     //     let valuations = state.state.valuations();
    //     //     for left in clock_zones::clocks(valuations) {
    //     //         for right in clock_zones::clocks(valuations) {
    //     //             println!(
    //     //                 "{:?} - {:?} {:?}",
    //     //                 left,
    //     //                 right,
    //     //                 valuations.get_bound(left, right)
    //     //             );
    //     //         }
    //     //     }
    //     //     println!("observables: {:?}", self.observable_indices);
    //     //     for transition in self.explorer.transitions(&state.state) {
    //     //         println!("{:?}", transition.result_action());
    //     //     }
    //     //     panic!("infinite recursion");
    //     // }

    //     match cache.get(&cache_key).map(|entry| entry.clone()) {
    //         Some(result) => result,
    //         None => {
    //             let mut states = im::HashSet::new();

    //             for mut transition in self.explorer.transitions(&state.state).into_iter() {
    //                 let action = transition.result_action();
    //                 let is_observable = action
    //                     .label_index()
    //                     .map_or_else(|| false, |index| self.observable_indices.contains(&index));
    //                 let is_expected = match &expected {
    //                     None => false,
    //                     Some(expected) => match &action {
    //                         Action::Labeled(labeled) => {
    //                             self.observer.get(*expected).action == *labeled
    //                         }
    //                         _ => false,
    //                     },
    //                 };

    //                 // match transition.result_action() {
    //                 //     Action::Silent => {}
    //                 //     Action::Labeled(labeled) => {
    //                 //         println!("{:?} {} {}", labeled, is_observable, is_expected);
    //                 //     }
    //                 // };

    //                 if is_observable && !is_expected {
    //                     continue;
    //                 }

    //                 // match transition.result_action() {
    //                 //     Action::Silent => {}
    //                 //     Action::Labeled(labeled) => {
    //                 //         panic!("should not happen")
    //                 //     }
    //                 // };

    //                 let mut failures = state.failures.clone();
    //                 if let Some(label_index) = action.label_index() {
    //                     if self.fault_indices.contains(&label_index) {
    //                         failures.insert(label_index);
    //                     }
    //                 }

    //                 let mut history = state.history.clone();

    //                 if is_expected && self.history_bound != Some(0) {
    //                     let (tracking_clock, mut valuations) = match self.history_bound {
    //                         None => {
    //                             let valuations = transition.valuations();
    //                             (
    //                                 clock_zones::Clock::variable(valuations.num_clocks()),
    //                                 valuations.resize(valuations.num_clocks() + 1),
    //                             )
    //                         }
    //                         Some(n) => {
    //                             if history.len() >= n {
    //                                 // recycle existing clock
    //                                 let oldest_item = history.pop_front().unwrap();
    //                                 (oldest_item.tracking_clock, transition.valuations().clone())
    //                             } else {
    //                                 let valuations = transition.valuations();
    //                                 (
    //                                     clock_zones::Clock::variable(valuations.num_clocks()),
    //                                     valuations.resize(valuations.num_clocks() + 1),
    //                                 )
    //                             }
    //                         }
    //                     };

    //                     valuations.reset(tracking_clock, NotNan::new(0.0).unwrap());
    //                     let observation = expected.unwrap();

    //                     valuations.add_constraints(
    //                         history
    //                             .iter()
    //                             .map(|item| {
    //                                 let delta = self.observer.imprecisions.approximate_delta(
    //                                     self.observer.get(observation),
    //                                     self.observer.get(item.observation),
    //                                 );
    //                                 vec![
    //                                     Constraint::new_diff_le(
    //                                         tracking_clock,
    //                                         item.tracking_clock,
    //                                         -delta.lower_bound,
    //                                     ),
    //                                     Constraint::new_diff_le(
    //                                         item.tracking_clock,
    //                                         tracking_clock,
    //                                         delta.upper_bound,
    //                                     ),
    //                                 ]
    //                                 .into_iter()
    //                             })
    //                             .flatten(),
    //                     );

    //                     history.push_back(HistoryItem {
    //                         observation,
    //                         tracking_clock,
    //                     });

    //                     transition = transition.replace_valuations(valuations);
    //                 }

    //                 if !transition.valuations().is_empty() {
    //                     for destination in self
    //                         .explorer
    //                         .destinations(&state.state, &transition)
    //                         .into_iter()
    //                     {
    //                         let successor =
    //                             self.explorer
    //                                 .successor(&state.state, &transition, &destination);
    //                         //trace.push((transition.edges(), transition.valuations().clone()));
    //                         states = states.union(
    //                             self.explore_states(
    //                                 Rc::new(DiagnosisState {
    //                                     state: successor,
    //                                     failures: failures.clone(),
    //                                     history: history.clone(),
    //                                 }),
    //                                 if is_expected { None } else { expected },
    //                                 // seen,
    //                                 // trace,
    //                                 cache,
    //                             )
    //                             .clone(),
    //                         );
    //                         //trace.pop();
    //                     }
    //                 }
    //             }

    //             if expected.is_none() {
    //                 states.insert(state.clone());
    //             }

    //             //seen.remove(&cache_key);

    //             cache.insert((state.clone(), expected), states.clone());

    //             states
    //         }
    //     }
    // }

    // fn explore_states<'c>(
    //     &self,
    //     state: Rc<DiagnosisState>,
    //     expected: Option<usize>,
    //     cache: &'c mut HashMap<
    //         (Rc<DiagnosisState>, Option<usize>),
    //         im::HashSet<Rc<DiagnosisState>>,
    //     >,
    // ) -> &'c im::HashSet<Rc<DiagnosisState>> {
    //     // This optimization is probably correct but does not conform to what we describe in our paper.
    //     // if expected.is_none() {
    //     //     return im::hashset![state];
    //     // }

    //     let final_key = (state.clone(), expected);

    //     let mut stack = vec![StackItem::Explore(final_key.clone())];

    //     let mut pending = hashbrown::HashSet::new();
    //     pending.insert(final_key.clone());

    //     while let Some(stack_item) = stack.pop() {
    //         match stack_item {
    //             StackItem::Explore(cache_key) => {
    //                 let (state, expected) = &cache_key;

    //                 if !cache.contains_key(&cache_key) {
    //                     let mut successors = Vec::new();

    //                     for mut transition in self.explorer.transitions(&state.state).into_iter() {
    //                         let action = transition.result_action();
    //                         let is_observable = action.label_index().map_or_else(
    //                             || false,
    //                             |index| self.observable_indices.contains(&index),
    //                         );
    //                         let is_expected = match &expected {
    //                             None => false,
    //                             Some(expected) => match &action {
    //                                 Action::Labeled(labeled) => {
    //                                     self.observer.get(*expected).action == *labeled
    //                                 }
    //                                 _ => false,
    //                             },
    //                         };

    //                         if is_observable && !is_expected {
    //                             continue;
    //                         }

    //                         let mut failures = state.failures.clone();
    //                         if let Some(label_index) = action.label_index() {
    //                             if self.fault_indices.contains(&label_index) {
    //                                 failures.insert(label_index);
    //                             }
    //                         }

    //                         let mut history = state.history.clone();

    //                         if is_expected && self.history_bound != Some(0) {
    //                             let (tracking_clock, mut valuations) = match self.history_bound {
    //                                 None => {
    //                                     let valuations = transition.valuations();
    //                                     (
    //                                         clock_zones::Clock::variable(valuations.num_clocks()),
    //                                         valuations.resize(valuations.num_clocks() + 1),
    //                                     )
    //                                 }
    //                                 Some(n) => {
    //                                     if history.len() >= n {
    //                                         // recycle existing clock
    //                                         let oldest_item = history.pop_front().unwrap();
    //                                         (
    //                                             oldest_item.tracking_clock,
    //                                             transition.valuations().clone(),
    //                                         )
    //                                     } else {
    //                                         let valuations = transition.valuations();
    //                                         (
    //                                             clock_zones::Clock::variable(
    //                                                 valuations.num_clocks(),
    //                                             ),
    //                                             valuations.resize(valuations.num_clocks() + 1),
    //                                         )
    //                                     }
    //                                 }
    //                             };

    //                             valuations.reset(tracking_clock, NotNan::new(0.0).unwrap());
    //                             let observation = expected.unwrap();

    //                             valuations.add_constraints(
    //                                 history
    //                                     .iter()
    //                                     .map(|item| {
    //                                         let delta =
    //                                             self.observer.imprecisions.approximate_delta(
    //                                                 self.observer.get(observation),
    //                                                 self.observer.get(item.observation),
    //                                             );
    //                                         vec![
    //                                             Constraint::new_diff_le(
    //                                                 tracking_clock,
    //                                                 item.tracking_clock,
    //                                                 -delta.lower_bound,
    //                                             ),
    //                                             Constraint::new_diff_le(
    //                                                 item.tracking_clock,
    //                                                 tracking_clock,
    //                                                 delta.upper_bound,
    //                                             ),
    //                                         ]
    //                                         .into_iter()
    //                                     })
    //                                     .flatten(),
    //                             );

    //                             history.push_back(HistoryItem {
    //                                 observation,
    //                                 tracking_clock,
    //                             });

    //                             transition = transition.replace_valuations(valuations);
    //                         }

    //                         if !transition.valuations().is_empty() {
    //                             for destination in self
    //                                 .explorer
    //                                 .destinations(&state.state, &transition)
    //                                 .into_iter()
    //                             {
    //                                 let successor = self.explorer.successor(
    //                                     &state.state,
    //                                     &transition,
    //                                     &destination,
    //                                 );
    //                                 //println!("source: {:?}", state.state);
    //                                 //println!("result action: {:?}", transition.result_action());
    //                                 // println!("target: {:?}", successor);
    //                                 let successor_key = (
    //                                     Rc::new(DiagnosisState {
    //                                         state: successor,
    //                                         failures: failures.clone(),
    //                                         history: history.clone(),
    //                                     }),
    //                                     if is_expected { None } else { expected.clone() },
    //                                 );
    //                                 successors.push(successor_key.clone());
    //                             }
    //                         }
    //                     }

    //                     stack.push(StackItem::Populate((cache_key.clone(), successors.clone())));
    //                     for successor in successors {
    //                         // if !pending.insert(successor.clone()) {
    //                         //     println!("{:?}", state);
    //                         //     println!("{:?}", successor.0);
    //                         //     panic!("endless recursion");
    //                         // }
    //                         stack.push(StackItem::Explore(successor));
    //                     }
    //                 }
    //             }
    //             StackItem::Populate((cache_key, successors)) => {
    //                 let mut states = im::HashSet::new();
    //                 for key in successors {
    //                     // if !cache.contains_key(&key) {
    //                     //     println!("{:?}", cache.len());
    //                     //     println!("{:?}", stack.len());
    //                     //     println!("{:?}", key);
    //                     // }
    //                     states = states.union(cache.get(&key).unwrap().clone());
    //                 }
    //                 // pending.remove(&cache_key);
    //                 cache.insert(cache_key, states);
    //             }
    //         }
    //     }

    //     cache.get(&final_key).unwrap()
    // }

    fn is_green(&self, observation: &Observation) -> bool {
        // ∀ α ∈ A_O: happens_before(ω, (t, 0, α))
        let delta = self
            .observer
            .imprecisions
            .approximate_drift_delta(observation.time, self.local_time);
        let bound = delta.upper_bound - observation.base_latency
            + observation.jitter_bound
            + self.observer.imprecisions.max_latency;
        bound.into_inner() < 0.0
    }

    fn frontier(&self, prefix: &Prefix) -> HashSet<ObservationReference> {
        let mut frontier = HashSet::new();
        for observation in self
            .pending
            .iter()
            .filter(|observation| !prefix.contains(observation))
        {
            if self
                .observer
                .before(*observation)
                .iter()
                .all(|predecessor| {
                    !self.pending.contains(predecessor) || prefix.contains(predecessor)
                })
            {
                // The `observation` is a member of the frontier of the prefix `prefix`
                // if and only if the prefix already contains all observations preceding
                // the observation `observation`.
                frontier.insert(*observation);
            }
        }
        frontier
    }

    fn is_prefix(&self, set: &im::HashSet<ObservationReference>) -> bool {
        self.pending.iter().all(|observation| {
            set.iter().all(|other_observation| {
                !self.observer.imprecisions.happens_before(
                    self.observer.get(*observation),
                    self.observer.get(*other_observation),
                ) || set.contains(observation)
            })
        })
    }

    fn advance_time(&mut self, time: NotNan<f64>) {
        self.local_time = time;

        let mut work_set: IndexSet<Prefix> = self
            .prefixes
            .keys()
            .filter(|set| self.is_prefix(set))
            .cloned()
            .collect();

        // println!("initial: {:?}", work_set);

        //let mut cache = HashMap::new();

        while let Some(prefix) = work_set.pop() {
            let frontier = self.frontier(&prefix);

            if !frontier.is_empty() {
                for state in self.prefixes[&prefix].states.clone() {
                    let marked = self
                        .prefixes
                        .get_mut(&prefix)
                        .unwrap()
                        .marked
                        .entry(state.clone())
                        .or_insert_with(|| HashSet::new())
                        .clone();

                    for observation in frontier.symmetric_difference(&marked) {
                        self.prefixes
                            .get_mut(&prefix)
                            .unwrap()
                            .marked
                            .get_mut(&state)
                            .unwrap()
                            .insert(*observation);
                        let mut successor = prefix.clone();
                        successor.insert(*observation);
                        if !self.prefixes.contains_key(&successor) {
                            self.prefixes.insert(
                                successor.clone(),
                                StateSet {
                                    states: HashSet::new(),
                                    marked: HashMap::new(),
                                },
                            );
                        }
                        let mut found_successors = false;
                        for successor_state in self.explore_states(
                            &[state.clone()],
                            Some(*observation),
                            // &mut hashbrown::HashMap::new(),
                            // &mut Vec::new(),
                            //&mut cache,
                        ) {
                            self.prefixes
                                .get_mut(&successor)
                                .unwrap()
                                .states
                                .insert(successor_state.clone());
                            found_successors = true;
                        }
                        // if is_interesting {
                        //     println!("found successors: {}", found_successors);
                        // }
                        work_set.insert(successor);
                    }
                }
                let is_stable = frontier
                    .iter()
                    .all(|observation| self.is_green(self.observer.get(*observation)));
                // println!("{}", is_stable);
                if is_stable {
                    self.prefixes.remove(&prefix);
                }
            }
        }

        let mut explained = self.pending.clone();
        for (key, state_set) in self.prefixes.iter() {
            if !state_set.states.is_empty() {
                explained = explained.intersection(key.clone());
            }
        }

        self.pending = self.pending.clone().symmetric_difference(explained.clone());

        // self.states.extend(
        //     self.states
        //         .drain()
        //         .filter(|(_, states)| !states.is_empty())
        //         .map(|(key, value)| (key.symmetric_difference(explained.clone()), value)),
        // );

        let mut fresh_states = HashMap::new();
        for (key, state_set) in self.prefixes.drain() {
            if !state_set.states.is_empty() {
                let new_key = key.difference(explained.clone());
                assert!(fresh_states.insert(new_key, state_set).is_none());
            }
        }

        mem::swap(&mut self.prefixes, &mut fresh_states);
    }

    pub fn active_prefixes(&self) -> hashbrown::hash_map::Keys<Prefix, StateSet> {
        self.prefixes.keys()
    }

    pub fn push(&mut self, observation: Observation) {
        let time = observation.time.clone();
        let reference = self.observer.insort(observation);
        self.observations.insert(reference);
        self.pending.insert(reference);
        assert!(self.local_time <= time);
        self.advance_time(time);
    }
}
