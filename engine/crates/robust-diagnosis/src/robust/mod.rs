//! An implementation of *robust real-time diagnosis*.

use std::hash::Hash;
use std::rc::Rc;
use std::{mem, sync::atomic::AtomicUsize};

use hashbrown::{HashMap, HashSet};

use indexmap::IndexSet;

use clock_zones::{Constraint, Zone};

use ordered_float::NotNan;

use momba_explore::{model, time, Action, Explorer, LabeledAction, State};

pub mod generate;
pub mod observer;

use observer::{Observation, ObservationIndex};

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct HistoryItem {
    observation: ObservationIndex,
    tracking_clock: clock_zones::Variable,
}

/// A state used by a diagnoser.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DiagnosisState {
    pub state: Rc<State<time::Float64Zone>>,
    pub expected: Option<ObservationIndex>,
    pub faults: im::HashSet<LabeledAction>,
    pub history: im::Vector<HistoryItem>,
}

impl DiagnosisState {
    fn with_expected(&self, expected: Option<ObservationIndex>) -> Self {
        Self {
            state: self.state.clone(),
            expected,
            faults: self.faults.clone(),
            history: self.history.clone(),
        }
    }
}

/// A prefix is a set of observation references.
type Prefix = im::HashSet<ObservationIndex>;

#[derive(Clone, Debug)]
pub struct StateSet {
    pub(crate) states: im::HashSet<Rc<DiagnosisState>>,
    marked: HashMap<Rc<DiagnosisState>, HashSet<ObservationIndex>>,
}

/// A *diagnoser* providing a diagnosis from observations.
pub struct Diagnoser {
    pub(crate) observer: observer::Observer,
    pub(crate) explorer: momba_explore::Explorer<time::Float64Zone>,

    pub(crate) explore_counter: AtomicUsize,

    observations: im::HashSet<ObservationIndex>,
    local_time: NotNan<f64>,

    pub(crate) prefixes: HashMap<Prefix, StateSet>,

    observable_indices: HashSet<usize>,
    fault_indices: HashSet<usize>,

    //marked: HashMap<Rc<DiagnosisState>, HashSet<usize>>,

    // Pending observations
    pending: im::HashSet<ObservationIndex>,

    /// For how many observations should we apply the precise timing check?
    ///
    /// [None] means that we keep an unbounded history in states.
    history_bound: Option<usize>,
}

enum StackItem {
    Explore(Rc<DiagnosisState>),
    Populate((Rc<DiagnosisState>, Vec<Rc<DiagnosisState>>)),
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct DiagnosisResult {
    pub consistent: bool,
    pub failure_possible: bool,
    pub failure_necessary: bool,
    pub possible_failures: HashSet<LabeledAction>,
    pub necessary_failures: HashSet<LabeledAction>,
    pub states: usize,
    pub prefixes: usize,
}

impl Diagnoser {
    pub fn new(
        imprecisions: observer::Imprecisions,
        network: model::Network,
        observable_indices: HashSet<usize>,
        fault_indices: HashSet<usize>,
        history_bound: Option<usize>,
    ) -> Self {
        let mut diagnoser = Diagnoser {
            observer: observer::Observer::new(imprecisions),
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
                states: diagnoser
                    .explorer
                    .initial_states()
                    .into_iter()
                    .map(|initial_state| {
                        diagnoser.explore_states(Rc::new(DiagnosisState {
                            state: Rc::new(initial_state),
                            expected: None,
                            faults: im::HashSet::new(),
                            history: im::Vector::new(),
                        }))
                    })
                    .fold(im::HashSet::new(), |states, element| states.union(element)),
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
            .filter(|observation| self.is_settled(self.observer.get(**observation)))
            .cloned()
            .collect();

        for (prefix, state_set) in self.prefixes.iter() {
            state_counter += state_set.states.len();

            if !green_observations.is_subset(prefix) {
                continue;
            }

            for state in state_set.states.iter() {
                // println!("{:?}", state.failures);
                possible_failures.extend(state.faults.iter().cloned());
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

    fn explore_states(&self, state: Rc<DiagnosisState>) -> im::HashSet<Rc<DiagnosisState>> {
        let mut stack: Vec<_> = vec![state.clone()];
        let mut visited: HashSet<_> = stack.iter().cloned().collect();

        let mut successors = im::HashSet::new();

        // let mut processed = 0;

        while let Some(state) = stack.pop() {
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

            let mut is_interaction_state = false;

            for mut transition in transitions {
                let action = transition.result_action();
                let is_observable = action
                    .label_index()
                    .map(|index| self.observable_indices.contains(&index))
                    .unwrap_or(false);
                let is_expected = state
                    .expected
                    .map(|expected| match &action {
                        Action::Labeled(labeled) => self.observer.get(expected).action == *labeled,
                        _ => false,
                    })
                    .unwrap_or(false);

                if is_observable {
                    if state.expected.is_none() {
                        is_interaction_state = true;
                    }
                    if !is_expected {
                        continue;
                    }
                }

                let mut faults = state.faults.clone();
                match action {
                    Action::Labeled(labeled) => {
                        if self.fault_indices.contains(&labeled.label_index()) {
                            faults.insert(labeled.clone());
                        }
                    }
                    _ => {}
                }

                let mut history = state.history.clone();
                if is_expected && self.history_bound != Some(0) {
                    let (tracking_clock, mut valuations) = match self.history_bound {
                        None => {
                            let valuations = transition.valuations();
                            (
                                clock_zones::Clock::variable(valuations.num_variables()),
                                valuations.resize(valuations.num_variables() + 1),
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
                                    valuations.resize(valuations.num_variables() + 1),
                                )
                            }
                        }
                    };

                    valuations.reset(tracking_clock, NotNan::new(0.0).unwrap());
                    // println!("reset tracking clock, {}", history.len());
                    // println!("{:?}", valuations);
                    let observation = state.expected.unwrap();

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
                        let successor = Rc::new(DiagnosisState {
                            state: Rc::new(self.explorer.successor(
                                &state.state,
                                &transition,
                                &destination,
                            )),
                            expected: if is_expected { None } else { state.expected },
                            faults: faults.clone(),
                            history: history.clone(),
                        });

                        if visited.insert(successor.clone()) {
                            stack.push(successor);
                        }
                    }
                }
            }

            if is_interaction_state {
                successors.insert(state);
            }
        }

        successors
    }

    /// Checks whether the given observation is settled.
    pub fn is_settled(&self, observation: &Observation) -> bool {
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

    fn frontier(&self, prefix: &Prefix) -> HashSet<ObservationIndex> {
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

    fn is_prefix(&self, set: &im::HashSet<ObservationIndex>) -> bool {
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
                                    states: im::HashSet::new(),
                                    marked: HashMap::new(),
                                },
                            );
                        }
                        let mut found_successors = false;
                        for successor_state in self.explore_states(
                            Rc::new(state.with_expected(Some(*observation))),
                            // Some(*observation),
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
                    .all(|observation| self.is_settled(self.observer.get(*observation)));
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