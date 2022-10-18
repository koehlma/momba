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

/// A *history item* with an observation and a *tracking clock*.
///
/// The *tracking clock* tracks the time since the event corresponding to
/// the observation has happened.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct HistoryItem {
    pub observation: ObservationIndex,
    pub tracking_clock: clock_zones::Variable,
}

/// A *diagnosis state* used by a [Diagnoser].
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct DiagnosisState {
    /// The abstract system state.
    pub state: Rc<State<time::Float64Zone>>,
    /// The set of occurred faults.
    pub faults: im::HashSet<LabeledAction>,
    /// An observation that is expected next (necessary for depth-first search).
    pub expected: Option<ObservationIndex>,
    /// The history (necessary for implementing the history bound.)
    pub history: im::Vector<HistoryItem>,
}

impl DiagnosisState {
    fn with_expected(&self, expected: Option<ObservationIndex>) -> Self {
        Self {
            state: self.state.clone(),
            faults: self.faults.clone(),
            expected,
            history: self.history.clone(),
        }
    }
}

/// A prefix is a set of observation references.
type Prefix = im::HashSet<ObservationIndex>;

#[derive(Clone, Debug)]
pub struct StateSet {
    marked: HashMap<Rc<DiagnosisState>, HashSet<ObservationIndex>>,
}

/// A *diagnoser* providing a diagnosis from observations.
pub struct Diagnoser {
    pub(crate) observer: observer::Observer,
    pub(crate) explorer: momba_explore::Explorer<time::Float64Zone>,

    pub(crate) explore_counter: AtomicUsize,

    observations: im::HashSet<ObservationIndex>,
    time: NotNan<f64>,

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

/// The output produced by the diagnoser.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct DiagnosisResult {
    /// Indicates whether the observations are consistent with some run of the model.
    pub consistent: bool,
    /// Indicates whether a fault possibly occurred.
    pub fault_possible: bool,
    /// Indicates whether a fault necessarily occurred.
    pub fault_necessary: bool,
    /// The set of possible faults.
    pub possible_faults: HashSet<LabeledAction>,
    /// The set of necessary faults.
    pub necessary_faults: HashSet<LabeledAction>,
    /// The number of diagnosis states (for experiments).
    pub states: usize,
    /// The number of active prefixes (for experiments).
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
            time: NotNan::new(0.0).unwrap(),

            prefixes: HashMap::new(),

            observable_indices,
            fault_indices,

            pending: im::HashSet::new(),

            history_bound,
        };

        // Create an initially empty prefix and invokes the exploration procedure
        // to establish the invariants (as described in the paper).
        diagnoser.prefixes.insert(
            im::HashSet::new(),
            StateSet {
                marked: diagnoser
                    .explore_states(
                        &diagnoser
                            .explorer
                            .initial_states()
                            .into_iter()
                            .map(|initial_state| {
                                Rc::new(DiagnosisState {
                                    state: Rc::new(initial_state),
                                    expected: None,
                                    faults: im::HashSet::new(),
                                    history: im::Vector::new(),
                                })
                            })
                            .collect::<Box<[_]>>(),
                    )
                    .into_iter()
                    .map(|state| (state, HashSet::new()))
                    .collect(),
                //marked: HashMap::new(),
            },
        );

        diagnoser
    }

    /// Extracts a [`DiagnosisResult`] from the diagnoser's state.
    pub fn result(&self) -> DiagnosisResult {
        let mut possible_faults = HashSet::new();
        let mut necessary_faults = None;

        let mut fault_necessary = true;

        let mut state_counter = 0;

        let settled_observations: im::HashSet<_> = self
            .pending
            .iter()
            .filter(|observation| self.is_settled(self.observer.get(**observation)))
            .cloned()
            .collect();

        // This iterates over all prefixes in order to decide which faults may and must have occurred.
        for (prefix, state_set) in self.prefixes.iter() {
            state_counter += state_set.marked.len();

            // We only take the active prefixes into account.
            if !settled_observations.is_subset(prefix) {
                continue;
            }

            for state in state_set.marked.keys() {
                // Extend the set of possible faults with the occurred faults from the diagnosis state.
                possible_faults.extend(state.faults.iter().cloned());
                // If the set is empty, then there are runs without faults. Hence, no fault is necessary.
                if state.faults.is_empty() {
                    fault_necessary = false;
                }
                // Intersect the set of necessary faults with the set of occurred faults from this state.
                match necessary_faults {
                    None => necessary_faults = Some(state.faults.clone()),
                    Some(failures) => {
                        necessary_faults = Some(failures.intersection(state.faults.clone()))
                    }
                }
            }
        }

        DiagnosisResult {
            consistent: !self.prefixes.is_empty(),
            fault_possible: !possible_faults.is_empty(),
            fault_necessary,
            possible_faults,
            necessary_faults: necessary_faults
                .map_or_else(|| HashSet::new(), |failures| failures.into_iter().collect()),
            states: state_counter,
            prefixes: self.prefixes.len(),
        }
    }

    /// A depth-first search variant of the exploration procedure.
    fn explore_states(&self, states: &[Rc<DiagnosisState>]) -> HashSet<Rc<DiagnosisState>> {
        // Stack and visited states for depth first search.
        let mut stack: Vec<_> = states.iter().cloned().collect();
        let mut visited: HashSet<_> = stack.iter().cloned().collect();

        // The set of successors `S` to be returned by the procedure.
        let mut successors = HashSet::new();

        while let Some(state) = stack.pop() {
            // Gets the transitions of the system model in the abstract system state.
            let transitions = self.explorer.transitions(&state.state);

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
            .approximate_drift_delta(observation.time, self.time);
        let bound = delta.upper_bound - observation.base_latency
            + observation.jitter_bound
            + self.observer.imprecisions.max_latency;
        bound.into_inner() < 0.0
    }

    /// Computes the frontier as described in the paper.
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

    /// Checks whether the given set of observations is a prefix.
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

    /// Advances the time of the diagnoser to the provided time.
    ///
    /// Applies the exploration procedure and maintains the invariants as described in the paper.
    fn advance_time(&mut self, time: NotNan<f64>) {
        self.time = time;

        // First, reduce the prefixes to re-establish invariant (A).
        let mut work_set: IndexSet<Prefix> = self
            .prefixes
            .keys()
            .filter(|set| self.is_prefix(set))
            .cloned()
            .collect();

        // We need to extend all states for which the prefix does contain not yet marked observations
        // in its frontier. This is what the `work_set` is for and what we do here.
        while let Some(prefix) = work_set.pop() {
            let frontier = self.frontier(&prefix);
            if !frontier.is_empty() {
                for observation in frontier.iter() {
                    let states = self
                        .prefixes
                        .get_mut(&prefix)
                        .unwrap()
                        .marked
                        .iter_mut()
                        .filter_map(|(state, marked)| {
                            if marked.insert(*observation) {
                                Some(Rc::new(state.with_expected(Some(*observation))))
                            } else {
                                None
                            }
                        })
                        .collect::<Box<[Rc<DiagnosisState>]>>();
                    if !states.is_empty() {
                        let mut successor = prefix.clone();
                        successor.insert(*observation);

                        if !self.prefixes.contains_key(&successor) {
                            self.prefixes.insert(
                                successor.clone(),
                                StateSet {
                                    marked: HashMap::new(),
                                },
                            );
                        }

                        // Explore the successor states.
                        for successor_state in self.explore_states(&states) {
                            let state_set = self.prefixes.get_mut(&successor).unwrap();
                            state_set
                                .marked
                                .entry(successor_state)
                                .or_insert_with(|| HashSet::new());
                        }
                        work_set.insert(successor);
                    }
                }

                // This removes a prefix if all its observations are settled.
                let is_stable = frontier
                    .iter()
                    .all(|observation| self.is_settled(self.observer.get(*observation)));
                if is_stable {
                    self.prefixes.remove(&prefix);
                }
            }
        }

        let mut explained = self.pending.clone();
        for (key, state_set) in self.prefixes.iter() {
            if !state_set.marked.is_empty() {
                explained = explained.intersection(key.clone());
            }
        }

        self.pending = self.pending.clone().symmetric_difference(explained.clone());

        let mut fresh_states = HashMap::new();
        for (key, state_set) in self.prefixes.drain() {
            if !state_set.marked.is_empty() {
                let new_key = key.difference(explained.clone());
                assert!(fresh_states.insert(new_key, state_set).is_none());
            }
        }

        mem::swap(&mut self.prefixes, &mut fresh_states);
    }

    /// Returns the active prefixes.
    pub fn active_prefixes(&self) -> hashbrown::hash_map::Keys<Prefix, StateSet> {
        self.prefixes.keys()
    }

    /// Pushes a new [Observation] into the diagnoser.
    pub fn push(&mut self, observation: Observation) {
        let time = observation.time.clone();
        let index = self.observer.insort(observation);
        self.observations.insert(index);
        self.pending.insert(index);
        assert!(self.time <= time);
        self.advance_time(time);
    }
}
