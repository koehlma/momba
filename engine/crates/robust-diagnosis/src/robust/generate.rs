//! Algorithms and data structures to generate traces of observations.

use ordered_float::NotNan;
use rand::seq::IteratorRandom;
use rand::Rng;
use std::ops::RangeInclusive;

use momba_explore as explore;
use momba_explore::model;
use momba_explore::time;

use crate::external;

use super::*;

pub struct GeneratorState {
    system_state: explore::State<time::Float64Zone>,
    observations: usize,
    time: NotNan<f64>,
    events: Vec<external::TimedEvent>,
}

pub trait Oracle {
    fn choose_transition<'c>(
        &mut self,
        state: &GeneratorState,
        transitions: Vec<explore::Transition<'c, time::Float64Zone>>,
    ) -> explore::Transition<'c, time::Float64Zone>;

    fn choose_time(
        &mut self,
        state: &GeneratorState,
        transition: &explore::Transition<time::Float64Zone>,
        range: RangeInclusive<NotNan<f64>>,
    ) -> NotNan<f64>;
}

pub enum InjectionTiming {
    AfterObservations(usize),
    AfterTime(NotNan<f64>),
}

pub struct Inject {
    pub(crate) label_index: usize,
    pub(crate) timing: InjectionTiming,
    // fault_indices: HashSet<usize>,
    // fault_injected: bool,
}

// impl Inject {
//     pub fn new(label_index: usize, timing: InjectionTiming, fault_indices: HashSet<usize>) -> Self {
//         Inject {
//             label_index,
//             timing,
//             fault_indices,
//             fault_injected: false,
//         }
//     }
// }

// impl Oracle for Inject {
//     fn choose_transition<'c>(
//         &mut self,
//         state: &GeneratorState,
//         transitions: Vec<explore::Transition<'c, time::GlobalTime>>,
//     ) -> explore::Transition<'c, time::GlobalTime> {
//         let contain_fault = transitions
//             .iter()
//             .any(|transition| match transition.result_action() {
//                 momba_explore::Action::Labeled(labeled) => {
//                     labeled.label_index() == self.label_index
//                 }
//                 _ => false,
//             });
//         let should_inject = contain_fault
//             && !self.fault_injected
//             && match self.timing {
//                 InjectionTiming::AfterObservations(after_observations) => {
//                     state.observations >= after_observations
//                 }
//                 InjectionTiming::AfterTime(after_time) => state.time >= after_time,
//             };

//         transitions
//             .into_iter()
//             .filter(|transition| match transition.result_action() {
//                 momba_explore::Action::Labeled(labeled) => {
//                     if self.fault_indices.contains(&labeled.label_index()) {
//                         should_inject && labeled.label_index() == self.label_index
//                     } else {
//                         !should_inject
//                     }
//                 }
//                 _ => !should_inject,
//             })
//             .choose(&mut rand::thread_rng())
//             .unwrap()
//     }

//     fn choose_time(
//         &mut self,
//         state: &GeneratorState,
//         transition: &explore::Transition<time::GlobalTime>,
//         range: RangeInclusive<NotNan<f64>>,
//     ) -> NotNan<f64> {
//         ordered_float::NotNan::new(
//             rand::thread_rng().gen_range(range.start().into_inner()..=range.end().into_inner()),
//         )
//         .unwrap()
//     }
// }

#[derive(Clone, Debug)]
pub struct Timing {
    pub(crate) base_latency: NotNan<f64>,
    pub(crate) jitter_bound: NotNan<f64>,
}

impl Timing {
    pub fn sample_latency(&self) -> NotNan<f64> {
        self.base_latency
            + rand::thread_rng()
                .gen_range(-self.jitter_bound.into_inner()..=self.jitter_bound.into_inner())
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct HybridTime {
    pub(crate) discrete: usize,
    pub(crate) continuous: NotNan<f64>,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Event {
    pub(crate) time: HybridTime,
    pub(crate) action: momba_explore::LabeledAction,
}

#[derive(Clone, Debug)]
pub struct ObservableEvent {
    pub(crate) time: HybridTime,
    pub(crate) latency: NotNan<f64>,
    pub(crate) action: momba_explore::LabeledAction,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub(crate) time: NotNan<f64>,
    pub(crate) event: ObservableEvent,
}

#[derive(Clone, Debug)]
pub struct GeneratorResult {
    pub(crate) events: Vec<Event>,
    pub(crate) observations: Vec<Observation>,
}

/// A generator for observation traces.
pub struct Generator {
    pub(crate) explorer: momba_explore::Explorer<time::Float64Zone>,
    global_clock: clock_zones::Variable,
    inject: Inject,
    observable_indices: HashSet<usize>,
    fault_indices: HashSet<usize>,
    imprecisions: super::observer::Imprecisions,
    timing: HashMap<usize, Timing>,
}

impl Generator {
    pub fn new(
        mut network: model::Network,
        inject: Inject,
        observable_indices: HashSet<usize>,
        fault_indices: HashSet<usize>,
        imprecisions: super::observer::Imprecisions,
        timing: HashMap<usize, Timing>,
    ) -> Self {
        network
            .declarations
            .clock_variables
            .insert("__global_clock".to_owned());
        let global_clock = clock_zones::Clock::variable(
            network
                .declarations
                .clock_variables
                .get_index_of("__global_clock")
                .unwrap(),
        );
        Generator {
            explorer: momba_explore::Explorer::new(network),
            global_clock,
            inject,
            observable_indices,
            fault_indices,
            imprecisions,
            timing,
        }
    }

    pub fn generate2(&self, observations: usize, oracle: &mut dyn Oracle) -> Vec<Observation> {
        let mut rng = rand::thread_rng();
        let mut state = GeneratorState {
            system_state: self
                .explorer
                .initial_states()
                .into_iter()
                .choose(&mut rng)
                .unwrap(),
            events: Vec::new(),
            observations: 0,
            time: NotNan::new(0.0).unwrap(),
        };

        while state.observations < observations {
            let choice =
                oracle.choose_transition(&state, self.explorer.transitions(&state.system_state));

            let time_lower_bound = choice
                .valuations()
                .get_lower_bound(self.global_clock)
                .unwrap();
            let time_upper_bound = choice
                .valuations()
                .get_upper_bound(self.global_clock)
                .unwrap();

            state.time = oracle.choose_time(&state, &choice, time_lower_bound..=time_upper_bound);

            let mut valuations = choice.valuations().clone();
            valuations.add_constraint(clock_zones::Constraint::new_le(
                self.global_clock,
                state.time,
            ));
            valuations.add_constraint(clock_zones::Constraint::new_ge(
                self.global_clock,
                state.time,
            ));

            let choice = choice.replace_valuations(valuations);
        }

        Vec::new()
    }

    pub fn generate(&self, simulation_time: NotNan<f64>) -> GeneratorResult {
        let mut rng = rand::thread_rng();
        let mut state = self
            .explorer
            .initial_states()
            .into_iter()
            .choose(&mut rng)
            .unwrap();
        let mut observable_events = Vec::new();
        let mut events = Vec::new();

        let mut observations = 0;
        let mut fault_injected = false;

        let mut discrete_time = 0;
        let mut continuous_time = NotNan::new(0.0).unwrap();

        while continuous_time < simulation_time {
            let transitions = self.explorer.transitions(&state);
            let contain_fault =
                transitions
                    .iter()
                    .any(|transition| match transition.result_action() {
                        momba_explore::Action::Labeled(labeled) => {
                            labeled.label_index() == self.inject.label_index
                        }
                        _ => false,
                    });
            let should_inject = contain_fault
                && !fault_injected
                && match self.inject.timing {
                    InjectionTiming::AfterObservations(after_observations) => {
                        observations >= after_observations
                    }
                    InjectionTiming::AfterTime(after_time) => continuous_time >= after_time,
                };

            fault_injected |= should_inject;

            let mut transition = self
                .explorer
                .transitions(&state)
                .into_iter()
                .filter(|transition| match transition.result_action() {
                    momba_explore::Action::Labeled(labeled) => {
                        if self.fault_indices.contains(&labeled.label_index()) {
                            should_inject && labeled.label_index() == self.inject.label_index
                        } else {
                            !should_inject
                        }
                    }
                    _ => !should_inject,
                })
                .choose(&mut rng)
                .unwrap();
            // the absolute time the transition happened
            let lower_bound = transition
                .valuations()
                .get_lower_bound(self.global_clock)
                .unwrap()
                .into_inner();
            let upper_bound = transition
                .valuations()
                .get_upper_bound(self.global_clock)
                .unwrap()
                .into_inner();
            continuous_time =
                ordered_float::NotNan::new(rng.gen_range(lower_bound..=upper_bound)).unwrap();
            // println!(
            //     "[{:?}, {:?}] {:?} {:?} {:?}",
            //     lower_bound,
            //     upper_bound,
            //     transition
            //         .valuations()
            //         .get_bound(self.global_clock, clock_zones::Clock::ZERO),
            //     transition
            //         .valuations()
            //         .get_bound(clock_zones::Clock::ZERO, self.global_clock),
            //     continuous_time
            // );
            let mut valuations = transition.valuations().clone();
            debug_assert!(!valuations.is_empty());
            valuations.add_constraint(clock_zones::Constraint::new_le(
                self.global_clock,
                continuous_time + 1e-10,
            ));
            // println!(
            //     "{:?} {:?} {:?}",
            //     valuations.get_bound(self.global_clock, clock_zones::Clock::ZERO),
            //     valuations.get_bound(clock_zones::Clock::ZERO, self.global_clock),
            //     valuations.get_bound(clock_zones::Clock::ZERO, clock_zones::Clock::ZERO),
            // );
            debug_assert!(!valuations.is_empty());
            valuations.add_constraint(clock_zones::Constraint::new_ge(
                self.global_clock,
                continuous_time - 1e-10,
            ));
            // println!(
            //     "{:?} {:?} {:?}",
            //     valuations.get_bound(self.global_clock, clock_zones::Clock::ZERO),
            //     valuations.get_bound(clock_zones::Clock::ZERO, self.global_clock),
            //     valuations.get_bound(clock_zones::Clock::ZERO, clock_zones::Clock::ZERO),
            // );
            debug_assert!(!valuations.is_empty());

            discrete_time += 1;

            transition = transition.replace_valuations(valuations);

            //println!("t = {:?}", t);
            match transition.result_action() {
                //momba_explore::Action::Silent => println!("τ"),
                momba_explore::Action::Labeled(labeled) => {
                    events.push(Event {
                        time: HybridTime {
                            discrete: discrete_time,
                            continuous: continuous_time,
                        },
                        action: labeled.clone(),
                    });

                    if self.observable_indices.contains(&labeled.label_index()) {
                        let latency = self
                            .timing
                            .get(&labeled.label_index())
                            .unwrap()
                            .sample_latency();
                        let event = ObservableEvent {
                            time: HybridTime {
                                discrete: discrete_time,
                                continuous: continuous_time,
                            },
                            latency,
                            action: labeled.clone(),
                        };
                        observable_events.push(event);
                        observations += 1;
                    };
                    // println!(
                    //     "{} {:?} (t={})",
                    //     labeled.label(&self.explorer.network).unwrap(),
                    //     labeled.arguments(),
                    //     continuous_time
                    // );
                }
                _ => (),
            }

            let destinations = self.explorer.destinations(&state, &transition);

            let threshold: f64 = rng.gen();
            let mut accumulated = 0.0;

            for destination in destinations {
                accumulated += destination.probability();
                if accumulated >= threshold {
                    state = self.explorer.successor(&state, &transition, &destination);
                    break;
                }
            }
        }

        observable_events.sort_by_cached_key(|event| event.time.continuous + event.latency);

        let mut observations = Vec::new();
        let mut observer_time = NotNan::new(42.1314).unwrap();
        let mut last_event_time = NotNan::new(0.0).unwrap();

        for event in observable_events {
            let event_time = event.time.continuous + event.latency;
            let delta = event_time - last_event_time;
            observer_time += delta
                * rng.gen_range(
                    self.imprecisions.min_drift_slope.into_inner()
                        ..=self.imprecisions.max_drift_slope.into_inner(),
                );
            // println!("{} {} {} {:?}", observer_time, event_time, delta, event);
            observations.push(Observation {
                time: observer_time,
                event,
            });
            last_event_time = event_time;
        }

        GeneratorResult {
            observations,
            events,
        }
    }
}
