//! Convenience functions for simulating random runs.

use rand::seq::IteratorRandom;
use rand::Rng;

use super::*;

/// Represents an *oracle* used to choose a transition.
///
/// When faced with a non-deterministic chose between multiple transitions, the simulator
/// uses an oracle to determine which transition to choose.
pub trait Oracle<T: time::Time> {
    /// Chooses a transition from a vector of transitions.
    fn choose<'e, 't>(
        &self,
        state: &State<T>,
        transitions: &'t Vec<DetachedTransition<'e, T>>,
    ) -> &'t DetachedTransition<'e, T>;
}

/// An [Oracle] choosing a transition uniformly at random.
pub struct UniformOracle {}

impl UniformOracle {
    /// Constructs a new uniform oracle.
    pub fn new() -> Self {
        UniformOracle {}
    }
}

impl<T: time::Time> Oracle<T> for UniformOracle {
    fn choose<'e, 't>(
        &self,
        _state: &State<T>,
        transitions: &'t Vec<DetachedTransition<'e, T>>,
    ) -> &'t DetachedTransition<'e, T> {
        let mut rng = rand::thread_rng();
        transitions.iter().choose(&mut rng).unwrap()
    }
}

/// Injects certain transitions after a specified delay.
///
/// It is often useful to be able to enforce specific transitions based on
/// their action after a certain amount of time.
/// This oracle allows injecting specific transitions after a certain
/// amount of transitions.
pub struct InjectionOracle {}

/// A simulator used to simulate random runs.
pub struct Simulator<O: Oracle<T>, T: time::Time> {
    pub(crate) oracle: O,

    _phontom_time_type: std::marker::PhantomData<T>,
}

impl<O: Oracle<T>, T: time::Time> Simulator<O, T> {
    /// Creates a new simulator with the given oracle.
    pub fn new(oracle: O) -> Self {
        Simulator {
            oracle,
            _phontom_time_type: std::marker::PhantomData,
        }
    }

    /// Returns the oracle used by the simulator.
    pub fn oracle(&self) -> &O {
        &self.oracle
    }

    /// Simulates a random run.
    pub fn simulate(&self, explorer: &Explorer<T>, steps: usize) {
        let mut rng = rand::thread_rng();
        let mut state = explorer
            .initial_states()
            .into_iter()
            .choose(&mut rng)
            .unwrap();

        for _ in 0..steps {
            let transition = explorer
                .transitions(&state)
                .into_iter()
                .choose(&mut rng)
                .unwrap();

            match transition.result_action() {
                Action::Silent => println!("τ"),
                Action::Labeled(labeled) => println!(
                    "{} {:?}",
                    labeled.label(&explorer.network).unwrap(),
                    labeled.arguments()
                ),
            }

            let destinations = explorer.destinations(&state, &transition);

            let threshold: f64 = rng.gen();
            let mut accumulated = 0.0;

            for destination in destinations {
                accumulated += destination.probability();
                if accumulated >= threshold {
                    state = explorer.successor(&state, &transition, &destination);
                    break;
                }
            }
        }
    }
}
