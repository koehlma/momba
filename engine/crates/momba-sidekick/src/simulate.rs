use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{atomic, Arc},
};

use momba_explore::{model::Value, *};
use rand::{rngs::StdRng, seq::IteratorRandom, Rng};
use rayon::{current_num_threads, prelude::*};

/// Represents the Output of the SPRT simulations, saying if we are above
/// or below of the provided *x* value
#[derive(Debug)]
pub enum SprtComparison {
    BiggerThan(i64),
    LesserThan(i64),
}
/// Output of each Simulation.
#[derive(Debug, PartialEq)]
pub enum SimulationOutput {
    /// The simulation reached a State where the goal predicate its satisfied
    GoalReached(usize),
    /// The simulation has made the max amount of steps without reaching an absorbent or a goal state.
    MaxSteps,
    /// We have reached an absorbent state and finished the simulation.
    NoStatesAvailable,
}

/// Represents and oracle that tells us how to resolve undeterminations.
pub trait Oracle<T: time::Time> {
    /// Chooses a transition between the provided ones, and can have access to all
    /// the information of the current state.
    /// Precondition: transitions slice is not empty.
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T>;
}
/// Oracle that resolves no-determinism uniformly
#[derive(Clone)]
pub struct UniformOracle {
    rng: RefCell<StdRng>,
}

impl UniformOracle {
    pub fn new(rng: StdRng) -> Self {
        UniformOracle {
            rng: RefCell::new(rng),
        }
    }
}

impl<T: time::Time> Oracle<T> for UniformOracle {
    fn choose<'s, 't>(
        &self, //&mut self
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        let elected_transition = transitions
            .into_iter()
            .choose(&mut *(self.rng.borrow_mut()))
            .unwrap();
        elected_transition
    }
}

/// Oracle that resolves no-determinism uniformly
#[derive(Clone)]
pub struct FIFOOracle {}

impl FIFOOracle {
    pub fn _new() -> Self {
        FIFOOracle {}
    }
}

impl<T: time::Time> Oracle<T> for FIFOOracle {
    fn choose<'s, 't>(
        &self, //&mut self
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        let elected_transition = transitions.first().unwrap();
        elected_transition
    }
}

/// Represents a Simulator
pub trait Simulator {
    /// the type of the states that the simulator will deal with.
    type State<'sim>
    where
        Self: 'sim;
    /// Gives the next state using a provided oracle.
    fn next(&mut self) -> Option<Self::State<'_>>;
    /// Resets the simulator.
    fn reset(&mut self) -> Self::State<'_>;
    /// Returns the current state that the simulation is in.
    fn current_state(&mut self) -> Self::State<'_>;
}

/// Struct that will iterate over the states.
#[derive(Clone)]
pub struct StateIter<T: time::Time, O: Oracle<T>> {
    /// The state hes in.
    pub state: State<T>,
    /// An explorer that will allow the Iterator interact with the model.
    explorer: Arc<Explorer<T>>,
    /// An oracle that will resolve no-determinism.
    oracle: O,
    /// RNG for the iterator.
    rng: RefCell<StdRng>,
}

/// Implementation of the State Iterator.
impl<T: time::Time, O: Oracle<T>> StateIter<T, O> {
    pub fn new(explorer: Arc<Explorer<T>>, oracle: O, rng: StdRng) -> Self {
        StateIter {
            state: explorer
                .initial_states()
                .into_iter()
                .choose(&mut rng.clone())
                .unwrap(),
            explorer,
            oracle,
            rng: RefCell::new(rng),
        }
    }
    /// WARNING:
    /// Only use on small models.
    /// Outputs a vector with the representation of the global variables on
    /// each state.
    fn _generate_trace(&mut self, steps: usize) -> Vec<HashMap<String, Value>> {
        self.reset();
        let mut c = 0;
        let mut visited: Vec<HashMap<String, Value>> = vec![];
        let mut state_rep: HashMap<String, Value> = HashMap::new();
        for (id, _) in &self.explorer.network.declarations.global_variables {
            let val = self.state.get_global_value(&self.explorer, id).unwrap();
            state_rep.insert(id.clone(), val.clone());
        }
        visited.push(state_rep);
        while c < steps {
            match self.next() {
                None => {
                    return visited;
                }
                Some(_) => {
                    let mut state_rep: HashMap<String, Value> = HashMap::new();
                    for (id, _) in &self.explorer.network.declarations.global_variables {
                        let val = self.state.get_global_value(&self.explorer, id).unwrap();
                        state_rep.insert(id.clone(), val.clone());
                    }
                    visited.push(state_rep);
                    c += 1;
                }
            };
        }
        visited
    }
}

/// Implementation of the Simulator trait for the iterator.
/// Panics if there aren't destination on the choosed transition.
impl<T: time::Time, O: Oracle<T>> Simulator for StateIter<T, O> {
    type State<'sim> = &'sim State<T> where Self:'sim;

    fn next(&mut self) -> Option<Self::State<'_>> {
        let mut rng = self.rng.borrow_mut();

        // for (id, _) in &self.explorer.network.declarations.global_variables {
        //     println!(
        //         "ID: {} - Value: {:?}",
        //         id,
        //         self.state.get_global_value(&self.explorer, &id).unwrap()
        //     );
        // }
        // println!("---------------");

        // for (id, _) in &self.explorer.network.declarations.transient_variables {
        //     let value = self
        //         .state
        //         .get_transient_value(&self.explorer.network, &id)
        //         .unwrap_vector();
        //     println!(
        //         "ID: {} - Value: {:?}. {:?}",
        //         id,
        //         value,
        //         value[0].unwrap_vector().len()
        //     );
        // }
        // println!("---------------");

        let transitions = self.explorer.transitions(&self.state);
        if transitions.is_empty() {
            return None;
        }

        let transition = self.oracle.choose(&self.state, &transitions);

        let destinations = self.explorer.destinations(&self.state, transition);

        if destinations.is_empty() {
            panic!("There are no destinations, something is wrong...");
        } else if destinations.len() == 1 {
            let destination = destinations.first().unwrap();
            self.state = self
                .explorer
                .successor(&self.state, transition, &destination);
        } else {
            let threshold: f64 = rng.gen();
            let mut accumulated = 0.0;
            for destination in destinations {
                accumulated += destination.probability();

                if accumulated >= threshold {
                    self.state = self
                        .explorer
                        .successor(&self.state, transition, &destination);
                    break;
                }
            }
        }
        Some(&self.state)
    }

    fn reset(&mut self) -> Self::State<'_> {
        let mut rng = self.rng.borrow_mut();
        self.state = self
            .explorer
            .initial_states()
            .into_iter()
            .choose(&mut *rng)
            .unwrap();
        &self.state
    }

    fn current_state(&mut self) -> Self::State<'_> {
        &self.state
    }
}

/// Struct that provides different types of simulation based on a
/// Simulator and a Goal predicate.
#[must_use]
pub struct StatisticalSimulator<'sim, S, G> {
    /// The Simulator to use.
    sim: &'sim mut S,
    /// The Goal function.
    goal: G,
    /// *ϵ* is the error used in the Okamoto bound for establishing the amount of runs.
    /// Default value: 0.01
    eps: f64,
    /// δ is the level of confidence provided.
    /// δ = 0.02 translates to a 98% in the confidence level.
    /// Default value: 0.05.
    delta: f64,
    /// The amount of steps for the simulations.
    /// Default value: 10000
    max_steps: usize,
    /// Fixed approximation provided for the usage of the SPRT algorithm.
    /// *P(eventually G)~x*
    /// Default value: 0
    x: f64,
    /// Type I error in SPRT algorithm.
    /// Default value: 1
    alpha: f64,
    /// Type II error in SPRT algorithm.
    /// Default value: II
    beta: f64,
    /// Indiference region used in the SPRT algorithm.
    /// Default value: 0
    ind_reg: f64,
    /// Number of threads to use when simulating with parallel implementations.
    /// Default value: 1
    n_threads: usize,
}

/// Implementation of the struct.
impl<'sim, S, G> StatisticalSimulator<'sim, S, G>
where
    S: Simulator,
    G: Fn(&S::State<'_>) -> bool,
{
    /// Create a new Statiscal Simulator.
    pub fn new(sim: &'sim mut S, goal: G) -> Self {
        Self {
            sim,
            goal,
            eps: 0.01,
            delta: 0.05,
            max_steps: 5000,
            x: 0.0,
            alpha: 1.0,
            beta: 1.0,
            ind_reg: 0.0,
            n_threads: 1,
        }
    }

    /// set field: steps
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// set field: eps
    pub fn _with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// set field: delta
    pub fn _with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// set field: x
    pub fn _with_x(mut self, x: f64) -> Self {
        self.x = x;
        self
    }

    /// set field: alpha
    pub fn _with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// set field: beta
    pub fn _with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// set field: ind_reg
    pub fn _with_ind_reg(mut self, ind_reg: f64) -> Self {
        self.ind_reg = ind_reg;
        self
    }

    /// set field: n_threads
    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .unwrap();
        self
    }

    fn number_of_runs(&self) -> u64 {
        println!(
            "P(error > ε) < δ.\nUsing ε = {:?} and δ = {:?}",
            self.eps, self.delta
        );
        let _runs = (2.0 / self.delta).ln() / (2.0 * self.eps.powf(2.0));
        //_runs as u64
        1 as u64
    }

    /// Simulation function.
    /// Simulates a run until satisfing the goal predicate, reaching the max
    /// amount of steps or just reaching a deadlock state with no outgoing transitions..
    pub fn simulate(&mut self) -> SimulationOutput {
        self.sim.reset();
        let mut c: usize = 0;
        while let Some(state) = self.sim.next() {
            let next_state = state.into();
            if (self.goal)(&next_state) {
                return SimulationOutput::GoalReached(c);
            } else if c >= self.max_steps {
                return SimulationOutput::MaxSteps;
            }
            c += 1;
        }
        return SimulationOutput::NoStatesAvailable;
    }

    /// Run Statistical Model Checking.
    /// Returns a tuple containing the amount of times that reached the goal state,
    /// and the number of runs.
    pub fn run_smc(mut self) -> (i64, i64) {
        let n_runs = self.number_of_runs();
        println!("Runs: {:?}. Max Steps: {:?}", n_runs, self.max_steps);
        let mut score: i64 = 0;
        let mut count_more_steps_needed = 0;
        let mut deadlock_count = 0;
        let mut _total_steps = 0;
        for _i in 0..n_runs {
            let v = self.simulate();
            match v {
                SimulationOutput::GoalReached(steps) => {
                    score += 1;
                    _total_steps += steps;
                }
                SimulationOutput::MaxSteps => count_more_steps_needed += 1,
                SimulationOutput::NoStatesAvailable => deadlock_count += 1,
            }
        }
        println!(
            "Results:\nMore steps needed: {:?}.\tReached: {:?}.\tDeadlocks: {:?}. steps taken: {:?}",
            count_more_steps_needed, score, deadlock_count, _total_steps
        );
        (score, n_runs as i64)
    }

    /// Explicitly run parallel SMC.
    /// Does not uses the Simulation Output enum, because this
    /// implementation uses the low level managment of threads.
    pub fn explicit_parallel_smc(&self) -> (i64, i64)
    where
        S: Simulator + Clone + Send,
        G: Fn(&S::State<'_>) -> bool + Clone + Send + Sync,
    {
        let n_runs = self.number_of_runs();
        let num_workers = self.n_threads;
        let countdown = atomic::AtomicI64::new(n_runs as i64);
        let max_steps = self.max_steps;
        println!(
            "Runs: {:?}. Max Steps: {:?}. Num Threads: {:?}. Runs per thread: {:?}",
            n_runs,
            max_steps,
            num_workers,
            (n_runs as f64 / num_workers as f64)
        );
        let goal_counter = atomic::AtomicUsize::new(0);
        let dead_counter = atomic::AtomicUsize::new(0);
        let more_steps_counter = atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..num_workers {
                let sim = self.sim.clone();
                let goal_counter = &goal_counter;
                let dead_counter = &dead_counter;
                let more_steps_counter = &more_steps_counter;
                let goal = &self.goal;
                let countdown = &countdown;
                scope.spawn(move || {
                    let mut sim = sim;
                    while countdown.fetch_sub(1, atomic::Ordering::Relaxed) > 0 {
                        match simulate_run(&mut sim, goal, max_steps) {
                            SimulationOutput::GoalReached(_) => {
                                goal_counter.fetch_add(1, atomic::Ordering::Relaxed);
                            }
                            SimulationOutput::MaxSteps => {
                                more_steps_counter.fetch_add(1, atomic::Ordering::Relaxed);
                            }
                            SimulationOutput::NoStatesAvailable => {
                                dead_counter.fetch_add(1, atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
        let score = goal_counter.into_inner() as i64;
        println!(
            "Results:\nMore steps needed: {:?}.\tReached: {:?}.\tDeadlocks: {:?}.",
            more_steps_counter.into_inner() as i64,
            score,
            dead_counter.into_inner() as i64
        );
        (score, n_runs as i64)
    }

    /// Runs parallel SMC usign the library rayon and distribution the wrokload
    /// on the amount of threads.
    pub fn _run_parallel_smc(self) -> (i64, i64)
    where
        S: Simulator + Send + Clone + Sync,
        G: Fn(&S::State<'_>) -> bool + Send + Clone + Sync,
    {
        let n_runs = self.number_of_runs();
        let n_threads = current_num_threads();
        println!(
            "Runs: {:?}. Max Steps: {:?}. Threads: {}",
            n_runs as u64, self.max_steps, n_threads
        );
        let mut score: i64 = 0;
        let mut _count_more_steps_needed = 0;
        let mut _deadlocks = 0;
        let updated = (0..n_runs as u64)
            .into_par_iter()
            .map(|_| _parallel_simulation(self.sim.clone(), self.goal.clone(), self.max_steps));

        let result: Vec<_> = updated.collect();
        for sout in result {
            match sout {
                SimulationOutput::GoalReached(_) => score += 1,
                SimulationOutput::MaxSteps => _count_more_steps_needed += 1,
                SimulationOutput::NoStatesAvailable => _deadlocks += 1,
            }
        }
        (score, n_runs as i64)
    }

    fn _run_parallel_smc_pool(self) -> (i64, i64)
    where
        S: Simulator + Send + Clone + Sync,
        G: Fn(&S::State<'_>) -> bool + Send + Clone + Sync,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build()
            .unwrap();
        let n_runs = self.number_of_runs();
        let mut score: i64 = 0;
        let mut _count_more_steps_needed = 0;
        let cycles = (n_runs as f64 / self.n_threads as f64) as i64;
        let mut simulators: Vec<&S> = vec![];
        for _ in 0..current_num_threads() {
            simulators.push((&self.sim).clone())
        }
        println!(
            "Runs: {:?}. Max Steps: {:?}. Cycles: {}. Threads: {}",
            n_runs, self.max_steps, cycles, self.n_threads
        );
        for _ in 0..cycles {
            let v = pool.broadcast(|_| {
                _parallel_simulation(self.sim.clone(), self.goal.clone(), self.max_steps)
            });
            for sout in v {
                match sout {
                    SimulationOutput::GoalReached(_) => score += 1,
                    SimulationOutput::MaxSteps => _count_more_steps_needed += 1,
                    SimulationOutput::NoStatesAvailable => {
                        println!("No States Available, something went wrong...");
                    }
                }
            }
        }
        (score, n_runs as i64)
    }

    /// Runs the simulation of SPRT algorithm.
    pub fn run_sprt(mut self) -> SprtComparison {
        let p0 = (self.x + self.ind_reg).min(1.0);
        let p1 = (self.x - self.ind_reg).max(0.0);
        let a = ((1.0 - self.alpha) / self.beta).log10();
        let b = (self.beta / (1.0 - self.alpha)).log10();
        let mut finisihed = false;
        let mut r: f64 = 0.0;
        let mut _count_more_steps_needed = 0;
        let mut runs = 0;
        let mut result: Option<SprtComparison> = None;
        while !finisihed {
            runs += 1;
            let v = self.simulate();
            match v {
                SimulationOutput::GoalReached(_) => r += p1.log10() - p0.log10(),
                SimulationOutput::MaxSteps => {
                    _count_more_steps_needed += 1;
                    r += (1.0 - p1).log10() - (1.0 - p0).log10()
                }
                SimulationOutput::NoStatesAvailable => {
                    println!("No States Available, something went wrong...");
                }
            }
            if r <= b {
                finisihed = true;
                println!("P(<>G)>={}>={}", p0, self.x);
                result = Some(SprtComparison::BiggerThan(runs));
            } else if r >= a {
                finisihed = true;
                println!("P(<>G)<={}<={}", p1, self.x);
                result = Some(SprtComparison::LesserThan(runs));
            }
        }
        result.unwrap()
    }
}

fn _parallel_simulation<S, G>(mut sim: S, goal: G, max_steps: usize) -> SimulationOutput
where
    S: Simulator,
    G: Fn(&S::State<'_>) -> bool,
{
    sim.reset();
    let mut c = 0;
    while let Some(state) = sim.next() {
        let next_state = state.into();
        if (goal)(&next_state) {
            return SimulationOutput::GoalReached(c);
        } else if c >= max_steps {
            return SimulationOutput::MaxSteps;
        }
        c += 1;
    }
    return SimulationOutput::NoStatesAvailable;
}

fn simulate_run<S, G>(sim: &mut S, goal: &G, max_steps: usize) -> SimulationOutput
where
    S: Simulator,
    G: Fn(&S::State<'_>) -> bool,
{
    sim.reset();
    let mut c = 0;
    while let Some(state) = sim.next() {
        let next_state = state.into();
        if (goal)(&next_state) {
            return SimulationOutput::GoalReached(c); //return true;
        } else if c >= max_steps {
            return SimulationOutput::MaxSteps; //return false;
        }
        c += 1;
    }
    return SimulationOutput::NoStatesAvailable; // return false;
}
