use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use momba_explore::{model::Value, *};
use rand::{rngs::StdRng, seq::IteratorRandom, Rng};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Write,
    sync::{atomic, Arc},
    thread::sleep,
    time::Duration,
};

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
    /// Precondition: transitions is not empty.
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'s, T>;
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
        &self,
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'s, T> {
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
        &self,
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'s, T> {
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
    /// Default value: 2500
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
    /// Number of runs in the simulation
    n_runs: Option<u64>,
    /// Display initial conditions and progress.
    display: bool,
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
            eps: 0.05,
            delta: 0.05,
            max_steps: 2500,
            x: 0.0,
            alpha: 1.0,
            beta: 1.0,
            ind_reg: 0.0,
            n_threads: 1,
            n_runs: None,
            display: false,
        }
    }

    /// set field: steps
    pub fn _max_steps(mut self, max_steps: usize) -> Self {
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

    // set number of runs
    pub fn _with_n_runs(mut self, runs: u64) -> Self {
        self.n_runs = Some(runs as u64);
        self
    }

    /// set field: n_threads
    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// set display
    pub fn _display(mut self, display: bool) -> Self {
        self.display = display;
        self
    }

    /// Set number of runs for the simulation.
    /// If not used, then the simulations uses the Okamoto bound.
    fn number_of_runs(&self) -> u64 {
        match self.n_runs {
            None => {
                if self.display {
                    println!(
                        "P(error > ε) < δ.\nUsing ε = {:?} and δ = {:?}",
                        self.eps, self.delta
                    );
                }
                let runs = (2.0 / self.delta).ln() / (2.0 * self.eps.powf(2.0));
                runs as u64
            }
            Some(r) => r,
        }
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

    /// Statistical Model Checking Algorithm
    /// Runs one simulation, return the values encoded in a tuple.
    fn smc(&mut self) -> (i64, i64, i64) {
        let mut score: i64 = 0;
        let mut count_more_steps_needed = 0;
        let mut deadlock_count = 0;
        let v = self.simulate();
        match v {
            SimulationOutput::GoalReached(_) => {
                score += 1;
            }
            SimulationOutput::MaxSteps => count_more_steps_needed += 1,
            SimulationOutput::NoStatesAvailable => deadlock_count += 1,
        }
        (score, count_more_steps_needed, deadlock_count)
    }

    /// Run Statistical Model Checking.
    /// Returns a tuple containing the amount of times that reached the goal state,
    /// and the number of runs.
    pub fn run_smc(mut self) -> (i64, i64) {
        let n_runs = self.number_of_runs();
        let mut score: i64 = 0;
        if self.display {
            let mut count_more_steps_needed = 0;
            let mut deadlock_count = 0;
            println!("Runs:\t\t{:?}\nMax Steps:\t{:?}", n_runs, self.max_steps);
            let pb = ProgressBar::new(n_runs);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% ({eta})",
                )
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                    write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                })
                .progress_chars("#>-"),
            );

            for i in 0..n_runs {
                pb.inc(1);
                pb.set_position(i);
                let (sim_score, sim_more_steps, sim_deadlock) = self.smc();
                score += sim_score;
                count_more_steps_needed += sim_more_steps;
                deadlock_count += sim_deadlock;
            }
            pb.finish_with_message("Simulation Finished. Results:\n");
            println!(
                "More steps needed: {:?}.\tReached: {:?}.\tDeadlocks: {:?}",
                count_more_steps_needed, score, deadlock_count
            );
        } else {
            for _ in 0..n_runs {
                let (sim_score, _, _) = self.smc();
                score += sim_score;
            }
        }
        (score, n_runs as i64)
    }

    /// Parallel SMC.
    /// Does not uses the Simulation Output enum, because this
    /// implementation uses the low level managment of threads.
    pub fn parallel_smc(&self) -> (i64, i64)
    where
        S: Simulator + Clone + Send,
        G: Fn(&S::State<'_>) -> bool + Clone + Send + Sync,
    {
        let n_runs = self.number_of_runs();
        let num_workers = self.n_threads;
        let max_steps = self.max_steps;
        let countdown = atomic::AtomicI64::new(n_runs as i64);
        let goal_counter = atomic::AtomicUsize::new(0);
        let dead_counter = atomic::AtomicUsize::new(0);
        let more_steps_counter = atomic::AtomicUsize::new(0);

        if self.display {
            let pb = ProgressBar::new(n_runs);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% ({eta})",
                )
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                    write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                })
                .progress_chars("#>-"),
            );
            println!(
                "Runs:\t\t\t{:?}\nMax Steps:\t\t{:?}\nNum Threads:\t\t{:?}\nRuns per thread:\t{:?}",
                n_runs,
                max_steps,
                num_workers,
                (n_runs as f64 / num_workers as f64)
            );

            std::thread::scope(|scope| {
                for _i in 0..num_workers {
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
                while countdown.fetch_sub(1, atomic::Ordering::Relaxed) > 0 {
                    let c = n_runs as i64 - countdown.fetch_add(0, atomic::Ordering::Relaxed);
                    pb.set_position(c as u64);
                    sleep(Duration::new(1, 0));
                }
            });
            pb.finish();
        } else {
            std::thread::scope(|scope| {
                for _i in 0..num_workers {
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
        }
        let score = goal_counter.into_inner() as i64;
        if self.display {
            println!(
                "Results:\nMore steps needed:\t{:?}\nReached:\t\t{:?}\nDeadlocks:\t\t{:?}",
                more_steps_counter.into_inner() as i64,
                score,
                dead_counter.into_inner() as i64
            );
        }
        (score, n_runs as i64)
    }

    /// Runs the simulation of SPRT algorithm.
    pub fn run_sprt(mut self) -> SprtComparison {
        if self.display {
            println!(
            "SPRT algorithm with: \n\t Max Steps: {:?}\n\t Indifference region: {:?}\n\t x: {:?}\n\t Type Err I: {:?} - Type Err II: {:?} ",
            self.max_steps, self.ind_reg, self.x, self.alpha, self.beta
        );
        }
        let p0 = (self.x + self.ind_reg).min(1.0);
        let p1 = (self.x - self.ind_reg).max(0.0);
        let a = ((1.0 - self.alpha) / self.beta).log10();
        let b = (self.beta / (1.0 - self.alpha)).log10();
        let mut finisihed = false;
        let mut r: f64 = 0.0;
        let mut runs = 0;
        let mut result: Option<SprtComparison> = None;
        while !finisihed {
            runs += 1;
            let v = self.simulate();
            match v {
                SimulationOutput::GoalReached(_) => r += p1.log10() - p0.log10(),
                SimulationOutput::MaxSteps => {
                    r += (1.0 - p1).log10() - (1.0 - p0).log10();
                }
                SimulationOutput::NoStatesAvailable => {
                    r += (1.0 - p1).log10() - (1.0 - p0).log10();
                }
            }
            if r <= b {
                finisihed = true;
                if self.display {
                    println!("P(<>G)>={}>={}. In {} runs.", p0, self.x, runs);
                }
                result = Some(SprtComparison::BiggerThan(runs));
            } else if r >= a {
                finisihed = true;
                if self.display {
                    println!("P(<>G)<={}<={}. In {} runs.", p1, self.x, runs);
                }
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
            return SimulationOutput::GoalReached(c);
        } else if c >= max_steps {
            return SimulationOutput::MaxSteps;
        }
        c += 1;
    }
    return SimulationOutput::NoStatesAvailable;
}
