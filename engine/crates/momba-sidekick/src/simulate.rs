use std::{
    collections::HashMap,
    sync::{atomic, Arc},
};

use momba_explore::{model::Value, *};
use rand::{seq::IteratorRandom, Rng};
use rayon::{current_num_threads, prelude::*};

#[derive(Debug)]
pub enum SprtComparison {
    BiggerThan(i64),
    LesserThan(i64),
}
#[derive(Debug, PartialEq)]
pub enum SimulationOutput {
    GoalReached(usize),
    MaxSteps,
    NoStatesAvailable,
}

pub trait Oracle<T: time::Time> {
    fn choose<'s, 't>(
        &self,
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T>;
}
#[derive(Clone)]
pub struct UniformOracle {}

impl UniformOracle {
    pub fn new() -> Self {
        UniformOracle {}
    }
}

impl<T: time::Time> Oracle<T> for UniformOracle {
    fn choose<'s, 't>(
        &self,
        _state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        let mut rng = rand::thread_rng();
        let elected_transition = transitions.into_iter().choose(&mut rng).unwrap();
        elected_transition
    }
}

pub trait Simulator {
    type State<'sim>
    where
        Self: 'sim;
    fn next(&mut self) -> Option<Self::State<'_>>;
    fn reset(&mut self) -> Self::State<'_>;
    fn current_state(&mut self) -> Self::State<'_>;
}

#[derive(Clone)]
pub struct StateIter<T: time::Time, O: Oracle<T>> {
    pub state: State<T>,
    explorer: Arc<Explorer<T>>,
    oracle: O,
    pub trace: Vec<HashMap<String, Value>>,
}

impl<T: time::Time, O: Oracle<T>> StateIter<T, O> {
    pub fn new(explorer: Arc<Explorer<T>>, oracle: O) -> Self {
        let mut rng = rand::thread_rng();
        StateIter {
            state: explorer
                .initial_states()
                .into_iter()
                .choose(&mut rng)
                .unwrap(),
            explorer,
            oracle,
            trace: vec![],
        }
    }

    pub fn _generate_trace(&mut self, steps: usize) -> Vec<HashMap<String, Value>> {
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

impl<T: time::Time, O: Oracle<T>> Simulator for StateIter<T, O> {
    type State<'sim> = &'sim State<T> where Self:'sim;
    fn next(&mut self) -> Option<Self::State<'_>> {
        let mut rng = rand::thread_rng();
        let transitions = self.explorer.transitions(&self.state);
        if transitions.is_empty() {
            return None;
        }
        let transition = self.oracle.choose(&self.state, &transitions);
        let destinations = self.explorer.destinations(&self.state, &transition);

        if destinations.is_empty() {
            return None;
        }
        let threshold: f64 = rng.gen();
        let mut accumulated = 0.0;

        for destination in destinations {
            accumulated += destination.probability();
            if accumulated >= threshold {
                self.state = self
                    .explorer
                    .successor(&self.state, &transition, &destination);
                break;
            }
        }
        Some(&self.state)
    }

    fn reset(&mut self) -> Self::State<'_> {
        let mut rng = rand::thread_rng();
        self.state = self
            .explorer
            .initial_states()
            .into_iter()
            .choose(&mut rng)
            .unwrap();
        &self.state
    }

    fn current_state(&mut self) -> Self::State<'_> {
        &self.state
    }

    // I commented it, because can be useful in the future.
    // fn state_representation(&mut self) -> HashMap<String, Value> {
    //     let mut state_rep: HashMap<String, Value> = HashMap::new();
    //     for (id, _) in &self.explorer.network.declarations.global_variables {
    //         let val = self.state.get_global_value(&self.explorer, id).unwrap();
    //         state_rep.insert(id.clone(), val.clone());
    //     }
    //     self.trace.push(state_rep.clone());
    //     state_rep
    // }
}

#[must_use]
pub struct StatisticalSimulator<'sim, S, G> {
    sim: &'sim mut S,
    goal: G,
    eps: f64,
    delta: f64,
    max_steps: usize,
    x: f64,
    alpha: f64,
    beta: f64,
    ind_reg: f64,
    n_threads: usize,
}

impl<'sim, S, G> StatisticalSimulator<'sim, S, G>
where
    S: Simulator,
    G: Fn(&S::State<'_>) -> bool,
{
    pub fn new(sim: &'sim mut S, goal: G) -> Self {
        Self {
            sim,
            goal,
            eps: 0.01,
            delta: 0.05,
            max_steps: 200,
            x: 0.0,
            alpha: 1.0,
            beta: 1.0,
            ind_reg: 0.0,
            n_threads: 4,
        }
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    pub fn with_x(mut self, x: f64) -> Self {
        self.x = x;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_ind_reg(mut self, ind_reg: f64) -> Self {
        self.ind_reg = ind_reg;
        self
    }

    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .unwrap();
        self
    }

    fn simulate(&mut self) -> SimulationOutput {
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

    pub fn explicit_parallel_smc(&self) -> (i64, i64)
    where
        S: Simulator + Clone + Send,
        G: Fn(&S::State<'_>) -> bool + Clone + Send + Sync,
    {
        let n_runs =
            (f64::log(2.0 / self.delta, std::f64::consts::E)) / (2.0 * self.eps.powf(2.0)) as f64;
        let num_workers = self.n_threads;
        let countdown = atomic::AtomicI64::new(n_runs as i64);
        let max_steps = self.max_steps;
        println!(
            "Runs: {:?}. Max Steps: {:?}. Num Threads: {:?}. Runs per thread: {:?}",
            n_runs as i64,
            max_steps,
            num_workers,
            (n_runs / num_workers as f64)
        );
        let goal_counter = atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..num_workers {
                let sim = self.sim.clone();
                let goal_counter = &goal_counter;
                let goal = &self.goal;
                let countdown = &countdown;
                scope.spawn(move || {
                    let mut sim = sim;
                    while countdown.fetch_sub(1, atomic::Ordering::Relaxed) > 0 {
                        if simulate_run(&mut sim, goal, max_steps) {
                            goal_counter.fetch_add(1, atomic::Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        (goal_counter.into_inner() as i64, n_runs as i64)
    }

    pub fn run_smc(mut self) -> (i64, i64) {
        let n_runs =
            (f64::log(2.0 / self.delta, std::f64::consts::E)) / (2.0 * self.eps.powf(2.0)) as f64;
        println!("Runs: {:?}. Max Steps: {:?}", n_runs as i64, self.max_steps);
        let mut score: i64 = 0;
        let mut count_more_steps_needed = 0;
        let mut deadlock_count = 0;
        let mut total_steps = 0;
        for _ in 0..n_runs as i64 {
            let v = self.simulate();
            match v {
                SimulationOutput::GoalReached(steps) => {
                    println!("Goal reached at step: {:?}", steps);
                    score += 1;
                    total_steps += steps;
                }
                SimulationOutput::MaxSteps => count_more_steps_needed += 1,
                SimulationOutput::NoStatesAvailable => deadlock_count += 1,
            }
        }
        println!(
            "More steps needed: {:?}. Reached: {:?}. Deadlocks: {:?}. Steps: {:?}",
            count_more_steps_needed, score, deadlock_count, total_steps
        );
        (score, n_runs as i64)
    }

    pub fn _run_parallel_smc(self) -> (i64, i64)
    where
        S: Simulator + Send + Clone + Sync,
        G: Fn(&S::State<'_>) -> bool + Send + Clone + Sync,
    {
        let n_runs =
            (f64::log(2.0 / self.delta, std::f64::consts::E)) / (2.0 * self.eps.powf(2.0)) as f64;
        let n_threads = current_num_threads();
        println!(
            "Runs: {:?}. Max Steps: {:?}. Threads: {}",
            n_runs as i64, self.max_steps, n_threads
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

    pub fn _run_parallel_smc_pool(self) -> (i64, i64)
    where
        S: Simulator + Send + Clone + Sync,
        G: Fn(&S::State<'_>) -> bool + Send + Clone + Sync,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build()
            .unwrap();
        let n_runs =
            (f64::log(2.0 / self.delta, std::f64::consts::E)) / (2.0 * self.eps.powf(2.0)) as f64;
        let mut score: i64 = 0;
        let mut _count_more_steps_needed = 0;
        let cycles = (n_runs as f64 / self.n_threads as f64) as i64;
        let mut simulators: Vec<&S> = vec![];
        for _ in 0..current_num_threads() {
            simulators.push((&self.sim).clone())
        }
        println!(
            "Runs: {:?}. Max Steps: {:?}. Cycles: {}. Threads: {}",
            n_runs as i64, self.max_steps, cycles, self.n_threads
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

    pub fn run_sprt(mut self) -> SprtComparison {
        let p0 = (self.x + self.ind_reg).min(1.0);
        let p1 = (self.x - self.ind_reg).max(0.0);
        let a = f64::log10((1.0 - self.alpha) / self.beta);
        let b = f64::log10(self.beta / (1.0 - self.alpha));
        let mut finisihed: bool = false;
        let mut r: f64 = 0.0;
        let mut _count_more_steps_needed = 0;
        let mut runs = 0;
        let mut result: Option<SprtComparison> = None;
        while !finisihed {
            runs += 1;
            let v = self.simulate();
            match v {
                SimulationOutput::GoalReached(_) => r += f64::log10(p1) - f64::log10(p0),
                SimulationOutput::MaxSteps => {
                    _count_more_steps_needed += 1;
                    r += f64::log10(1.0 - p1) - f64::log10(1.0 - p0)
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

fn simulate_run<S, G>(sim: &mut S, goal: &G, max_steps: usize) -> bool
where
    S: Simulator,
    G: Fn(&S::State<'_>) -> bool,
{
    sim.reset();
    let mut c = 0;
    while let Some(state) = sim.next() {
        let next_state = state.into();
        if (goal)(&next_state) {
            return true;
        } else if c >= max_steps {
            return false;
        }
        c += 1;
    }
    return false;
}
