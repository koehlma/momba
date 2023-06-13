use crate::{
    simulate::{self, Oracle, SimulationOutput, StatisticalSimulator},
    MinMax,
};
use ahash::RandomState;
use momba_explore::*;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

/// Implementation of the Oracle for a Hash Function.
/// Keeps a reference to the explorer and the Hash Builder.
#[derive(Clone)]
pub struct HashOracle<T: time::Time> {
    hash_builder: RandomState,
    explorer: Arc<Explorer<T>>,
}

impl<T: time::Time> HashOracle<T> {
    ///Create a new Hash Oracle.
    pub fn new(explorer: Arc<Explorer<T>>, seed: usize) -> Self {
        let hash_builder = RandomState::with_seed(seed);
        HashOracle {
            hash_builder,
            explorer,
        }
    }

    /// Translate the State to a vectorial representation, just like in the NN implementation
    /// but we parse all the element to bits or u64, because floats cant be hashed properly.
    fn state_representation(&self, state: &State<T>) -> Vec<u64> {
        let mut values = vec![];
        for (id, t) in &self.explorer.network.declarations.global_variables {
            if id.starts_with("local_") {
                continue;
            }
            match t {
                model::Type::Vector { element_type: _ } => panic!("Type not valid"),
                model::Type::Unknown => panic!("Type not valid"),
                model::Type::Bool => values.push(
                    state
                        .get_global_value(&self.explorer, &id)
                        .unwrap()
                        .unwrap_bool() as u64,
                ),
                model::Type::Float64 => values.push(
                    state
                        .get_global_value(&self.explorer, &id)
                        .unwrap()
                        .unwrap_float64()
                        .into_inner()
                        .to_bits(),
                ),
                model::Type::Int64 => values.push(
                    state
                        .get_global_value(&self.explorer, &id)
                        .unwrap()
                        .unwrap_int64() as u64,
                ),
            }
        }
        values
    }
}

impl<T: time::Time> Oracle<T> for HashOracle<T> {
    /// Choosing method for the Hash Oracle.
    /// It Hashes the representation of the state, then we choose the transition taking
    /// the hash value modulo the amount of transitions.
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'s, T> {
        if transitions.len() > 1 {
            let values = self.state_representation(state);
            let hashed = self.hash_builder.hash_one(values);
            let idx = hashed as usize % transitions.len();
            let elected_transition = transitions
                .into_iter()
                .enumerate()
                .filter(|(i, _)| *i == idx)
                .map(|(_, t)| t)
                .next()
                .unwrap();
            elected_transition
        } else {
            transitions.first().unwrap()
        }
    }
}

/// Strutcture that will create Simulators and State Iterator for each oracle.
/// Then, keeping the best one,
pub struct SchedulerSampler<T, G>
where
    T: time::Time,
    G: Fn(&&State<T>) -> bool + Sync + Send + Copy,
{
    explorer: Arc<Explorer<T>>,
    goal: G,
    n_threads: usize,
    max_steps: usize,
    n_runs: usize,
    op: MinMax,
}

impl<T, G> SchedulerSampler<T, G>
where
    T: time::Time + 'static + Clone,
    G: Fn(&&State<T>) -> bool + Sync + Send + Copy,
{
    /// Create new Sampler.
    pub fn new(explorer: Arc<Explorer<T>>, goal: G, op: MinMax) -> Self {
        SchedulerSampler {
            explorer,
            goal,
            n_runs: 500,
            max_steps: 500,
            n_threads: 1,
            op,
        }
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
    /// set field: steps
    pub fn _max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    // set number of runs
    pub fn _with_n_runs(mut self, runs: usize) -> Self {
        self.n_runs = runs;
        self
    }

    /// Sample Schedulers.
    /// It does it in parallel using rayon lib.
    pub fn sample_schedulers(&mut self, amount_samples: usize) -> (usize, f64) {
        println!(
            "Sampling between {:?} schedulers. Type: {:?}",
            amount_samples, self.op
        );
        let mut best: (usize, f64);
        match self.op {
            MinMax::Max => best = (0, 0.0),
            MinMax::Min => best = (0, 1.0),
        }
        // let pb = ProgressBar::new(amount_samples as u64);
        // See how to fix the progress bar with rayon on parallel.
        // pb.set_style(
        //     ProgressStyle::with_template(
        //         "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% ({eta})",
        //     )
        //     .unwrap()
        //     .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
        //         write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
        //     })
        //     .progress_chars("#>-"),
        // );
        let updated = (0..amount_samples).into_par_iter().map(|i| {
            //pb.set_position(i);
            _run_sample(
                self.explorer.clone(),
                self.goal,
                i,
                self.max_steps,
                self.n_runs,
            )
        });
        // pb.finish();
        let result: Vec<_> = updated.collect();
        for (seed, score) in result {
            match self.op {
                MinMax::Max => {
                    if score > best.1 {
                        best = (seed, score);
                    }
                }
                MinMax::Min => {
                    if score < best.1 {
                        best = (seed, score);
                    }
                }
            }
        }
        println!(
            "Sampling finished. Seed: {:?} Score: {:?}\n",
            best.0, best.1
        );
        best
    }

    fn _simulate(&mut self, seed: usize) -> SimulationOutput {
        let mut state_iterator = simulate::StateIter::new(
            self.explorer.clone(),
            HashOracle::new(self.explorer.clone(), seed),
            StdRng::seed_from_u64(10),
        );
        let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, self.goal);
        stat_checker = stat_checker._max_steps(self.max_steps);
        stat_checker.simulate()
    }

    /// Run with one specific seed.
    pub fn run_specific_sample(&self, seed: usize) -> (usize, f64) {
        let mut state_iterator = simulate::StateIter::new(
            self.explorer.clone(),
            HashOracle::new(self.explorer.clone(), seed),
            StdRng::seed_from_u64(10),
        );
        let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, self.goal);
        stat_checker = stat_checker._max_steps(self.max_steps);

        let (score, n_runs) = stat_checker.parallel_smc();
        let sim_result = score as f64 / n_runs as f64;
        (seed, sim_result)
    }
}

/// Helper function for the parallelism
fn _run_sample<T, G>(
    explorer: Arc<Explorer<T>>,
    goal: G,
    seed: usize,
    max_steps: usize,
    n_runs: usize,
) -> (usize, f64)
where
    T: time::Time + 'static,
    G: Fn(&&State<T>) -> bool,
{
    let mut state_iterator = simulate::StateIter::new(
        explorer.clone(),
        HashOracle::new(explorer.clone(), seed),
        StdRng::seed_from_u64(10),
    );
    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker
        ._with_n_runs(n_runs as u64)
        ._max_steps(max_steps)
        ._display(false);
    let (score, n_runs) = stat_checker.run_smc();
    let sim_result = score as f64 / n_runs as f64;
    (seed, sim_result)
}
