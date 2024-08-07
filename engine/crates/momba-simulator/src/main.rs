use hashbrown::HashSet;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use clap_derive::Parser;

use momba_explore::{model::Expression, time::Float64Zone, *};

mod nn_oracle;
mod sched_sampler;
mod simulate;
use crate::nn_oracle::*;
use crate::simulate::StatisticalSimulator;

// #[derive(Clap)]
// #[clap(
//     version = "0.1.0",
//     about = "A command line tool directly exposing some model related functionality."
// )]

#[derive(Parser, Debug)]
struct Arguments {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Parser, Clone, Debug)]
enum Command {
    #[clap(about = "Counts the number of states/zones of the model")]
    Info(Info),
    #[clap(about = "Runs SMC with an uniform scheduler")]
    SMC(SMC),
    #[clap(about = "Runs SMC using parallelism with an uniform scheduler")]
    ParSMC(ParSMC),
    #[clap(about = "Runs SPRT with an uniform scheduler")]
    SPRT(SPRT),
    #[clap(about = "Runs SMC using a NN as an oracle")]
    DSMC(NN),
    #[clap(about = "Samples through different random schedulers")]
    SchedSampler(SchedSampler),
}
#[derive(Parser, Debug, Clone)]
struct SchedSampler {
    #[clap(help = "A MombaCR model")]
    model: String,
    #[clap(help = "A property generated in the MombaCR style")]
    property: String,
    #[clap(help = "Amount of schedulers to sample")]
    num_schedulers: usize,
    #[clap(short, long, default_value = "1", help = "number of thread to use")]
    n_threads: usize,
    #[clap(short, long, help = "display the progress bar")]
    display: bool,
}

#[derive(Parser, Debug, Clone)]
struct Info {
    #[clap(help = "A MombaCR model")]
    model: String,
}

#[derive(Parser, Debug, Clone)]
struct SMC {
    #[clap(help = "A MombaCR model")]
    model: String,
    #[clap(help = "A property generated in the MombaCR style")]
    property: String,
    #[clap(short, long, help = "display the progress bar")]
    display: bool,
}
#[derive(Parser, Debug, Clone)]
struct ParSMC {
    #[clap(help = "A MombaCR model")]
    model: String,
    #[clap(help = "A property generated in the MombaCR style")]
    property: String,
    #[clap(short, long, default_value = "1", help = "number of thread to use")]
    n_threads: usize,
    #[clap(short, long, help = "display the progress bar")]
    display: bool,
}
#[derive(Parser, Debug, Clone)]
struct SPRT {
    #[clap(help = "A MombaCR model")]
    model: String,
    #[clap(help = "A property generated in the MombaCR style")]
    property: String,
    #[clap(help = "x such that P(F goal)~x")]
    x: f64,
    #[clap(short, long, help = "display the progress bar")]
    display: bool,
}

#[derive(Parser, Debug, Clone)]
struct NN {
    #[clap(help = "A MombaCR model")]
    model: String,
    #[clap(help = "A property generated in the MombaCR style")]
    property: String,
    #[clap(help = "Neural Network describe in a json file.")]
    nn: String,
    #[clap(
        short,
        long,
        default_value = " ",
        help = "Name of the controlled instance."
    )]
    instance_name: String,
    #[clap(short, long, default_value = "1", help = "number of thread to use")]
    n_threads: usize,
    #[clap(short, long, help = "display the progress bar")]
    display: bool,
}

fn info_of_the_model(info_command: Info) {
    let model_path = Path::new(&info_command.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let start = Instant::now();

    println!(
        "Automatons in the network: {:?}",
        explorer.network.automata.keys()
    );
    let mut g_vars_count = 0;
    for (id, _) in &explorer.network.declarations.global_variables {
        if id.starts_with("local_") {
            continue;
        }
        g_vars_count += 1;
    }
    let mut num_actions: i64 = 0;
    for (_, l) in (&explorer.network.automata.values().next().unwrap().locations).into_iter() {
        num_actions += l.edges.len() as i64;
    }
    println!(
        "Number of global variables (NN input size): {:?}",
        g_vars_count
    );
    println!("Number of actions (NN output size): {:?}", num_actions);
    println!(
        "Number of Transient variables: {:?}",
        explorer.network.declarations.transient_variables.len()
    );

    println!("Exploring...");

    let mut visited: HashSet<State<_>> = HashSet::new();
    let mut pending: Vec<_> = explorer.initial_states();

    let mut count_transitions = 0;
    let mut count_destinations = 0;

    let mut processed = 0;

    while let Some(state) = pending.pop() {
        processed += 1;

        if processed % 20000 == 0 {
            let duration = start.elapsed();
            println!(
                "States: {} ({:.2} [states/s], [waiting {}])",
                visited.len(),
                (processed as f64) / duration.as_secs_f64(),
                pending.len(),
            )
        }

        if !visited.contains(&state) {
            let transitions = explorer.transitions(&state);

            for transition in transitions {
                count_transitions += 1;
                let destinations = explorer.destinations(&state, &transition);
                for destination in destinations {
                    count_destinations += 1;
                    let successor = explorer.successor(&state, &transition, &destination);
                    pending.push(successor);
                }
            }

            visited.insert(state);
        }
    }

    let duration = start.elapsed();

    println!("Time elapsed: {:?}", duration);
    println!("States: {}", visited.len());
    println!("Transitions: {}", count_transitions);
    println!("Destinations: {}", count_destinations);
    println!(
        "{:.2} [states/s]",
        (visited.len() as f64) / duration.as_secs_f64()
    )
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
struct Property {
    operator: MinMax,
    goal: Expression,
    dead: Expression,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MinMax {
    Min,
    Max,
}

fn process_input(command: Command) -> (Explorer<time::Float64Zone>, Property, String, bool) {
    let model = match command {
        Command::SMC(ref c) => &c.model,
        Command::DSMC(ref c) => &c.model,
        Command::ParSMC(ref c) => &c.model,
        Command::SPRT(ref c) => &c.model,
        Command::SchedSampler(ref c) => &c.model,
        Command::Info(ref c) => &c.model,
    };
    let prop = match command {
        Command::SMC(ref c) => &c.property,
        Command::DSMC(ref c) => &c.property,
        Command::ParSMC(ref c) => &c.property,
        Command::SPRT(ref c) => &c.property,
        Command::SchedSampler(ref c) => &c.property,
        _ => panic!("Wrong command"),
    };
    let display = match command {
        Command::SMC(ref c) => c.display,
        Command::DSMC(ref c) => c.display,
        Command::ParSMC(ref c) => c.display,
        Command::SPRT(ref c) => c.display,
        Command::SchedSampler(ref c) => c.display,
        _ => panic!("Wrong command"),
    };

    let name = prop
        .split("/")
        .last()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .strip_prefix("prop_")
        .unwrap()
        .to_owned();

    let model_path = Path::new(&model);
    let prop_path = Path::new(&prop);
    let model_file = File::open(model_path).expect("Unable to open model file!");
    let prop_file = File::open(prop_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop: Property = serde_json::from_reader(BufReader::new(prop_file)).unwrap();
    (explorer, prop, name, display)
}

fn smc(smc_command: SMC) {
    let (explorer, prop, prop_name, display) = process_input(Command::SMC(smc_command));
    let goal_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&prop.goal);

    let oracle_seed = 23;
    let state_iter_seed = 23;
    let mut state_iterator = simulate::StateIter::new(
        Arc::new(explorer),
        simulate::UniformOracle::new(StdRng::seed_from_u64(oracle_seed)),
        StdRng::seed_from_u64(state_iter_seed),
    );
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let stat_checker = StatisticalSimulator::new(&mut state_iterator, goal)._display(display);

    println!("Checking Property: {}", prop_name);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.run_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed: {:?}. Estimated Probability: {:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
}

fn par_smc(psmc_command: ParSMC) {
    let n_threads = psmc_command.n_threads.clone();
    let (explorer, prop, prop_name, display) = process_input(Command::ParSMC(psmc_command));
    let goal_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&prop.goal);
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let oracle_seed = 23;
    let state_iter_seed = 23;
    let mut state_iterator = simulate::StateIter::new(
        Arc::new(explorer),
        simulate::UniformOracle::new(StdRng::seed_from_u64(oracle_seed)),
        StdRng::seed_from_u64(state_iter_seed),
    );

    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker.n_threads(n_threads)._display(display);

    println!("Checking Property: {}", prop_name);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.parallel_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed: {:?}. Estimated Probability: {:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
}

fn sprt(sprt_command: SPRT) {
    let x = sprt_command.x.clone();
    let (explorer, prop, _, display) = process_input(Command::SPRT(sprt_command));
    let goal_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&prop.goal);

    let oracle_seed = 23;
    let state_iter_seed = 23;
    let mut state_iterator = simulate::StateIter::new(
        Arc::new(explorer),
        simulate::UniformOracle::new(StdRng::seed_from_u64(oracle_seed)),
        StdRng::seed_from_u64(state_iter_seed),
    );
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let mut stat_checker = simulate::StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker
        ._with_x(x)
        ._with_ind_reg(0.005)
        ._with_alpha(0.05)
        ._with_beta(0.05)
        ._display(display);
    let result = stat_checker.run_sprt();
    println!("Result: {:?}", result);
}

fn dsmc_nn(nn_command: NN) {
    let nn_path = Path::new(&nn_command.nn);
    let nn_file = File::open(nn_path).expect("Unable to open model file!");

    let n_threads = nn_command.n_threads.clone();
    let instance_name = nn_command.instance_name.clone();
    let (explorer, prop, prop_name, display) = process_input(Command::DSMC(nn_command));
    let goal_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&prop.goal);

    let readed_nn: nn_oracle::JsonNN = serde_json::from_reader(BufReader::new(nn_file))
        .expect("Error while reading model file. Are all the layers supported?");

    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let arc_explorer = Arc::new(explorer);
    let oracle_seed = 23;
    let nn_oracle = NnOracle::build(
        readed_nn,
        arc_explorer.clone(),
        Some(String::from(instance_name)),
        StdRng::seed_from_u64(oracle_seed),
    );

    let state_iter_seed = 23;
    let mut simulator = simulate::StateIter::new(
        arc_explorer.clone(),
        nn_oracle,
        StdRng::seed_from_u64(state_iter_seed),
    );

    let mut stat_checker = StatisticalSimulator::new(&mut simulator, goal);
    stat_checker = stat_checker
        .n_threads(n_threads)
        ._display(false)
        ._display(display);

    println!("Checking Property: {}", prop_name);
    if n_threads > 1 {
        let start = Instant::now();
        let (score, n_runs) = stat_checker.parallel_smc();
        let duration = start.elapsed();
        println!(
            "Time elapsed: {:?}. Estimated Probability: {:?}",
            duration,
            (score as f64 / n_runs as f64)
        )
    } else {
        let start = Instant::now();
        let (score, n_runs) = stat_checker.run_smc();
        let duration = start.elapsed();
        println!(
            "Time elapsed: {:?}. Estimated Probability: {:?}",
            duration,
            (score as f64 / n_runs as f64)
        )
    };
}

fn sample_schedulers(sample_command: SchedSampler) {
    let num_schedulers = sample_command.num_schedulers.clone();
    let n_threads = sample_command.n_threads.clone();
    let (explorer, prop, prop_name, display) = process_input(Command::SchedSampler(sample_command));
    let goal_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&prop.goal);
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let arc_explorer = Arc::new(explorer);
    let mut sampler = sched_sampler::SchedulerSampler::new(arc_explorer, goal, prop.operator);
    sampler = sampler
        ._max_steps(500)
        ._with_n_runs(500)
        .n_threads(n_threads)
        ._display(display);

    let start = Instant::now();
    let best = sampler.sample_schedulers(num_schedulers);
    let duration = start.elapsed();
    let best_result = sampler.run_specific_sample(best.0);
    println!("Checking Property: {:?}", prop_name);
    println!(
        "Time elapsed: {:?}. Estimated Probability: {:?}. Seed: {:?}",
        duration, best_result.1, best_result.0
    );
}

fn main() {
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Info(info_command) => info_of_the_model(info_command),
        Command::SMC(smc_command) => smc(smc_command),
        Command::ParSMC(psmc_command) => par_smc(psmc_command),
        Command::SPRT(sprt_command) => sprt(sprt_command),
        Command::DSMC(nn_command) => dsmc_nn(nn_command),
        Command::SchedSampler(sample_command) => sample_schedulers(sample_command),
    }
}
