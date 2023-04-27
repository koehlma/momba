use hashbrown::HashSet;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use std::{env, fs::File};

use clap::Clap;

use momba_explore::{model::Expression, time::Float64Zone, *};

mod nn_oracle;
mod simulate;
use crate::nn_oracle::*;
use crate::simulate::{StatisticalSimulator, SimulationOutput};

#[derive(Clap)]
#[clap(
    version = "0.1.0",
    about = "A command line tool directly exposing some model related functionality."
)]

struct Arguments {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Clap)]
enum Command {
    #[clap(about = "Counts the number of states/zones of the model")]
    Count(Count),
    #[clap(about = "Simulates a random run of the model")]
    Simulate(Simulate),
    #[clap(about = "Runs SMC with an uniformn scheduler")]
    SMC(SMC),
    #[clap(about = "Runs SPRT with an uniformn scheduler")]
    SPRT(SPRT),
    #[clap(about = "Runs SPRT with an uniformn scheduler")]
    NN(NN),
}

#[derive(Clap)]
struct Count {
    #[clap(about = "A MombaCR model")]
    model: String,
}

#[derive(Clap)]
struct Simulate {
    #[clap(about = "A MombaCR model")]
    model: String,
    property: String,
}

#[derive(Clap)]
struct SMC {
    #[clap(about = "A MombaCR model")]
    model: String,
    property: String,
}

#[derive(Clap)]
struct SPRT {
    #[clap(about = "A MombaCR model")]
    model: String,
    property: String,
}

#[derive(Clap)]
struct NN {
    #[clap(about = "A MombaCR model")]
    model: String,
    property: String,
    nn: String,
}

fn count_states(count: Count) {
    let model_path = Path::new(&count.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let start = Instant::now();

    println!("Exploring...");

    let mut visited: HashSet<State<_>> = HashSet::new();
    let mut pending: Vec<_> = explorer.initial_states();

    let mut count_transitions = 0;

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
                    let successor = explorer.successor(&state, &transition, &destination);
                    pending.push(successor);
                }
            }

            visited.insert(state);
        }
    }

    let duration = start.elapsed();

    println!("Time elapsed is: {:?}", duration);
    println!("States: {}", visited.len());
    println!("Transitions: {}", count_transitions);
    println!(
        "{:.2} [states/s]",
        (visited.len() as f64) / duration.as_secs_f64()
    )
}

fn random_walk(walk: Simulate) {
    let model_path = Path::new(&walk.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let _prop_path = Path::new(&walk.property);
    let _prop_file = File::open(_prop_path).expect("Unable to open model file!");
    let expr: Expression = serde_json::from_reader(BufReader::new(_prop_file)).unwrap();
    let comp_expr = explorer.compiled_network.compile_global_expression(&expr);

    let mut state_iterator = simulate::StateIter::new(explorer, simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&comp_expr).unwrap_bool();

    let stat_checker = simulate::StatisticalSimulator::new(&mut state_iterator, goal)
        .max_steps(99)
        .with_eps(0.01);
    let start = Instant::now();
    let score = stat_checker.run_parallel_smc();
    let duration = start.elapsed();
    println!("Time elapsed is: {:?}. Score:{:?}", duration, score);
}

fn smc(walks: SMC) {
    let model_path = Path::new(&walks.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&walks.property);
    let prop_file = File::open(prop_path).expect("Unable to open model file!");

    let expr: Expression = serde_json::from_reader(BufReader::new(prop_file)).unwrap();
    let comp_expr = explorer.compiled_network.compile_global_expression(&expr);
    let mut state_iterator = simulate::StateIter::new(explorer, simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&comp_expr).unwrap_bool();

    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker. max_steps(99).with_delta(0.05).with_eps(0.05);
    let score = stat_checker.run_smc();

    println!("Score: {}", score);
}

fn sprt(walks: SPRT) {
    let model_path = Path::new(&walks.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&walks.property);
    let prop_file = File::open(prop_path).expect("Unable to open model file!");

    let expr: Expression = serde_json::from_reader(BufReader::new(prop_file)).unwrap();
    let comp_expr = explorer.compiled_network.compile_global_expression(&expr);

    let mut state_iterator = simulate::StateIter::new(explorer, simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&comp_expr).unwrap_bool();

    // build the new structure smc
    // and call the method check.
    // better_smc(&mut state_iterator, closure_is_goal, None, None);

    let mut stat_checker = simulate::StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker
        .with_x(0.20)
        .max_steps(999)
        .with_ind_reg(0.05)
        .with_alpha(0.1)
        .with_beta(0.1);
    let testt = stat_checker.run_sprt();
    println!("Score: {:?}", testt);
}

fn _print_type_of<T>(_: &T) {
    println!("TYPE OF: {}", std::any::type_name::<T>())
}

fn check_nn(nn_command: NN) {
    let model_path = Path::new(&nn_command.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");
    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&nn_command.property);
    let prop_file = File::open(prop_path).expect("Unable to open model file!");

    let expr: Expression = serde_json::from_reader(BufReader::new(prop_file)).unwrap();
    let comp_expr_v2 = explorer.compiled_network.compile_global_expression(&expr);
    let comp_expr = explorer.compiled_network.compile_global_expression(&expr);

    let nn_path = Path::new(&nn_command.nn);
    let nn_file = File::open(nn_path).expect("Unable to open model file!");
    let readed_nn: nn_oracle::NeuralNetwork =
        serde_json::from_reader(BufReader::new(nn_file)).expect("Error while reading model file!");
    let (model, input_size) = build_nn(readed_nn);

    let goal_v2 = move |s: &&State<Float64Zone>| s.evaluate(&comp_expr_v2).unwrap_bool();
    let goal = move |s: &State<Float64Zone>| s.evaluate(&comp_expr).unwrap_bool();

    let mut simulator = NnSimulator::new(model, &explorer, goal, input_size);
    //let mut simulator = simulate::StateIter::new(explorer, simulate::UniformOracle::new());
    let stat_checker = StatisticalSimulator::new(&mut simulator, goal_v2);
    
    // let start = Instant::now();
    // let n_runs = 100;
    // let max_steps = 99;
    // println!("Runs: {:?}. Max Steps: {:?}", n_runs, max_steps);
    // let mut score: i64 = 0;
    // for _ in 0..n_runs as i64 {
    //     let v = simulator.simulate();
    //     match v {
    //         SimulationOutput::GoalReached => score += 1,
    //         SimulationOutput::MaxSteps => {},
    //         SimulationOutput::NoStatesAvailable => {
    //             println!("No States Available, something went wrong...");
    //         }
    //     }
    // }
    // let duration = start.elapsed();
    // println!("Score: {}. Time Elapsed:{:?}", score as f64 / n_runs as f64, duration);
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let arguments = Arguments::parse();
    /*The explorer should be in the main, then, pass a reference of the explorer in each case. */
    match arguments.command {
        Command::Count(count) => count_states(count),
        Command::Simulate(walk) => random_walk(walk),
        Command::SMC(walks) => smc(walks),
        Command::SPRT(walks) => sprt(walks),
        Command::NN(nn_command) => check_nn(nn_command),
    }
}
