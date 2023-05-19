use hashbrown::HashSet;
use momba_explore::model::ConstantExpression;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use clap::Clap;

use momba_explore::{model::Expression, time::Float64Zone, *};

mod nn_oracle;
mod simulate;
use crate::nn_oracle::*;
use crate::simulate::StatisticalSimulator;

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
    #[clap(about = "Runs SMC with an uniform scheduler")]
    SMC(SMC),
    #[clap(about = "Runs SMC using parallelism with an uniform scheduler")]
    ParSMC(ParSMC),
    #[clap(about = "Runs SPRT with an uniform scheduler")]
    SPRT(SPRT),
    #[clap(about = "Runs SMC using a NN as an oracle")]
    NN(NN),
}

#[derive(Clap)]
struct Count {
    #[clap(about = "A MombaCR model")]
    model: String,
}

#[derive(Clap)]
struct SMC {
    #[clap(about = "A MombaCR model")]
    model: String,
    #[clap(about = "A property generated in the MombaCR style")]
    property: String,
}
#[derive(Clap)]
struct ParSMC {
    #[clap(about = "A MombaCR model")]
    model: String,
    #[clap(about = "A property generated in the MombaCR style")]
    property: String,
    #[clap(short, long, default_value = "2", about = "number of thread to use")]
    n_threads: usize,
}
#[derive(Clap)]
struct SPRT {
    #[clap(about = "A MombaCR model")]
    model: String,
    #[clap(about = "A property generated in the MombaCR style")]
    property: String,
}

#[derive(Clap)]
struct NN {
    #[clap(about = "A MombaCR model")]
    model: String,
    #[clap(about = "A property generated in the MombaCR style")]
    goal_property: String,
    #[clap(about = "Neural Network in a json file.")]
    nn: String,
    #[clap(
        short,
        long,
        default_value = " ",
        about = "Name of the controlled instance."
    )]
    instance_name: String,
    #[clap(short, long, default_value = "2", about = "number of thread to use")]
    n_threads: usize,
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

fn smc(walks: SMC) {
    let model_path = Path::new(&walks.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&walks.property);
    let prop_file = File::open(prop_path).expect("Unable to open model file!");
    let prop_name = walks
        .property
        .split("/")
        .last()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .strip_prefix("prop_")
        .unwrap();

    let expr: Expression = serde_json::from_reader(BufReader::new(prop_file)).unwrap();

    let dead_expr: Expression = match &expr {
        //When the goal is binary, with an Until: (phi U psi)
        // => dead: not phi
        Expression::Binary(bin_expr) => Expression::Unary(model::UnaryExpression {
            operator: model::UnaryOperator::Not,
            operand: bin_expr.left.clone(),
        }),
        // When the goal is a Finally, then is true U psi => dead: false
        // And the default value for the dead expression will be the false predicate.
        _ => Expression::Constant(ConstantExpression {
            value: model::Value::Bool(false),
        }),
    };
    let goal_comp_expr = explorer.compiled_network.compile_global_expression(&expr);
    let _dead_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&dead_expr);

    let mut state_iterator =
        simulate::StateIter::new(Arc::new(explorer), simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker.max_steps(500).with_delta(0.05).with_eps(0.01);
    println!("Checking Property: {}", prop_name);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.run_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed: {:?}. Estimated Probability:{:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
}

fn par_smc(walks: ParSMC) {
    let model_path = Path::new(&walks.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&walks.property);
    let prop_file = File::open(prop_path).expect("Unable to open model file!");
    let prop_name = walks
        .property
        .split("/")
        .last()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .strip_prefix("prop_")
        .unwrap();

    let expr: Expression = serde_json::from_reader(BufReader::new(prop_file)).unwrap();

    let dead_expr: Expression = match &expr {
        //When the goal is binary, with an Until: (phi U psi)
        // => dead: not phi
        Expression::Binary(bin_expr) => Expression::Unary(model::UnaryExpression {
            operator: model::UnaryOperator::Not,
            operand: bin_expr.left.clone(),
        }),
        // When the goal is a Finally, then is true U psi => dead: false
        // And the default value for the dead expression will be the false predicate.
        _ => Expression::Constant(ConstantExpression {
            value: model::Value::Bool(false),
        }),
    };
    let goal_comp_expr = explorer.compiled_network.compile_global_expression(&expr);
    let _dead_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&dead_expr);

    let mut state_iterator =
        simulate::StateIter::new(Arc::new(explorer), simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker.max_steps(500).n_threads(walks.n_threads);

    println!("Checking Property: {}", prop_name);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.explicit_parallel_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed: {:?}. Estimated Probability:{:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
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

    //let mut state_iterator = simulate::StateIter::new(explorer, simulate::UniformOracle::new());
    let mut state_iterator =
        simulate::StateIter::new(Arc::new(explorer), simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&comp_expr).unwrap_bool();

    let mut stat_checker = simulate::StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker
        .with_x(0.20)
        .max_steps(999)
        .with_ind_reg(0.05)
        .with_alpha(0.1)
        .with_beta(0.1);
    let testt = stat_checker.run_sprt();
    println!("Estimated Probability: {:?}", testt);
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
    let prop_name = nn_command
        .goal_property
        .split("/")
        .last()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .strip_prefix("prop_")
        .unwrap();
    let prop_path = Path::new(&nn_command.goal_property);
    let goal_prop_file = File::open(prop_path).expect("Unable to open model file!");
    let expr: Expression = serde_json::from_reader(BufReader::new(goal_prop_file)).unwrap();

    let goal_comp_expr = explorer.compiled_network.compile_global_expression(&expr);

    let nn_path = Path::new(&nn_command.nn);
    let nn_file = File::open(nn_path).expect("Unable to open model file!");
    let readed_nn: nn_oracle::JsonNN =
        serde_json::from_reader(BufReader::new(nn_file)).expect("Error while reading model file!");

    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();
    let arc_explorer = Arc::new(explorer);

    let nn_oracle = NnOracle::build(
        readed_nn,
        arc_explorer.clone(),
        Some(String::from(nn_command.instance_name)),
    );
    let mut simulator = simulate::StateIter::new(arc_explorer.clone(), nn_oracle);
    let mut stat_checker = StatisticalSimulator::new(&mut simulator, goal);
    stat_checker = stat_checker.n_threads(nn_command.n_threads).max_steps(500);

    println!("Checking Property: {}", prop_name);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.explicit_parallel_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed: {:?}. Estimated Probability:{:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
}

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Count(count) => count_states(count),
        Command::SMC(walks) => smc(walks),
        Command::ParSMC(walks) => par_smc(walks),
        Command::SPRT(walks) => sprt(walks),
        Command::NN(nn_command) => check_nn(nn_command),
    }
}
