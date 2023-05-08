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
    #[clap(about = "Simulates a random run of the model")]
    Simulate(Simulate),
    #[clap(about = "Runs SMC with an uniform scheduler")]
    SMC(SMC),
    #[clap(about = "Runs SPRT with an uniformn scheduler")]
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
    goal_property: String,
    //dead_property: String,
    //It would be really cool, if we can have all the properties in the same file.
    // And then iterate over each property.
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

    println!("Time elapsed is: {:?}", duration);
    println!("States: {}", visited.len());
    println!("Transitions: {}", count_transitions);
    println!("Destinations: {}", count_destinations);
    println!(
        "{:.2} [states/s]",
        (visited.len() as f64) / duration.as_secs_f64()
    )
}

fn random_walk(walk: Simulate) {
    let model_path = Path::new(&walk.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let tracer_exp: Explorer<time::Float64Zone> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );
    let prop_path = Path::new(&walk.property);
    let _prop_file = File::open(prop_path).expect("Unable to open model file!");
    let _prop_name = walk
        .property
        .split("/")
        .last()
        .unwrap()
        .strip_suffix(".json")
        .unwrap()
        .strip_prefix("prop_")
        .unwrap();

    let mut tracer = simulate::StateIter::new(Arc::new(tracer_exp), simulate::UniformOracle::new());

    let res = tracer.generate_trace(30);
    for (i, e) in res.into_iter().enumerate() {
        println!("Step: {:?}:\n{:#?}", i, e);
    }

    //let stat_checker = simulate::StatisticalSimulator::new(&mut state_iterator, goal)
    //.max_steps(99)
    //.with_eps(0.01);
    //let start = Instant::now();
    //let score = stat_checker.run_parallel_smc();
    //let duration = start.elapsed();
    //println!("Time elapsed is: {:?}. Score:{:?}", duration, score);
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
    //let comp_expr = explorer.compiled_network.compile_global_expression(&expr);

    // let dead_expr: Expression = match &expr {
    //     //When the goal is binary, with an Until: (phi U psi)
    //     // => dead: not phi
    //     Expression::Binary(bin_expr) => Expression::Unary(model::UnaryExpression {
    //         operator: model::UnaryOperator::Not,
    //         operand: bin_expr.left.clone(),
    //     }),
    //     // When the goal is a Finally, then is true U psi => dead: false
    //     // And the default value for the dead expression will be the false predicate.
    //     _ => Expression::Constant(ConstantExpression {
    //         value: model::Value::Bool(false),
    //     }),
    // };
    let goal_comp_expr = explorer.compiled_network.compile_global_expression(&expr);
    // let _dead_comp_expr = explorer
    //     .compiled_network
    //     .compile_global_expression(&dead_expr);
    // println!("Goal Expression:\n{:#?}\n\n", &expr);

    let mut state_iterator =
        simulate::StateIter::new(Arc::new(explorer), simulate::UniformOracle::new());
    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();

    let mut stat_checker = StatisticalSimulator::new(&mut state_iterator, goal);
    stat_checker = stat_checker
        .max_steps(12)
        .with_delta(0.5)
        .with_eps(0.2)
        .n_threads(2);
    let start = Instant::now();
    let (score, n_runs) = stat_checker.run_smc();
    let duration = start.elapsed();
    println!(
        "Property: {}.\nTime elapsed is: {:?}. Prob:{:?}",
        prop_name,
        duration,
        (score as f64 / n_runs as f64)
    );
    println!("Trace: {:#?}", state_iterator.trace);
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

    // for (k, v) in &explorer.network.declarations.global_variables {
    //     println!("Key: {:?}. Value: {:?}", k, v);
    // }
    println!(
        "Amount global declarations: {:?}",
        explorer.network.declarations.global_variables.len()
    );
    // for (k, v) in &explorer.network.declarations.transient_variables {
    //     println!("Key: {:?}. Value: {:?}", k, v);
    // }
    println!(
        "Amount transient declarations: {:?}",
        explorer.network.declarations.transient_variables.len()
    );

    let prop_path = Path::new(&nn_command.goal_property);
    let goal_prop_file = File::open(prop_path).expect("Unable to open model file!");
    let expr: Expression = serde_json::from_reader(BufReader::new(goal_prop_file)).unwrap();

    let dead_expr: Expression = match &expr {
        Expression::Binary(bin_expr) => Expression::Unary(model::UnaryExpression {
            operator: model::UnaryOperator::Not,
            operand: bin_expr.left.clone(),
        }),
        _ => Expression::Constant(ConstantExpression {
            value: model::Value::Bool(false),
        }),
    };
    let goal_comp_expr = explorer.compiled_network.compile_global_expression(&expr);
    let dead_comp_expr = explorer
        .compiled_network
        .compile_global_expression(&dead_expr);

    let ini_state = &explorer.initial_states().into_iter().next().unwrap();

    println!(
        "Goal expr: {:?}.\nDead Expr: {:?}",
        ini_state.evaluate(&goal_comp_expr),
        ini_state.evaluate(&dead_comp_expr)
    );

    let nn_path = Path::new(&nn_command.nn);
    let nn_file = File::open(nn_path).expect("Unable to open model file!");
    let readed_nn: nn_oracle::NeuralNetwork =
        serde_json::from_reader(BufReader::new(nn_file)).expect("Error while reading model file!");

    let goal = |s: &&State<Float64Zone>| s.evaluate(&goal_comp_expr).unwrap_bool();
    let arc_explorer = Arc::new(explorer);

    let nn_oracle = NnOracle::build(readed_nn, arc_explorer.clone());
    let mut simulator = simulate::StateIter::new(arc_explorer.clone(), nn_oracle);
    let mut stat_checker = StatisticalSimulator::new(&mut simulator, goal);
    stat_checker = stat_checker
        .max_steps(100)
        .with_delta(0.5)
        .with_eps(0.05)
        .n_threads(8);

    let start = Instant::now();
    let (score, n_runs) = stat_checker.explicit_parallel_smc();
    let duration = start.elapsed();
    println!(
        "Time elapsed is: {:?}. Score:{:?}",
        duration,
        (score as f64 / n_runs as f64)
    );
}

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Count(count) => count_states(count),
        Command::Simulate(walk) => random_walk(walk),
        Command::SMC(walks) => smc(walks),
        Command::SPRT(walks) => sprt(walks),
        Command::NN(nn_command) => check_nn(nn_command),
    }
}
