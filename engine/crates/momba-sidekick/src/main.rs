use hashbrown::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use clap::Clap;

use clock_zones::Zone;

use momba_explore::*;

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

    for state in pending.iter() {
        println!("{:?}", state);
        let mut valuations = state.valuations().clone();
        println!("{:?}", valuations);
        valuations.future();
        println!("{:?}", valuations);
        visited.insert(state.clone());
    }

    let mut count_transitions = 0;

    let mut processed = 0;

    while let Some(state) = pending.pop() {
        processed += 1;

        if processed % 5000 == 0 {
            let duration = start.elapsed();
            println!(
                "States: {} ({:.2} [states/s], [waiting {}])",
                visited.len(),
                (processed as f64) / duration.as_secs_f64(),
                pending.len(),
            )
        }
        let transitions = explorer.transitions(&state);

        for transition in transitions {
            count_transitions += 1;
            let destinations = explorer.destinations(&state, &transition);
            for destination in destinations {
                let successor = explorer.successor(&state, &transition, &destination);
                //pending.push(successor);
                if visited.insert(successor.clone()) {
                    pending.push(successor);
                }
            }
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

    let simulator = simulate::Simulator::new(simulate::UniformOracle::new());

    simulator.simulate(&explorer, 100);
}

fn main() {
    let arguments = Arguments::parse();

    match arguments.command {
        Command::Count(count) => count_states(count),
        Command::Simulate(walk) => random_walk(walk),
    }
}
