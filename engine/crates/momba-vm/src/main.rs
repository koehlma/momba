// use crate::engine::EvaluationContext;

use fnv::FnvHashSet;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use clap::Clap;

mod model;
use model::*;

mod types;
mod values;

mod compiled;

// mod engine;

#[derive(Clap)]
#[clap(version = "0.1.0", about = "A VM for MombaCR.")]
struct Opts {
    #[clap(about = "A MombaCR model")]
    model: String,

    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    #[clap(about = "Counts the number of states of the model")]
    Count(Count),
}

#[derive(Clap)]
struct Count {}

fn main() {
    println!("Hello!");
    let opts: Opts = Opts::parse();
    let model_path = Path::new(&opts.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    println!("Reading...");
    let network: Network =
        serde_json::from_reader(model_file).expect("Error while reading model file!");

    println!("Compiling...");
    let compiled = network.compile();

    let start = Instant::now();

    println!("Exploring...");

    let mut visited: FnvHashSet<(Box<[values::Value]>, Box<[usize]>)> = FnvHashSet::default();
    let mut pending: Vec<compiled::CompiledState> = compiled.initial_states().into();

    for state in pending.iter() {
        visited.insert((state.values.clone(), state.locations.clone()));
    }

    let mut count_transitions = 0;

    let mut processed = 0;

    while pending.len() != 0 {
        let state = pending.pop().unwrap();
        processed += 1;

        if processed % 5000 == 0 {
            let duration = start.elapsed();
            println!(
                "{:.2} [states/s]",
                (processed as f64) / duration.as_secs_f64()
            )
        }
        let transitions = state.transitions();

        for transition in transitions {
            count_transitions += 1;
            let destinations = state.execute(transition);
            for destination in destinations {
                let compiled::ComputedDestination { probability, state } = destination;
                let compiled::CompiledState {
                    network: _,
                    values,
                    locations,
                } = state;
                let index_tuple = (values, locations);
                if !visited.contains(&index_tuple) {
                    visited.insert(index_tuple.clone());
                    pending.push(compiled::CompiledState {
                        network: &compiled,
                        values: index_tuple.0,
                        locations: index_tuple.1,
                    })
                }
            }
            // for destination in  {
            //     let values_locations = (
            //         destination.state.values.clone(),
            //         destination.state.locations.clone(),
            //     );
            //     if !visited.contains(&values_locations) {
            //         visited.insert(values_locations.clone());
            //         pending.push(destination.state.clone());
            //     }
            // }
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

    // for initial_state in initial_states.iter() {
    //     println!("found state")
    // }

    //println!("{:?}", network);

    // println!("{:?}", compiled);
}
