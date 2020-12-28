use hashbrown::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use clap::Clap;

use momba_explore::explore;
use momba_explore::model;

use model::*;

#[derive(Clap)]
#[clap(version = "0.1.0", about = "A VM for MombaCR.")]
struct Arguments {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Clap)]
enum Command {
    #[clap(about = "Counts the number of states of the model")]
    Count(Count),
}

#[derive(Clap)]
struct Count {
    #[clap(about = "A MombaCR model")]
    model: String,
}

fn main() {
    let arguments = Arguments::parse();
    let model_path = Path::new(match &arguments.command {
        Command::Count(count) => &count.model,
    });
    let model_file = File::open(model_path).expect("Unable to open model file!");

    println!("Reading...");
    let network: Network = serde_json::from_reader(BufReader::new(model_file))
        .expect("Error while reading model file!");

    let explorer: explore::Explorer<()> = explore::Explorer::new(&network);
    let start = Instant::now();

    println!("Exploring...");

    let mut visited: HashSet<explore::State<_>> = HashSet::new();
    let mut pending: Vec<_> = explorer.initial_states(&network);

    for state in pending.iter() {
        visited.insert(state.clone());
    }

    let mut count_transitions = 0;

    let mut processed = 0;

    while pending.len() != 0 {
        let state = pending.pop().unwrap();
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
