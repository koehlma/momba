use hashbrown::{hash_set, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use clap::Clap;

use momba_explore::*;

#[derive(Clap)]
#[clap(
    version = "0.1.0",
    about = "A command line tool for verifying synthetic sensors."
)]

struct Arguments {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Clap)]
enum Command {
    Verify(Verify),
}

#[derive(Clap, Debug)]
struct Verify {
    model: String,
}

fn explore_states(explorer: &Explorer<time::NoClocks>) -> HashSet<State<()>> {
    let start = Instant::now();

    println!("Exploring state space...");

    let mut visited: HashSet<State<_>> = HashSet::new();
    let mut pending: Vec<_> = explorer.initial_states();

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

        if !visited.contains(&state) {
            let transitions = explorer.transitions(&state);

            for transition in transitions {
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
    println!(
        "{:.2} [states/s]",
        (visited.len() as f64) / duration.as_secs_f64()
    );

    visited
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CombinedState {
    actual_state: State<()>,
    possible_states: im::HashSet<State<()>>,
}

fn explore_until_action(
    explorer: &Explorer<time::NoClocks>,
    state: State<()>,
    action_index: usize,
) -> im::HashSet<State<()>> {
    let mut states = im::HashSet::new();

    for transition in explorer.transitions(&state) {
        let action = transition.result_action();
        if action.label_index() != Some(action_index) {
            for destination in explorer.destinations(&state, &transition) {
                let successor = explorer.successor(&state, &transition, &destination);
                states = states.union(explore_until_action(explorer, successor, action_index))
            }
        }
    }

    states.insert(state);

    states
}

fn verify(args: Verify) {
    let model_path = Path::new(&args.model);
    let model_file = File::open(model_path).expect("Unable to open model file!");

    let explorer: Explorer<time::NoClocks> = Explorer::new(
        serde_json::from_reader(BufReader::new(model_file))
            .expect("Error while reading model file!"),
    );

    let mut pending = Vec::new();
    let mut visited = HashSet::new();

    let sense_index = explorer
        .network
        .declarations
        .action_labels
        .get_index_of("sense")
        .expect("no sense action found");

    for initial_state in explorer.initial_states() {
        pending.push(CombinedState {
            possible_states: explore_until_action(&explorer, initial_state.clone(), sense_index),
            actual_state: initial_state,
        });
    }

    let start = Instant::now();
    let mut processed = 0;

    while let Some(state) = pending.pop() {
        processed += 1;

        if processed % 1 == 0 {
            let duration = start.elapsed();
            println!(
                "States: {} ({:.2} [states/s], [waiting {}])",
                visited.len(),
                (processed as f64) / duration.as_secs_f64(),
                pending.len(),
            )
        }

        if !visited.contains(&state) {
            let transitions = explorer.transitions(&state.actual_state);

            for transition in transitions {
                let action = transition.result_action();

                if action.label_index() == Some(sense_index) {
                    let destinations = explorer.destinations(&state.actual_state, &transition);
                    for destination in destinations {
                        let successor =
                            explorer.successor(&state.actual_state, &transition, &destination);
                        // observation is `sense` + value of sense_climb_rate
                        let observation =
                            successor.get_global_value(&explorer, "global_sense_climb_rate");
                        let mut possible_states = im::HashSet::new();
                        for possible_state in &state.possible_states {
                            // can we make the observation in this state?
                            for possible_transition in explorer.transitions(possible_state) {
                                let possible_action = possible_transition.result_action();
                                if possible_action.label_index() == Some(sense_index) {
                                    for possible_destination in
                                        explorer.destinations(possible_state, &possible_transition)
                                    {
                                        let possible_successor = explorer.successor(
                                            possible_state,
                                            &possible_transition,
                                            &possible_destination,
                                        );
                                        if possible_successor
                                            .get_global_value(&explorer, "global_sense_climb_rate")
                                            == observation
                                        {
                                            possible_states =
                                                possible_states.union(explore_until_action(
                                                    &explorer,
                                                    possible_successor,
                                                    sense_index,
                                                ));
                                        }
                                    }
                                }
                            }
                        }

                        pending.push(CombinedState {
                            actual_state: successor,
                            possible_states: possible_states.clone(),
                        });
                    }
                } else {
                    let destinations = explorer.destinations(&state.actual_state, &transition);
                    for destination in destinations {
                        let successor =
                            explorer.successor(&state.actual_state, &transition, &destination);
                        pending.push(CombinedState {
                            actual_state: successor,
                            possible_states: state.possible_states.clone(),
                        });
                    }
                }
            }

            visited.insert(state);
        }
    }

    let duration = start.elapsed();

    println!("Time elapsed is: {:?}", duration);
    println!("States: {}", visited.len());
    println!(
        "{:.2} [states/s]",
        (visited.len() as f64) / duration.as_secs_f64()
    )
}

fn main() {
    let arguments = Arguments::parse();

    match arguments.command {
        Command::Verify(args) => verify(args),
    }
}
