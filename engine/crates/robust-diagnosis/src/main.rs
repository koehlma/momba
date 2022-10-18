// use clap::Clap;
// use serde_json;
// use std::fs;
// use std::io::BufReader;
// use std::path::Path;

// pub mod commands;
// pub mod external;
// pub mod robust;

// use commands::Command;

// #[derive(Clap)]
// #[clap(
//     version = "0.1.0",
//     about = "Robust diagnosis of real-time systems",
//     author = "Maximilian A. Köhl <koehl@cs.uni-saarland.de>"
// )]
// struct Arguments {
//     #[clap(about = "The MombaIR model to use")]
//     model: String,

//     #[clap(subcommand)]
//     command: Command,
// }

// fn main() {
//     let arguments = Arguments::parse();
//     let network = serde_json::from_str(
//         &fs::read_to_string(Path::new(&arguments.model)).expect("Unable to read model file!"),
//     )
//     .expect("Unable to parse model file!");
//     match arguments.command {
//         Command::Diagnose(command) => command.run(network),
//         Command::Generate(command) => command.run(network),
//     }
// }

pub mod robust;

pub mod external;

use rand_distr::Distribution;

use std::convert::TryInto;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use ordered_float::NotNan;

use clap::Clap;

use momba_explore::model;

mod parameters;

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
struct ExternalObservation {
    time: NotNan<f64>,
    label: String,
    discrete_time: usize,
    arguments: Vec<model::Value>,
    base_latency: NotNan<f64>,
    jitter_bound: NotNan<f64>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
struct ExternalEvent {
    discrete_time: usize,
    continuous_time: NotNan<f64>,
    label: String,
    arguments: Vec<model::Value>,
}

#[derive(Clap)]
#[clap(version = "0.1.0", about = "Robust diagnosis of real-time systems.")]
struct Arguments {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Clap)]
enum Command {
    #[clap(about = "Diagnoses a system based on observations and a model")]
    Diagnose(DiagnoseCommand),

    #[clap(about = "Generates a trace of observations by simulation")]
    Generate(GenerateCommand),
}

#[derive(Clap)]
struct DiagnoseCommand {
    #[clap(about = "The MombaIR model used for diagnosis")]
    model: String,

    #[clap(about = "The parameters for the diagnosis algorithm")]
    parameters: String,

    #[clap(about = "The observations in JSON format")]
    observations: String,

    #[clap(about = "File to write the diagnosis results to")]
    output: String,

    #[clap(long, about = "History bound")]
    history_bound: Option<usize>,
}

#[derive(Clap)]
struct GenerateCommand {
    #[clap(about = "The MombaIR model used for generator")]
    model: String,

    #[clap(about = "The parameters for the diagnosis algorithm")]
    diagnose_parameters: String,

    #[clap(about = "The parameters for the generator")]
    generate_parameters: String,

    #[clap(about = "File to write the observations to")]
    observations: String,

    #[clap(about = "File to write the events to")]
    events: String,

    #[clap(about = "Time to run the simulation for")]
    simulation_time: NotNan<f64>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
struct ResultRecord {
    consistent: bool,
    states: usize,
    prefixes: usize,
    fault_necessary: bool,
    fault_possible: bool,
    duration: f64,
}

fn main() {
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Diagnose(command) => {
            let network: model::Network = serde_json::from_reader(BufReader::new(
                File::open(Path::new(&command.model)).expect("Unable to open model file!"),
            ))
            .expect("Error while reading model file!");

            let observations: Vec<ExternalObservation> = serde_json::from_reader(BufReader::new(
                File::open(Path::new(&command.observations))
                    .expect("Unable to open observations file!"),
            ))
            .expect("Error while reading observations!");

            let parameters = parameters::load_diagnose_parameters(command.parameters)
                .expect("Error while loading diagnose parameters!");

            println!("Found {} observations.", observations.len());

            println!("{:?}", parameters);

            println!("Imprecisions:");
            println!("  Clock drift: {}", parameters.clock_drift);

            let min_latency = parameters.min_latency();

            let max_latency = parameters.max_latency();

            println!("  Minimal latency: {}", min_latency);
            println!("  Maximal latency: {}", max_latency);

            let start = Instant::now();

            let mut diagnoser = robust::Diagnoser::new(
                robust::observer::Imprecisions::new(
                    // allow for some extra slack due to floating point imprecisions
                    parameters.clock_drift + 0.001,
                    max_latency + 0.001,
                    min_latency - 0.001,
                ),
                network.clone(),
                parameters
                    .observables
                    .keys()
                    .map(|observable| {
                        network
                            .declarations
                            .action_labels
                            .get_index_of(observable)
                            .unwrap()
                    })
                    .collect(),
                parameters
                    .fault_types
                    .iter()
                    .map(|fault| {
                        network
                            .declarations
                            .action_labels
                            .get_index_of(fault)
                            .unwrap()
                    })
                    .collect(),
                command.history_bound,
            );

            println!("{:?}", diagnoser.result());
            println!(
                "{:?}",
                diagnoser
                    .explore_counter
                    .fetch_add(0, std::sync::atomic::Ordering::AcqRel)
            );

            let mut records = Vec::new();

            for (index, observation) in observations.into_iter().enumerate() {
                println!("Pushing observation {}.", index);
                println!("{:?}", observation);
                diagnoser.push(robust::observer::Observation::new(
                    observation.time,
                    momba_explore::LabeledAction::new_with_network(
                        &network,
                        &observation.label,
                        observation.arguments.into(),
                    ),
                    observation.base_latency,
                    // allow for some extra slack due to floating point imprecisions
                    observation.jitter_bound + 0.001,
                ));
                let result = diagnoser.result();
                println!("{:?}", result);
                records.push({
                    ResultRecord {
                        consistent: result.consistent,
                        states: result.states,
                        prefixes: result.prefixes,
                        fault_necessary: result.fault_necessary,
                        fault_possible: result.fault_possible,
                        duration: start.elapsed().as_secs_f64(),
                    }
                });
            }

            let duration = start.elapsed();

            println!("Time elapsed is: {:?}", duration);

            let active_prefixes = diagnoser.active_prefixes();

            println!("Active prefixes: {}", active_prefixes.len());

            for prefix in active_prefixes {
                println!("{:?}", prefix);
            }

            serde_json::to_writer(
                BufWriter::new(
                    File::create(Path::new(&command.output))
                        .expect("Unable to create output file!"),
                ),
                &records,
            )
            .unwrap();
        }
        Command::Generate(command) => {
            let network: model::Network = serde_json::from_reader(BufReader::new(
                File::open(Path::new(&command.model)).expect("Unable to open model file!"),
            ))
            .expect("Error while reading model file!");

            let diagnose_parameters =
                parameters::load_diagnose_parameters(command.diagnose_parameters)
                    .expect("Error loading diagnose parameters!");

            let generate_parameters =
                parameters::load_generate_parameters(command.generate_parameters)
                    .expect("Error loading generate parameters!");

            let inject_timing = match generate_parameters.inject.typ {
                parameters::InjectionType::AfterObservations { observations } => {
                    robust::generate::InjectionTiming::AfterObservations(observations)
                }
                parameters::InjectionType::WithExpRate { rate } => {
                    let exp = rand_distr::Exp::new(rate.into_inner()).unwrap();
                    let time = exp.sample(&mut rand::thread_rng());
                    robust::generate::InjectionTiming::AfterTime(time.try_into().unwrap())
                }
            };

            let generator = robust::generate::Generator::new(
                network.clone(),
                robust::generate::Inject {
                    label_index: network
                        .declarations
                        .action_labels
                        .get_index_of(&generate_parameters.inject.label)
                        .unwrap(),
                    timing: inject_timing,
                },
                diagnose_parameters
                    .observables
                    .keys()
                    .map(|observable| {
                        network
                            .declarations
                            .action_labels
                            .get_index_of(observable)
                            .unwrap()
                    })
                    .collect(),
                diagnose_parameters
                    .fault_types
                    .iter()
                    .map(|fault| {
                        network
                            .declarations
                            .action_labels
                            .get_index_of(fault)
                            .unwrap()
                    })
                    .collect(),
                robust::observer::Imprecisions::new(
                    diagnose_parameters.clock_drift,
                    diagnose_parameters.min_latency(),
                    diagnose_parameters.max_latency(),
                ),
                diagnose_parameters
                    .observables
                    .iter()
                    .map(|(label, timing)| {
                        (
                            network
                                .declarations
                                .action_labels
                                .get_index_of(label)
                                .unwrap(),
                            robust::generate::Timing {
                                base_latency: timing.base_latency,
                                jitter_bound: timing.jitter_bound,
                            },
                        )
                    })
                    .collect(),
            );

            let result = generator.generate(command.simulation_time);

            let observations: Vec<_> = result
                .observations
                .into_iter()
                .map(|observation| {
                    let label = network
                        .declarations
                        .action_labels
                        .get_index(observation.event.action.label_index())
                        .unwrap()
                        .0;
                    let timing = diagnose_parameters.observables.get(label).unwrap();
                    ExternalObservation {
                        time: observation.time,
                        discrete_time: observation.event.time.discrete,
                        label: label.clone(),
                        arguments: observation.event.action.arguments().to_vec(),
                        base_latency: timing.base_latency,
                        jitter_bound: timing.jitter_bound,
                    }
                })
                .collect();

            let events: Vec<_> = result
                .events
                .into_iter()
                .map(|event| {
                    let label = network
                        .declarations
                        .action_labels
                        .get_index(event.action.label_index())
                        .unwrap()
                        .0;
                    ExternalEvent {
                        continuous_time: event.time.continuous,
                        discrete_time: event.time.discrete,
                        label: label.clone(),
                        arguments: event.action.arguments().to_vec(),
                    }
                })
                .collect();

            serde_json::to_writer(
                BufWriter::new(
                    File::create(Path::new(&command.observations))
                        .expect("Unable to create output file!"),
                ),
                &observations,
            )
            .unwrap();

            serde_json::to_writer(
                BufWriter::new(
                    File::create(Path::new(&command.events))
                        .expect("Unable to create output file!"),
                ),
                &events,
            )
            .unwrap();
        }
    }
}
