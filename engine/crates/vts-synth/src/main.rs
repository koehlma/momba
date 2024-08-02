// Required for hiding iterator types returned by TS traits.
#![feature(impl_trait_in_assoc_type)]

use std::{path::PathBuf, sync::Arc};

use clap::{Args, Parser, Subcommand};

pub mod algorithms;
pub mod cudd;
pub mod domains;
pub mod frontends;
pub mod lattice;
pub mod logic;
pub mod synthesis;
pub mod ts;
pub mod ts_traits;

#[derive(Parser)]
pub struct Arguments {
    model_path: PathBuf,

    feature_model: Option<PathBuf>,

    #[clap(long)]
    steps: Option<usize>,

    #[clap(long)]
    output_path: Option<PathBuf>,

    #[clap(long)]
    unobservable: Vec<Arc<str>>,

    // #[clap(long)]
    // feature: Vec<String>,
    #[clap(long)]
    output_state_spaces: bool,

    #[clap(long)]
    with_lookahead_refinement: bool,

    #[clap(long)]
    without_minimization: bool,

    #[command(flatten)]
    minimization: MinimizationOpts,

    #[clap(long)]
    simulate: bool,
}

#[derive(Args)]
pub struct MinimizationOpts {
    #[clap(long)]
    relax_language: bool,
}

#[derive(Subcommand)]
pub enum Command {
    /// Synthesize a configuration monitor.
    Confmon,
    /// Synthesize a diagnoser.
    Diagnoser,
}

pub fn main() {
    tracing_subscriber::fmt::init();

    let arguments = Arguments::parse();

    synthesis::confmon::synthesize(&arguments);
}
