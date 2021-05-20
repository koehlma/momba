//! The sub-commands of the diagnosis tool.

use clap::Clap;

pub(crate) mod diagnose;
pub(crate) mod generate;

#[derive(Clap)]
pub(crate) enum Command {
    #[clap(about = "Diagnoses the system based on provided observations")]
    Diagnose(diagnose::Diagnose),
    #[clap(about = "Generates observations by simulating the model")]
    Generate(generate::Generate),
}
