//! The `diagnose` sub-command.

use clap::Clap;

use momba_explore::model;

#[derive(Clap)]
pub(crate) struct Diagnose {}

impl Diagnose {
    pub fn run(self, network: model::Network) {
        println!("starting diagnosis");
    }
}
