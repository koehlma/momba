//! The `generate` sub-command.

use clap::Clap;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use toml;

use momba_explore::model;

use crate::external;
use crate::robust;

/// Specifies the timing of fault injections.
#[derive(Deserialize, Serialize, Eq, PartialEq, Clone, Debug)]
#[serde(untagged)]
pub(crate) enum InjectionTiming {
    /// Inject a fault after `after_observations` observations have been made.
    AfterObservations { after_observations: usize },
    /// Inject a fault with an exponential rate of `with_rate`.
    WithExpRate { with_rate: NotNan<f64> },
}

/// Specifies whether and how to inject a fault.
#[derive(Deserialize, Serialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct Inject {
    /// The type of the fault to inject.
    fault_type: String,
    /// The timing of the injection.
    timing: InjectionTiming,
}

/// A configuration based on which to generate observations.
#[derive(Deserialize, Serialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct Config {
    /// The clock drift parameter ε.
    clock_drift: NotNan<f64>,
    /// The labeles of the fault types.
    fault_types: HashSet<String>,
    /// The observables and their latency intervals.
    observables: HashMap<String, external::LatencyInterval>,
    /// An optional fault injection description.
    inject: Option<Inject>,
}

#[derive(Clap)]
pub(crate) struct Generate {
    #[clap(about = "The configuration file to use for generating observations")]
    config: String,

    #[clap(about = "The file to write the observations to")]
    observations: String,

    #[clap(about = "The file to write the event information to")]
    events: Option<String>,
}

impl Generate {
    pub fn run(self, network: model::Network) {
        let config: Config = toml::from_str(
            &fs::read_to_string(Path::new(&self.config))
                .expect("Unable to read configuration file!"),
        )
        .expect("Unable to parse configuration file!");
    }
}
