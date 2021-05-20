use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use anyhow::Result;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct Timing {
    pub(crate) base_latency: NotNan<f64>,
    pub(crate) jitter_bound: NotNan<f64>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct DiagnoseParameters {
    pub(crate) clock_drift: NotNan<f64>,
    pub(crate) fault_types: Vec<String>,
    pub(crate) observables: HashMap<String, Timing>,
}

impl DiagnoseParameters {
    pub(crate) fn min_latency(&self) -> NotNan<f64> {
        return self
            .observables
            .values()
            .map(|timing| timing.base_latency - timing.jitter_bound)
            .min()
            .unwrap();
    }

    pub(crate) fn max_latency(&self) -> NotNan<f64> {
        return self
            .observables
            .values()
            .map(|timing| timing.base_latency + timing.jitter_bound)
            .max()
            .unwrap();
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum InjectionType {
    AfterObservations { observations: usize },
    WithExpRate { rate: NotNan<f64> },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct Inject {
    pub(crate) label: String,

    #[serde(flatten)]
    pub(crate) typ: InjectionType,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub(crate) struct GenerateParameters {
    pub(crate) inject: Inject,
}

pub(crate) fn load_diagnose_parameters<P: AsRef<Path>>(path: P) -> Result<DiagnoseParameters> {
    Ok(toml::from_str(&fs::read_to_string(path)?)?)
}

pub(crate) fn load_generate_parameters<P: AsRef<Path>>(path: P) -> Result<GenerateParameters> {
    Ok(toml::from_str(&fs::read_to_string(path)?)?)
}
