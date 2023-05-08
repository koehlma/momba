use std::{error::Error, path::PathBuf, sync::Arc, time::Instant};

use clap::Parser;
use momba_explicit::{
    compiler::{compile_model, Options, StateLayout},
    count_states, count_states_concurrent,
    params::Params,
};
use momba_model::{models::Model, values::Value};

#[derive(Parser)]
pub struct Arguments {
    model_path: PathBuf,
    #[clap(long)]
    params: Option<String>,
    #[clap(long)]
    threads: Option<usize>,
}

pub fn parse_params(string: &str) -> Params {
    let mut params = Params::new();
    for part in string.split(",") {
        if part.is_empty() {
            continue;
        }
        let Some((name, value)) = part.split_once("=") else {
            panic!("Invalid params!");
        };
        let value = match value {
            "true" | "True" => Value::Bool(true),
            "false" | "False" => Value::Bool(false),
            value @ _ => {
                if let Ok(value) = value.parse::<i64>() {
                    Value::Int(value)
                } else if let Ok(value) = value.parse::<f64>() {
                    Value::Float(value)
                } else {
                    panic!()
                }
            }
        };
        params.set(name, value);
    }
    params
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let args = Arguments::parse();

    let model: Model = serde_json::from_str(&std::fs::read_to_string(&args.model_path)?)?;
    let params = parse_params(args.params.as_ref().map(String::as_str).unwrap_or(""));

    if let Some(threads) = args.threads {
        count_states_concurrent(&model, &params, threads)?;
    } else {
        count_states(&model, &params)?;
    }

    Ok(())
}
