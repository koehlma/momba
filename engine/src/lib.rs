use momba_explore::explore::initial_states;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use momba_explore::compiled;
use momba_explore::explore;
use momba_explore::model;
use momba_explore::values;

#[pyclass]
struct CompiledModel {
    network: Arc<compiled::CompiledNetwork>,
}

#[pyclass]
struct State {
    network: Arc<compiled::CompiledNetwork>,
    bare_state: explore::BareState,
}

#[pymethods]
impl State {
    pub fn as_json(&self) -> String {
        serde_json::to_string(&self.bare_state.state(&self.network.network)).unwrap()
    }

    pub fn successors(&self) -> Vec<State> {
        let compiled_state = self.bare_state.clone().into_compiled(self.network.borrow());
        let mut successors = Vec::new();
        for transition in compiled_state.transitions() {
            for destination in compiled_state.execute(transition) {
                successors.push(State {
                    network: self.network.clone(),
                    bare_state: destination.state.into(),
                })
            }
        }
        successors
    }
}

#[pymethods]
impl CompiledModel {
    #[new]
    fn new(json_representation: &str) -> Self {
        let network: model::Network =
            serde_json::from_str(json_representation).expect("Error while reading model file!");

        CompiledModel {
            network: Arc::new(network.compile()),
        }
    }

    fn initial_states(&self) -> Vec<State> {
        initial_states(self.network.borrow())
            .unwrap()
            .into_iter()
            .map(|bare_state| State {
                network: self.network.clone(),
                bare_state: bare_state,
            })
            .collect()
    }

    fn count_states(&self) -> usize {
        let start = Instant::now();

        println!("Exploring...");

        let mut visited: HashSet<(Box<[values::Value]>, Box<[usize]>)> = HashSet::new();
        let mut pending: Vec<compiled::CompiledState> = self.network.initial_states().into();

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
                            network: &self.network,
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

        // println!("Time elapsed is: {:?}", duration);
        // println!("States: {}", visited.len());
        // println!("Transitions: {}", count_transitions);
        // println!(
        //     "{:.2} [states/s]",
        //     (visited.len() as f64) / duration.as_secs_f64()
        // );

        visited.len()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn momba_engine(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CompiledModel>()?;

    Ok(())
}
