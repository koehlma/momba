use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashSet;
use std::time::Instant;

use momba_explore::compiled;
use momba_explore::model;
use momba_explore::values;

#[pyclass]
struct CompiledModel {
    network: compiled::CompiledNetwork,
}

#[pymethods]
impl CompiledModel {
    #[new]
    fn new(json_representation: &str) -> Self {
        let network: model::Network =
            serde_json::from_str(json_representation).expect("Error while reading model file!");

        CompiledModel {
            network: network.compile(),
        }
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

        println!("Time elapsed is: {:?}", duration);
        println!("States: {}", visited.len());
        println!("Transitions: {}", count_transitions);
        println!(
            "{:.2} [states/s]",
            (visited.len() as f64) / duration.as_secs_f64()
        );

        visited.len()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn momba_engine(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CompiledModel>()?;

    Ok(())
}
