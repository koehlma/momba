# Momba Simulator

<!-- [![crate](???)](???) -->
<!-- [![documentation](???)](???) -->

A library that provides Simulation capabilities to *Momba*.
This crates contains a implementation of sequential and parallel 
*Statistical Model Checking* for DTMC[[1]] and an implementation of the SPRT[[3]] algorithm. 
For MDPs[[2]], the nondeterminism can be resolved by DSMC[[4]] taken a JSON file
with the layers of the networks, or by using a Uniform Oracle.

The code is designed to be used to explore and learn mehtods on Markov
Decision Processes, and can be easily used to explore new oracles.


[1]: https://en.wikipedia.org/wiki/Discrete-time_Markov_chain
[2]: https://en.wikipedia.org/wiki/Markov_decision_process
[3]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
[4]: http://dx.doi.org/10.22028/D291-36816

## Commands
- `python3 -m momba.engine translate <model>.jani <output_folder>`: tranlates the JANI model into the output folder.
    - Writes JSON files, one for the model and one for each of the properties on the JANI file.
- `cargo run --release --bin momba-simulator <run_option> --help`: gives help for running the specified option.
- `cargo run --release --bin momba-simulator info <model>.json`: gives information about the model.
- `cargo run --release --bin momba-simulator smc <model>.json <prop>.json`: runs SMC on the model, using a uniform oracle. If its DTMC, just simulates runs.
- `cargo run --release --bin momba-simulator par-smc <model>.json <prop>.json`: runs SMC parallel, using a uniform oracle.
    - Optionals: `-n <number of threads>`
- `cargo run --release --bin momba-simulator dsmc <model>.json <prop>.json <nn>.json`: runs SMC using the nn specified in the json.
    - Optionals: `-n <number of threads>`; `-i <controlled instance name>`
    - If not specified the instance name, uses the one with index 0.
- `cargo run --release --bin momba-simulator sched-sampler <model>.json <prop>.json <amount_of_schedulers>`: runs the sampling for the model, with the specified number of tries.
    - Optionals: `-n <number of threads>`



## Features
- Okamoto bound for amount of runs needed.
- SMC for DTMC, sequential and in parallel.
- SPRT implementation for DTMC. 
- DSMC using a NN json file as the oracle.
- Scheduler Sampling with hashing functions for MDPs.

## Future work

- Read and evaluate multiple properties on one run.
- Support other types of time beyond Float64Zone.
- Other simulation bounds.
- Smart sampling of schedulers.
- Generalization of oracle function.
- Dead predicates.
- Implementation of training for the NN.
- Action resolver by label.
- Support for expected-like properties.