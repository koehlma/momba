# Momba Simulator

<!-- [![crate](???)](???) -->
<!-- [![documentation](???)](???) -->

A library that provides Simulation capabilities to *Momba*.
This crates contains a implementation of sequential and parallel 
*Statistical Model Checking* for DTMC[[1]] and an implementation of SPRT[[3]]. 
For MDPs[[2]], the undeterminism can be resolved by DSMC[[4]] taken a JSON file
with the layers of the networks, or by using a Uniform Oracle.

The code is designed to be used to explore and learn mehtods on Markov
Decision Processes, and can be easily used to explore new oracles.


[1]: https://en.wikipedia.org/wiki/Discrete-time_Markov_chain
[2]: https://en.wikipedia.org/wiki/Markov_decision_process
[3]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
[4]: http://dx.doi.org/10.22028/D291-36816

## Commands
- `info <model>.json`: gives information about the model
- `smc <model>.json <prop>.json`: runs SMC on the model, using a uniform oracle. If its DTMC, just simulates runs.
- `par-smc <model>.json <prop>.json`: runs SMC parallel, using a uniform oracle.
    - Optionals: `-n <number of threads>`
- `nn <model>.json <prop>.json <nn>.json`: runs SMC using the nn specified in the json.
    - Optionals: `-n <number of threads>`; `-i <controlled instance name>`
    - If not specified the instance name, uses the one with index 0.

## Features
- SMC for DTMC, sequential and in parallel.
- SPRT implementation for DTMC. 
- DSMC using a NN json file, or a custom Oracle of your choice.

## Future work

- Generalization of oracle funtion.
- Dead predicates.
- Implementation of training for the NN.
- Action resolver by label.
- Support for expected-like properties.
- Use of local variables for the training of the model.