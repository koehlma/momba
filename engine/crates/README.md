# Momba Engine Crates :package:

A collection of individually useful Rust crates developed for Momba.

- :package: [`clock-zones`](clock-zones) is a crate for handling [clock zones](https://link.springer.com/chapter/10.1007/978-3-540-27755-2_3) representing sets of clock constraints.
- :package: [`momba-explore`](momba-explore) provides a state space exploration engine for MDPs and PTAs.
- :package: [`momba-sidekick`](momba-sidekick) is a command line tool directly exposing some model related functionality.

The Python package [`momba_engine`](https://pypi.org/project/momba_engine/) provides Python bindings to these crates exposed as part of Momba's API.
There is no need to use these crates directly except if you would like to use them to develop your own tools in Rust.
