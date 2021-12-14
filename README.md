<p align="center">
  <img src="https://raw.githubusercontent.com/koehlma/momba/master/docs/_static/images/logo_with_text.svg" alt="Momba Logo" width="200px">
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/momba"><img alt="PyPi Package" src="https://img.shields.io/pypi/v/momba.svg?label=latest%20version"></a>
  <a href="https://github.com/koehlma/momba/actions"><img alt="Tests" src="https://img.shields.io/github/workflow/status/koehlma/momba/Pipeline?label=tests"></a>
  <a href="https://koehlma.github.io/momba/"><img alt="Docs" src="https://img.shields.io/static/v1?label=docs&message=master&color=blue"></a>
  <a href="https://github.com/psf/black"><img alt="Code Style: Black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://gitter.im/koehlma/momba?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge"><img alt="Gitter" src="https://badges.gitter.im/koehlma/momba.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.4519376"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.4519376.svg"></a>
</p>

*Momba* is a Python framework for dealing with quantitative models centered around the [JANI-model](http://www.jani-spec.org/) interchange format.
Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis.
It provides convenience functions for the modular construction of models effectively turning Python into a syntax-aware macro language for quantitative models.
Momba's built-in exploration engine allows gaining confidence in a model, for instance, by rapidly prototyping a tool for interactive model exploration and visualization, or by connecting it to a testing framework.
Finally, thanks to the JANI-model interchange format, several state-of-the-art model checkers and other tools are readily available for model analysis.

Please cite Momba as follows:

Maximilian A. Köhl, Michaela Klauck, and Holger Hermanns: *Momba: JANI Meets Python*. In: J. F. Groote and K. G. Larsen (eds.) 27th International Conference on Tools and Algorithms for the Construction and Analysis of Systems, TACAS 2021. DOI: https://doi.org/10.1007/978-3-030-72013-1_23.


## Features

* first-class **import and export** of **JANI models**
* **syntax-aware macros** for the modular construction of models with Python code
* **built-in exploration engine** for PTAs, MDPs and other model types
* interfaces to state-of-the-art model checkers, e.g., [The Modest Toolset](http://www.modestchecker.net/) and [Storm](https://www.stormchecker.org/)
* pythonic and **statically typed** APIs to thinker with formal models
* hassle-free out-of-the-box support for **Windows, Linux, and MacOS**


## Getting Started

Momba is available from the [Python Package Index](https://pypi.org/):
```sh
pip install momba[all]
```
Installing Momba with the `all` feature flag will install all optional dependencies unleashing the full power and all features of Momba.
Check out the [examples](https://koehlma.github.io/momba/examples) or read the [user guide](https://koehlma.github.io/momba/guide) to learn more.

If you aim at a fully reproducible modeling environment, we recommend using [Pipenv](https://pypi.org/project/pipenv/) or [Poetry](https://python-poetry.org/) for dependency management.
We also provide a [GitHub Template](https://github.com/koehlma/momba-pipenv-template) for Pipenv.


## Rust Crates

The exploration engine of Momba is written in [Rust](https://rust-lang.org) levering [PyO3](https://pyo3.rs/) for Python bindings.
In case you are a Rust developer you might find some of the crates in [engine/crates](engine/crates) useful.
In particular, the crate [momba-explore](https://crates.io/crates/momba-explore) allows developing model analysis tools with JANI support in Rust based on Momba's explicit state space exploration engine.
The Rust command line tool [`momba-sidekick`](https://crates.io/crates/momba-sidekick) directly exposes some of this functionality.


## Acknowledgements

This project is partially supported by the ERC Advanced Investigators Grant 695614 ([POWVER](https://powver.org)), by the German Research Foundation (DFG) under grant No. 389792660, as part of [TRR 248](https://perspicuous-computing.science), and by the Key-Area Research and Development Program Grant 2018B010107004 of Guangdong Province.

Thanks to Sarah Sterz for the awesome Momba logo.
