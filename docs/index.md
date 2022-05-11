Welcome to Momba's documentation!
=================================

[![PyPi Package](https://img.shields.io/pypi/v/momba.svg?label=latest%20version)](https://pypi.python.org/pypi/momba)
[![Tests](https://img.shields.io/github/workflow/status/koehlma/momba/Pipeline?label=tests)](https://github.com/koehlma/momba/actions)
[![Docs](https://img.shields.io/static/v1?label=docs&message=master&color=blue)](https://koehlma.github.io/momba/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Gitter](https://badges.gitter.im/koehlma/momba.svg)](https://gitter.im/koehlma/momba?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4519376.svg)](https://doi.org/10.5281/zenodo.4519376)

*Momba* is a Python framework for dealing with quantitative models centered around the [JANI-model](http://www.jani-spec.org/) interchange format.
Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis.
It provides convenience functions for the modular construction of models effectively turning Python into a syntax-aware macro language for quantitative models.
Momba's built-in exploration engine allows gaining confidence in a model, for instance, by rapidly prototyping a tool for interactive model exploration and visualization, or by connecting it to a testing framework.
Finally, thanks to the JANI-model interchange format, several state-of-the-art model checkers and other tools are readily available for model analysis.

For academic publications, please cite Momba as follows:

Maximilian A. Köhl, Michaela Klauck, and Holger Hermanns: *Momba: JANI Meets Python*. In: J. F. Groote and K. G. Larsen (eds.) 27th International Conference on Tools and Algorithms for the Construction and Analysis of Systems, TACAS 2021. DOI: https://doi.org/10.1007/978-3-030-72013-1_23.

In case you made anything with Momba or plan to do so, we would highly appreciate if you let us know about your exciting project by [opening a discussion](https://github.com/koehlma/momba/discussions/new?category=show-and-tell) or dropping us a message. 🙌


## ✨ Features

* first-class **import and export** of **JANI models**
* **syntax-aware macros** for the modular construction of models with Python code
* **built-in exploration engine** for PTAs, MDPs and other model types
* interfaces to state-of-the-art model checkers, e.g., the [Modest Toolset](http://www.modestchecker.net/) and [Storm](https://www.stormchecker.org/)
* **an [OpenAI Gym](https://gym.openai.com) compatible interface** for training agents on formal models
* pythonic and **statically typed** APIs to tinker with formal models
* hassle-free out-of-the-box support for **Windows, Linux, and MacOS**


## 🚀 Getting Started

Momba is available from the [Python Package Index](https://pypi.org/):
```sh
pip install momba[all]
```
Installing Momba with the `all` feature flag will install all optional dependencies unleashing the full power and all features of Momba.
Check out the [examples](https://koehlma.github.io/momba/examples) or read the [user guide](https://koehlma.github.io/momba/guide) to learn more.

If you aim at a fully reproducible modeling environment, we recommend using [Pipenv](https://pypi.org/project/pipenv/) or [Poetry](https://python-poetry.org/) for dependency management.
We also provide a [GitHub Template](https://github.com/koehlma/momba-pipenv-template) for Pipenv.


## 🏗 Contributing

We welcome all kinds of contributions!

For minor changes and bug fixes feel free to simply open a pull request. For major changes impacting the overall design of Momba, please first [start a discussion](https://github.com/koehlma/momba/discussions/new?category=ideas) outlining your idea.

To get you started, we provide a [development container for VS Code](https://code.visualstudio.com/docs/remote/containers) containing everything you need for development. The easiest way to get up and running is by clicking on the following badge:

[![VS Code: Open in Container](https://img.shields.io/static/v1?label=VS%20Code&message=Open%20in%20Container&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/koehlma/momba.git)

Opening the link in VS Code will clone this repository into its own Docker volume and then start the provided development container inside VS Code so you are ready to start coding.

By submitting a PR, you agree to license your contributions under MIT.


## 🦀 Rust Crates

The exploration engine of Momba is written in [Rust](https://rust-lang.org) levering [PyO3](https://pyo3.rs/) for Python bindings.
In case you are a Rust developer you might find some of the crates in [engine/crates](engine/crates) useful.
In particular, the crate [momba-explore](https://crates.io/crates/momba-explore) allows developing model analysis tools with JANI support in Rust based on Momba's explicit state space exploration engine.
The Rust command line tool [`momba-sidekick`](https://crates.io/crates/momba-sidekick) directly exposes some of this functionality.


## 🙏 Acknowledgements

This project is partially supported by the ERC Advanced Investigators Grant 695614 ([POWVER](https://powver.org)), by the German Research Foundation (DFG) under grant No. 389792660, as part of [TRR 248](https://perspicuous-computing.science), and by the Key-Area Research and Development Program Grant 2018B010107004 of Guangdong Province.

Thanks to Sarah Sterz for the awesome Momba logo.


```{toctree}
:hidden:

guide/index
examples/index
reference/index
incubator/index
```

```{toctree}
:caption: Tools
:hidden:

tools/modest
tools/storm
```

```{toctree}
:caption: Features
:hidden:

gym/index
```

```{toctree}
:caption: Development
:hidden:

contributing/index
GitHub Repository <https://github.com/koehlma/momba>
```
