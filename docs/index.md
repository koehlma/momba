Welcome to Momba's documentation!
=================================

[![PyPi Package](https://img.shields.io/pypi/v/momba.svg?label=latest%20version)](https://pypi.python.org/pypi/momba)
[![Tests](https://img.shields.io/github/workflow/status/koehlma/momba/Pipeline?label=tests)](https://github.com/koehlma/momba/actions)
[![Docs](https://img.shields.io/static/v1?label=docs&message=master&color=blue)](https://koehlma.github.io/momba/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Gitter](https://badges.gitter.im/koehlma/momba.svg)](https://gitter.im/koehlma/momba?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4519376.svg)](https://doi.org/10.5281/zenodo.4519376)

ℹ️ **News:** We are delighted to announce an [upcoming Momba tutorial](https://fm21.momba.dev) at [FM’21]( http://lcs.ios.ac.cn/fm2021/). We'll keep you posted!

*Momba* is a Python framework for dealing with quantitative models centered around the [JANI-model](http://www.jani-spec.org/) interchange format.
Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis.
It provides convenience functions for the modular construction of models effectively turning Python into a syntax-aware macro language for quantitative models.
Momba's built-in exploration engine allows gaining confidence in a model, for instance, by rapidly prototyping a tool for interactive model exploration and visualization, or by connecting it to a testing framework.
Finally, thanks to the JANI-model interchange format, several state-of-the-art model checkers and other tools are readily available for model analysis.

Please cite Momba as follows:

Maximilian A. Köhl, Michaela Klauck, and Holger Hermanns: *Momba: JANI Meets Python*. In: J. F. Groote and K. G. Larsen (eds.) 27th International Conference on Tools and Algorithms for the Construction and Analysis of Systems, TACAS 2021. DOI: [https://doi.org/10.1007/978-3-030-72013-1_23](https://doi.org/10.1007/978-3-030-72013-1_23).


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
Check out the [examples](examples) or read the [user guide](guide) to learn more.

If you aim at a fully reproducible modeling environment, we recommend using [Pipenv](https://pypi.org/project/pipenv/) or [Poetry](https://python-poetry.org/) for dependency management.
We also provide a [GitHub Template](https://github.com/koehlma/momba-pipenv-template) for Pipenv.

## Acknowledgements

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
