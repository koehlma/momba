# Momba

[![PyPi Package](https://img.shields.io/pypi/v/momba.svg?label=latest%20version)](https://pypi.python.org/pypi/momba)
[![GitLab Pipeline](https://dgit.cs.uni-saarland.de/koehlma/momba/badges/master/pipeline.svg)](https://dgit.cs.uni-saarland.de/koehlma/momba/pipelines)
[![Coverage](https://dgit.cs.uni-saarland.de/koehlma/momba/badges/master/coverage.svg)](https://dgit.cs.uni-saarland.de/koehlma/momba/pipelines)
[![Docs](https://img.shields.io/static/v1?label=docs&message=master&color=blue)](https://depend.cs.uni-saarland.de/~koehl/momba/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


> :warning: **Momba is still in its early stage of development. Please expect things to break. The API is mostly unstable and might change without further notice and deprecation period.**

*Momba* is a Python library for working with quantitative models.
Momba's core modeling formalism are networks of interacting *Stochastic Hybrid Automata* (SHA) as per the [JANI Specification](http://www.jani-spec.org/).
Momba aims to be a platform for prototyping and the development of new techniques and algorithms for the analysis of quantitative models.
Momba does explicitly not aim to be a model checker.
Instead, Momba relies on external tools for model checking via JANI.
Model checkers supporting JANI are for instance [The Modest Toolset](http://www.modestchecker.net/) or [Storm](https://www.stormchecker.org/).


## Features

* import and export models from and to [JANI](http://www.jani-spec.org/)
* build JANI models using Python as a meta-programming language
* simulate any PTA, TA, MDP, LTS, or DTMC model


## How to use Momba?
Please read [the documentation](https://depend.cs.uni-saarland.de/~koehl/momba/).