# Momba

[![PyPi Package](https://img.shields.io/pypi/v/momba.svg?label=latest%20version)](https://pypi.python.org/pypi/momba)
[![Basic Checks](https://img.shields.io/github/workflow/status/koehlma/momba/Basic%20Checks?label=basic%20checks)](https://github.com/koehlma/momba/actions)
[![Tests](https://img.shields.io/github/workflow/status/koehlma/momba/Run%20Tests?label=tests)](https://github.com/koehlma/momba/actions)
[![Docs](https://img.shields.io/static/v1?label=docs&message=master&color=blue)](https://koehlma.github.io/momba/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Momba* is a Python framework for dealing with quantitative models centered around the [JANI-model](http://www.jani-spec.org/) interchange format.
Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis.
It provides convenience functions for the modular constructions of models effectively turning Python into a syntax-aware macro language for quantitative models.
Momba's built-in simulator allows gaining confidence in a model, for instance, by rapidly prototyping a tool for interactive model exploration and visualization, or by connecting it to a testing framework.
Finally, thanks to the JANI-model interchange format, several state-of-the-art model checkers and other tools are readily available for analysis.


## Features

* first-class **import and export** of **JANI models**
* **syntax-aware macros** for the modular constructions of models with Python code
* **built-in simulator** for PTAs, MDPs and other model types
* interfaces to state-of-the-art model checkers, e.g., [The Modest Toolset](http://www.modestchecker.net/) and [Storm](https://www.stormchecker.org/)
* pythonic and statically typed APIs to thinker with formal models


## Getting Started

Momba is available from the [Python Package Index](https://pypi.org/):
```sh
pip install momba
```
Check out the [examples](./examples) or read the [documentation](https://koehlma.github.io/momba/) to learn how to use Momba.


## Why?

The idea to harvest a general purpose programming environment for formal modelling is not new at all.
For instance, the [SVL language](https://link.springer.com/chapter/10.1007/0-306-47003-9_24) combines the power of process algebraic modelling with the power of the bourne shell.
Many formal modelling tools also already provide Python bindings, e.g., [Storm](https://moves-rwth.github.io/stormpy/) and [Spot](https://spot.lrde.epita.fr/).
Momba tries not to be yet another incarnation of these ideas.
While the construction of formal models clearly is an integral part of Momba, Momba is more than just a framework for constructing models with the help of Python.
Most importantly, it also provides features to work with these models such as a simulator or an interface to different model checking tools.
At the same time, it is not just a binding to an API developed for another language, like C++.
Momba is tool-agnostic and aims to provide a pythonic interface for dealing with formal models while leveraging existing tools.
Momba covers the whole process from model creation through validation to analysis.
To this end, it is centered around the well-entrenched JANI-model interchange format.
