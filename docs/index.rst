Welcome to Momba's documentation!
=================================

.. warning::
    **Momba is still in its early stages of development.
    Please expect things to break.
    The API is unstable and might change without further notice and deprecation period.
    Some parts of this documentation are wishful thinking describing features not yet implemented.**

*Momba* is a Python library for working with quantitative models.
Momba's core modeling formalism are networks of interacting *stochastic hybrid automata* (SHA) as per the `JANI specification`_.
Momba aims to be a platform for prototyping and the development of new techniques and algorithms for the analysis of quantitative models.
For the time being, Momba does not aim to be a model checker itself.
Instead, Momba relies on external tools for model checking via the JANI interaction protocol.
In particular, Momba works well with `The Modest Toolset`__ and `EPMC`__.

__ http://www.modestchecker.net/
__ https://github.com/ISCAS-PMC/ePMC

.. _JANI specification: http://www.jani-spec.org/


Getting Started
---------------
Momba requires Python 3.8 or newer. To install Momba from `PyPi`__ simply run:

.. code-block:: bash

    pip install momba

A good way of getting started with Momba is to read this documentation, in particular, the section :ref:`Momba models` which describes how quantitative models are represented in Momba.
Also, check out `the examples`__.

__ https://pypi.org/
__ https://dgit.cs.uni-saarland.de/koehlma/momba/tree/master/examples


Development
-----------
Momba uses `Poetry`_ for dependency management. Run :code:`poetry install` to create a virtual environment in :code:`.venv` containing all the dependencies needed for development.
Momba comes with a configuration for `Visual Studio Code`_ which requires the virtual environment to be present in :code:`.venv` and enables linting and type checking.
Before *pushing* ensure that :code:`poetry run tox` runs without any problems.
This will run the tests and perform type checking as well as linting.

.. _`Poetry`: https://python-poetry.org/
.. _`Visual Studio Code`: https://code.visualstudio.com/

To build this documentation run :code:`poetry run sphinx-build docs build/docs`.


Contents
--------

.. toctree::
    :maxdepth: 2

    model/index
    moml/index
    explore/index
    analysis/index
    external/index
