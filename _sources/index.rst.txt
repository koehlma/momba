Welcome to Momba's documentation!
=================================

Momba is a Python framework for dealing with quantitative models centered around the `JANI-model`_ interchange format. Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis. It provides convenience functions for the modular constructions of models effectively turning Python into a syntax-aware macro language for quantitative models. Momba's built-in simulator allows gaining confidence in a model, for instance, by rapidly prototyping a tool for interactive model exploration and visualization, or by connecting it to a testing framework. Finally, thanks to the JANI-model interchange format, several state-of-the-art model checkers and other tools are readily available for analysis.

__ http://www.modestchecker.net/
__ https://github.com/ISCAS-PMC/ePMC

.. _JANI-model: http://www.jani-spec.org/


Getting Started
---------------
Momba requires Python 3.8 or newer. To install Momba from `PyPi`_ simply run:

.. code-block:: bash

    pip install momba

A good way of getting started with Momba is to read this documentation, in particular, the section :ref:`Momba models` which describes how quantitative models are represented in Momba.
Also, check out `the examples`_.

.. _`PyPi`: https://pypi.org/
.. _`the examples`: https://dgit.cs.uni-saarland.de/koehlma/momba/tree/master/examples


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
