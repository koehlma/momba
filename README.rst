Momba
=====

|pypi| |build| |coverage| |docs| |black|

**Momba is still in its early stages of development.
Please expect things to break.
The API is unstable and might change without further notice and deprecation period.**

*Momba* is a Python library for working with quantitative models.
Momba's core modeling formalism are networks of interacting *stochastic hybrid automata* (SHA) as per the `JANI specification`_.
Momba aims to be a platform for prototyping and the development of new techniques and algorithms for the analysis of quantitative models.
For the time being, Momba does not aim to be a model checker itself.
Instead, Momba relies on external tools for model checking via the JANI interaction protocol. In particular, Momba works well with `The Modest Toolset`__ and `EPMC`__.

__ http://www.modestchecker.net/
__ https://github.com/ISCAS-PMC/ePMC

.. _JANI specification: http://www.jani-spec.org/


How to use Momba?
-----------------
Please read `the documentation`_.

.. _the documentation: https://depend.cs.uni-saarland.de/~koehl/momba/


.. |pypi| image:: https://img.shields.io/pypi/v/momba.svg?label=latest%20version
    :target: https://pypi.python.org/pypi/momba

.. |build| image:: https://dgit.cs.uni-saarland.de/koehlma/momba/badges/master/pipeline.svg
    :target: https://dgit.cs.uni-saarland.de/koehlma/momba/pipelines

.. |coverage| image:: https://dgit.cs.uni-saarland.de/koehlma/momba/badges/master/coverage.svg
    :target: https://dgit.cs.uni-saarland.de/koehlma/momba/pipelines

.. |docs| image:: https://img.shields.io/static/v1?label=docs&message=master&color=blue
    :target: https://depend.cs.uni-saarland.de/~koehl/momba/

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
