.. _Momba models:

.. currentmodule:: momba.model


Momba Models
============

The package :mod:`momba.model` contains the core data-structures for the representation of quantitative models.
Essentially, a *Momba model* is a network of interacting *stochastic hybrid automata* (SHA) as per the `JANI specification`_.
At the heart of every model is a *modeling context* represented by a :class:`Context`-object.
The modeling context specifies the model type (see :class:`ModelType` for an overview) and allows the definition of constants, variables, automata, and automata networks.


Individual SHAs are implemented by :class:`momba.model.Automaton`.
SHAs are built from a finite set of *locations* connected via *edges*.
SHAs, as we understand them, are  models with *variables*.
A *valuation* maps variables to values.
Each edge leads from a single *source location* via a *guard* and an *action* to a *symbolic probability distribution* over *variable assignments* and *successor locations*.


.. toctree::
   :maxdepth: 2

   expressions


.. _JANI specification: http://www.jani-spec.org/


.. autoclass:: momba.model.Context
    :members:
    :member-order: bysource


.. autoclass:: momba.model.Scope
    :members:
    :member-order: bysource


.. autoclass:: momba.model.ModelType
    :members:
    :undoc-members:
    :member-order: bysource



.. autoclass:: momba.model.Location
    :members:
    :member-order: bysource


.. autoclass:: momba.model.Automaton
    :members:
    :member-order: bysource


.. autoclass:: momba.model.Network
    :members:
    :member-order: bysource
