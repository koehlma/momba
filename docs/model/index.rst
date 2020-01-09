.. _Momba models:

.. currentmodule:: momba


Momba Models
============

A *Momba model* is a network of interacting *stochastic hybrid automata* (SHA)  as per the `JANI specification`_.
SHAs are built from a finite set of *locations* connected via *edges*.
SHAs, as we understand them, are  models with *variables*.
A *valuation* maps variables to values.
Each edge leads from a single *source location* via a *guard* and an *action* to a *symbolic probability distribution* over *variable assignments* and *successor locations*.

Individual SHAs are implemented by :class:`momba.model.Automaton`.


.. _JANI specification: http://www.jani-spec.org/


.. autoclass:: momba.model.Context
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
