JANI Value Passing
==================

This document specifies the JANI feature *x-momba-value-passing* which extends the *JANI model interchange format* (`JANI-model <http://www.jani-spec.org/>`_) with *value passing* via synchronization inspired by LOTOS.
In principle, JANI-model already allows for efficient value passing via transient global variables [BDHHJT17]_.
While perfectly adequate for model checking purposes, this approach to value passing has two shortcomings:
(1) it restricts compositionality by not allowing two instances of the same automaton to participate in a synchronization while passing values over the same global variables and
(2) it does not allow one to reason about an automaton solely on the basis of its observable behavior in terms of actions.
Thus, we extend JANI-model with a feature for explicit value passing.

This document is structured as follows: after presenting the general idea informally, we proceed by specifying the necessary syntactic extensions to the JSON-schema of JANI-model.
Subsequently, we describe the static and dynamic semantics of the extension.


General Idea
------------

The intuitive idea is to extend actions with *parameters* giving rise to *action schemas*.
An *action schema* :math:`\alpha(\tau_1, \ldots, \tau_n)` consists of a name :math:`\alpha` and a list of parameter types :math:`\tau_i`.
A concrete action :math:`\alpha(\nu_1, \ldots, \nu_n)`, then, consists of a name :math:`\alpha` uniquely identifying an action schema :math:`\alpha(\tau_1, \ldots, \tau_n)` and *arguments* :math:`\nu_i` such that each :math:`\nu_i` has a type which is assignable to :math:`\tau_i`.

An *action pattern* :math:`\alpha(x_1, \ldots, x_n)` consists of a name :math:`\alpha` uniquely identifying an action schema :math:`\alpha(\tau_1, \ldots, \tau_n)` and identifiers :math:`x_i`.
The `action` fields of an edge now becomes an action pattern and the corresponding identifiers are available in the guard and for assignments.

Each edge with an action pattern introduces a new *edge scope* in which the guard and all expressions occurring in a destination specification are evaluated.

A synchronization vector has the form :math:`\langle p_1, \ldots, p_m\rangle \to p` where all :math:`p`s are action patterns.

Generally, we allow identifiers to occur multiple times if they have the same time.
We replace all except one of the occurances with different identifiers and add an assertion that they are all equal.


Syntactical Extension
---------------------

We extend the each action declaration of the `actions` field in `Model` with an optional field `parameters`:

.. code-block:: javascript

    {
        "name": Identifier,
        "?parameters": Array.of({
            "type": Type,
            "?comment": String
        }),
        "?comment": String
    }

We extend the `action` fields of an `edge` with:

.. code-block:: javascript

    var ActionPattern = schema([
        Identifier,
        {
            "name": Identifier,
            "?identifiers": Array.of([
                Identifier
            ])
        }
    ])


We extend a synchronization vector as follows:

.. code-block:: javascript

    {
        "synchronise": Array.of([ActionPattern, null])
        "?result": ActionPattern,
        "?condition": Expression,
        "?comment": String
    }


Static Semantics
----------------

If there are no parameters in place, then we do not care.

If there are parameters, the number of arguments for an action pattern and the number of parameters of the respective action type has to be identical.

Within an edge, the following has to hold: For each argument, if the respective parameter has "READ" direction, the argument is an identifier, otherwise, it is an arbitrary expression.
Each identifier must occur at most once as "READ" direction argument.

For each synchronization vector, for each pattern in the vector: For each argument, if the respective parameter has "WRITE" direction, the argument is an identifier, otherwise, it is an arbitrary expression.
Within each synchronization vector, each identifier must occur at most once as "WRITE" direction argument.
For the `result` fields this is reversed.


Scoping: An action pattern introduces a new scope.

Within an edge: Identifiers occurring as "READ" direction arguments can be used and contain a value of the respective parameter type for expressions in destinations and the `guard` field.

Within a synchronization vector: Identifiers occurring as "WRITE" direction arguments can be used and contain a value of the respective type for expressions occurring in "READ" direction argument respectively "WRITE" direction arguments in the resulting action.

Expressions occurring in `WRITE`-parameters must not contain any sampling expressions. They may, however, contain the `nondet` operator as specified in JANI feature *nondet-selection*.


Dynamic Semantics
-----------------

As the guard may refer identifiers bound in the action pattern, we consider possible synchronization actions before evaluating the guard.



The semantics are as follows:

1. For all automata the expressions occurring in `WRITE` direction arguments are evaluated.
2. Those values are bound to the respective identifiers in the synchronization vector.
3. The condition of the synchronization is evaluated. If it evaluates to false, the respective actions are not possible, if it evaluates to true, we proceed.
4. The expressions in the `READ` direction arguments in the synchronization vector are evaluated.
5. The results of those evaluations are bound to the respective identifiers in the pattern for each automata.
6. The `guard` of each of those actions is evaluated with those values.
7. If all guards evaluate to `true` then the automata atomically take the respective edges together. The values bound to the identifiers in the automata are available for the assignments and probabilities for each destination. The assignments are resolved as usually using their indices.


.. [BDHHJT17] Carlos E. Budde, Christian Dehnert, Ernst Moritz Hahn, Arnd Hartmanns, Sebastian Junges, and Andrea Turrini:
    JANI: Quantitative Model and Tool Interaction. TACAS (2) 2017: 151-168
