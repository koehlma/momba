MOML Language
=============

The *Momba Modeling Language* (MOML) is **a model description language** based on the `JANI specification <http://www.jani-spec.org/>`_.
In comparison to JANI, MOML is less verbose and thus easier to read and write.
However, MOML is still a **low-level language** forcing you to specify every detail of your model explicitly.
Depending on your use-case such fine grained control over the constructed model might not be necessary and you may want to consider using a *higher-level* language like `Modest <http://www.modestchecker.net/>`_.
If you, however, want explicit control over every aspect of your model, give MOML a try.

Momba comes with a converter to convert between MOML and JANI.
Hence, you can use your MOML model with any tool supporting the JANI specification.


.. code-block:: moml

    model_type DTMC

    variable side : int[0,7] := 0





.. toctree::
    :maxdepth: 2

    grammar


VS Code Support
---------------

We provide a `VS Code extension for MOML <https://marketplace.visualstudio.com/items?itemName=koehlma.moml-language>`_:

.. image:: vscode.jpg

