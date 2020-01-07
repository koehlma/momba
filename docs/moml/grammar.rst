MOML Grammar
============

This is the full abstract grammar of MOML:

.. code-block::

    <model> ::=
        | [<model-type>] <definition>*


    <model-type> ::=
        | ‘model_type’ MODEL-TYPE


    <definition> ::=
        | <variable-definition>
        | <constant-definition>
        | <action-definition>
        | <automaton-definition>
        | <system-definition>
        | <property-definition>

    <variable-definition> ::=
        | [‘transient’] ‘variable’ IDENTIFIER ‘:’ <type> [‘:=’ <expression>] [<comment>]

    <constant-definition> ::=
        | ‘constant’ IDENTIFIER ‘:’ <type> [‘:=’ <expression>] [<comment>]

    <action-definition> ::=
        | ‘action’ IDENTIFIER [<comment>]

    <automaton-definition> ::=
        | ‘automaton’ IDENTIFIER ‘:’ INDENT <automaton-specification> DEDENT

    <system-definition> ::=
        | ‘system’ [IDENTIFIER] ‘:’ INDENT <system-specification> DEDENT

    <property-definition> ::=
        | ‘property’ IDENTIFIER ‘:=’ <property-expression> [<comment>]


    <location-definition> ::=
        | [‘initial’] ‘location’ IDENTIFIER [‘:’ INDENT <location-body> DEDENT]

    <location-body> ::=
        | ‘invariant’ <expression>

    <edge-definition> ::=
        | ‘edge’ ‘from’ IDENTIFIER ‘:’ INDENT <edge-body> DEDENT

    <edge-body> ::=
        | ‘action’ IDENTIFIER
        | ‘guard’ <expression> [<comment>]
        | ‘rate’ <expression> [<comment>]
        | ‘to’ IDENTIFIER [‘:’ INDENT <destination-body> DEDENT]

    <destination-body> ::=
        | ‘probability’ <expression> [<comment>]
        | <assignment>

    <assignment> ::=
        | IDENTIFIER ‘:=’ <expression>


    <expression> ::=
        | <constant>
        | IDENTIFIER
        | IDENTIFIER ‘(’ (<expression> [,])* ‘)’
        | <UNARY-OPERATOR> <expression>
        | <expression> <BINARY-OPERATOR> <expression>
        | <expression> ? <expression> : <expression>

    <constant> ::=
        | ‘true’ | ‘false’
        | /[0-9]+/ ‘.’ /[0-9]+/ | ‘real[’ <NAMED-REAL> ‘]’
        | /[0-9]+/

    <UNARY-OPERATOR> ::=
        | ‘¬’ | ‘not’

    <BINARY-OPERATOR> ::=
        | ‘∨’ | ‘or’
        | ‘∧’ | ‘and’
        | ‘⊕’ | ‘xor’
        | ‘⇒’ | ‘==>’ | ‘implies’
        | ‘⇔’ | ‘<=>’ | ‘equiv’
        | ‘==’ | ‘!=’ | ‘=’ | ‘≠’
        | ‘<’ | ‘≤’ | ‘≥’ | ‘>’
        | ‘+’ | ‘-’ | ‘*’ | ‘%’
        | ‘/’ | ‘//’

    <property-expression> ::=
        | <expression>

    <comment> ::=
        | ‘"’ /([^"]|\")/ ‘"’


    <type> ::=
        | <primitive-type>
        | <bounded-type>
        | <array-type>

    <primitive-type> ::=
        | ‘bool’
        | <numeric-type>

    <numeric-type> ::=
        | ‘int’
        | ‘real’
        | ‘clock’
        | ‘continuous’

    <bounded-type> ::=
        | <numeric-type> ‘[’ <integer> ‘,’ <integer> ‘]’

    <array-type> ::=
        | <type> ‘[]’


If the model type is omitted the file must not contain anything else than property definitions.
This allows to separate property and model definitions.
