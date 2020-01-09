MOML Grammar
============

This is the full abstract grammar of MOML:


.. code-block:: bnf

    <model> ::=
        | [<model-type>] <specification>*


    <model-type> ::=
        | ‘model_type’ MODEL-TYPE


    <specification> ::=
        | <metadata-definition>
        | <variable-declaration>
        | <constant-declaration>
        | <action-declaration>
        | <automaton-definition>
        | <network-definition>
        | <property-definition>

    <metadata-definition> ::=
        | ‘metadata’ ‘:’ INDENT <metadata-field>* DEDENT

    <variable-declaration> ::=
        | [‘transient’] ‘variable’ IDENTIFIER ‘:’ <type> [‘:=’ <expression>] [<comment>]

    <constant-declaration> ::=
        | ‘constant’ IDENTIFIER ‘:’ <type> [‘:=’ <expression>] [<comment>]

    <action-declaration> ::=
        | ‘action’ IDENTIFIER [<comment>]

    <automaton-definition> ::=
        | ‘automaton’ IDENTIFIER ‘:’ INDENT <automaton-specification>* DEDENT

    <network-definition> ::=
        | ‘network’ [IDENTIFIER] ‘:’ INDENT <network-specification>* DEDENT

    <property-definition> ::=
        | ‘property’ IDENTIFIER ‘:=’ <property> [<comment>]


    <metadata-field> ::=
        | <string> ‘:’ <string>


    <automaton-specification> ::=
        | <variable-declaration>
        | <location-definition>
        | <edge-definition>

    <network-specification> ::=
        | <instance-definition>
        | <restrict-initial>
        | <composition>


    <location-definition> ::=
        | [‘initial’] ‘location’ IDENTIFIER [‘:’ INDENT <location-specification>* DEDENT]

    <location-specification> ::=
        | ‘invariant’ <expression>
        | <assignment>


    <edge-definition> ::=
        | ‘edge’ ‘from’ IDENTIFIER ‘:’ INDENT <edge-specification>* DEDENT

    <edge-specification> ::=
        | ‘action’ IDENTIFIER
        | ‘guard’ <expression> [<comment>]
        | ‘rate’ <expression> [<comment>]
        | ‘to’ IDENTIFIER [‘:’ INDENT <destination-specification>* DEDENT]

    <destination-specification> ::=
        | ‘probability’ <expression> [<comment>]
        | <assignment>

    <assignment> ::=
        | IDENTIFIER ‘:=’ <expression>


    <instance-definition> ::=
        | ‘instance’ IDENTIFIER IDENTIFIER [‘:’ INDENT <instance-specification>* DEDENT]

    <instance-specification> ::=
        | ‘input’ ‘enable’ IDENTIFIER [‘,’ IDENTIFIER]


    <restrict-initial> ::=
        | ‘restrict’ ‘initial’ <expression>


    <composition> ::=
        | ‘composition’ IDENTIFIER [‘|’ IDENTIFIER] [‘:’ INDENT <composition-specification>* DEDENT]

    <composition-specification> ::=
        | ‘synchronize’ <action> [‘|’ <action>] (‘->’ | ‘→’) <action>

    <action> ::=
        | IDENTIFIER
        | ‘-’ | ‘τ’


    <expression> ::=
        | <constant>
        | IDENTIFIER ‘(’ [<expression> [‘,’ <expression>]] ‘)’
        | IDENTIFIER
        | <unary-operator> <expression>
        | <expression> <binary-operator> <expression>
        | <expression> ‘?’ <expression> ‘:’ <expression>

    <constant> ::=
        | ‘true’ | ‘false’
        | /[0-9]+/ ‘.’ /[0-9]+/ | ‘real[’ <NAMED-REAL> ‘]’
        | /[0-9]+/

    <unary-operator> ::=
        | ‘¬’ | ‘not’

    <binary-operator> ::=
        | ‘∨’ | ‘or’
        | ‘∧’ | ‘and’
        | ‘⊕’ | ‘xor’
        | ‘⇒’ | ‘==>’
        | ‘⇔’ | ‘<=>’
        | ‘=’ | ‘!=’ | ‘≠’
        | ‘<’ | ‘≤’ | ‘≥’ | ‘>’
        | ‘+’ | ‘-’ | ‘*’ | ‘%’
        | ‘/’ | ‘//’


    <property> ::=
        | <expression>
        | … TODO …


    <comment> ::=
        | ‘"’ /([^"]|\")/ ‘"’


    <string> ::=
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
        | <numeric-type> ‘[’ <expression> ‘,’ <expression> ‘]’

    <array-type> ::=
        | <type> ‘[]’


If the model type is omitted the file must not contain anything else than property definitions.
This allows to separate property and model definitions.
