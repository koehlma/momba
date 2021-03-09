# Model Analysis

To harvest the results of formal modeling, Momba exposes unified interfaces to state-of-the-art tools.
Currently, [Storm](https://www.stormchecker.org/) and the [Modest Toolset](https://www.modestchecker.net/) are supported.
As installation can sometimes be a hassle, especially for beginners, Momba also tries its best to install these tools for you.
Note, that these tools are not distributed with Momba but downloaded on-demand.
They are not a part of Momba and distributed under their own respective licenses.
For more information, please visit the website of the respective tool and make sure your usage complies with their license.

As Momba tries to install the tools for you, it suffices to just install Momba with:
```sh
pip install momba[all]
```

For the following examples, we have to import the model defined in [Model Construction](model_construction).


```{jupyter-execute}
from momba_guide import model
```

## Model Checking

Imagine, you would like to know the maximal probability of winning the game assuming the player preforms optimal moves.
Put into a property definition, this may look as follows:

```{jupyter-execute}
from momba.moml import expr, prop

properties = {
    "goal": prop(
        "min({ Pmax(F($is_finished)) | initial })",
        is_finished=model.has_finished(expr("pos_x"), model.track)
    ),
}
```

Here, we use Momba's {func}`~momba.moml.prop` function to define the property.
Intuitively, we aggregate over all initial states by taking the minimum over all initial states of the maximal probability of winning the game from the respective initial state.
We call `has_finished` as defined for our model to build an expression being true if and only if the game has been won by moving over the finishing line.

As a unified interface to model checkers, Momba provides the abstract base class {class}`~momba.analysis.Checker`.
Let's say we would like to check the property with both, Storm and the Modest Toolset.
To this end, we import both tools and obtain a {class}`~momba.analysis.Checker` instance for each tool, respectively:

```{jupyter-execute}
from momba.tools import modest, storm

modest_checker = modest.get_checker(accept_license=True)
storm_checker = storm.get_checker(accept_license=True)
```

We have to explicitly pass `accept_license=True` to the `get_checker` functions to indicate that we have read and accept the licenses applying to these tools which are not a part of Momba.
For Storm, [Docker](https://www.docker.com/) has to be running on the system, as Momba will use the [Docker image of Storm](https://www.stormchecker.org/documentation/obtain-storm/docker.html).

Checking the defined property with both model checkers is now straightforward:

```{jupyter-execute}
results = {
    checker: checker.check(model.network, properties=properties)
    for checker in [modest_checker, storm_checker]
}

for checker, values in results.items():
    print(f"{checker.__class__.__name__}")
    for prop_name, prop_value in values.items():
        print(f"  Property {prop_name!r}: {float(prop_value)}")
```

There also exists a *cross checker* which automatically cross validates the results produced by different model checkers and raises an exception if the results do not match:

```{jupyter-execute}
from momba.analysis import checkers

cross_checker = checkers.CrossChecker([modest_checker, storm_checker])
cross_checker.check(model.network, properties=properties)
```


## JANI Export

Momba is centered around the [JANI-model](https://jani-spec.org) modeling formalism and interchange format.
Hence, models defined with Momba can easily be exported in JANI-model format:

```{jupyter-execute}
from momba import jani

jani.dump_model(model.network)
```

The resulting JANI model can then be fed in any compatible tool for further analysis.
