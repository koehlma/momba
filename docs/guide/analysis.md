# Model Analysis

To harvest the results of formal modeling, several state-of-the-art tools are available.

First, we import the model:

```{jupyter-execute}
from momba_guide import model

model.network
```


## JANI Export

Being based on [JANI-model](https://jani-spec.org), models can easily be exported:

```{jupyter-execute}
from momba import jani

jani.dump_model(model.network)
```

The resulting JANI model can then be fed into any compatible tool.


## Analysis APIs

Momba also provides APIs to directly interface with various tools.

```{jupyter-execute}
from momba.moml import expr, prop
from momba.tools.modest import checker as modest_checker


values = modest_checker.check(
    model.network,
    properties={
        "goal": prop(
            "min({ Pmax(F($is_finished)) | initial })",
            is_finished=model.has_finished(expr("pos_x"), model.track)
        ),
    },
)

float(values["goal"])
```

Here, we invoke `mcsta` of the [Modest Toolset](https://www.modestchecker.net/) to compute the maximal probability of winning the game on the given track, i.e., the probability to win the game if all moves taken are optimal.


```{jupyter-execute}
from momba.tools.storm_docker import checker as storm_checker


values = storm_checker.check(
    model.network,
    properties={
        "goal": prop(
            "min({ Pmax(F($is_finished)) | initial })",
            is_finished=model.has_finished(expr("pos_x"), model.track)
        ),
    },
)

print(float(values["goal"]))
```
