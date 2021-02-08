# Model Construction



```{jupyter-execute}
from momba import model

ctx = model.Context(model.ModelType.MDP)
ctx
```

XYZ

```{jupyter-execute}
ctx.create_automaton(name="die")
```

ABC

```{jupyter-execute}
from momba.moml import expr

expr("3 + 4")
```
