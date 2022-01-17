# Momba Gym

*Momba Gym* enables (a) the usage of JANI models as training environments for *Reinforcement Learning* and (b) the rigorous assessment of trained decision agents based on *Deep Statistical Model Checking*.

```{eval-rst}
.. autofunction:: momba.gym.create_generic_env

.. autoclass:: momba.gym.env.MombaEnv
    :members:
```


## Deep Statistical Model Checking

```{eval-rst}
.. autofunction:: momba.gym.checker.check_oracle

.. autofunction:: momba.gym.checker.check_nn
```


```{toctree}
:maxdepth: 2
:hidden:

abstract
generic
```