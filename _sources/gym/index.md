# Momba Gym

This documentation describes the *Momba Gym* API which is a part of *MoGym*.

MoGym is an integrated toolbox enabling the training and verification of  machine-learned decision-making agents based on formal models. Given a formal representation of a decision-making problem in the JANI format and a reach-avoid objective, MoGym (a) enables training a decision-making agent with respect to that objective directly on the model using reinforcement learning (RL) techniques, and (b) it supports the rigorous assessment of the quality of the induced decision-making agent by means of [deep statistical model checking (DSMC)](https://link.springer.com/chapter/10.1007/978-3-030-50086-3_6). MoGym implements the standard interface for training environments established by [OpenAI Gym](https://gym.openai.com), thereby connecting to the vast body of existing work in the RL community.

A tool paper describing MoGym is currently under submission for CAV22.

The Momba Gym API has two parts:

1. It exposes an OpenAI Gym compatible training environment.
2. It exposes convenience functions for deep statistical model checking.


## Training Environment

The OpenAI Gym compatible training environment is implemented by the class {class}`~momba.gym.MombaEnv`. It is generic over an *explorer* implementing the abstract base class {class}`~momba.gym.abstract.Explorer`:

```{eval-rst}
.. autoclass:: momba.gym.env.MombaEnv
    :members:
```

A generic {class}`~momba.gym.env.MombaEnv` working with a large class of JANI models and providing various configuration options, e.g., with regard to the observation and action space, can be constructed with {func}`~momba.gym.create_generic_env`:

```{eval-rst}
.. autofunction:: momba.gym.create_generic_env
```


## Deep Statistical Model Checking

The DSMC functionality of Momba Gym is based on the statistical model checker `modes` of the [Modest Toolset](https://www.modestchecker.net) which has been extended by two new resolution strategies for nondeterminism.
Unfortunately, the Modest Toolset is not open-source and the extensions have not been integrated into the official version yet.
Binary versions of the extended version are [available here](https://github.com/udsdepend/cav22-mogym-artifact/tree/main/vendor/Modest).
Please make sure that you have the extended version in your `PATH`, otherwise the following two functions will not work.

The Momba Gym API provides two functions for (a) checking an arbitrary Python function, {func}`~momba.gym.checker.check_oracle`, and (b) for checking a [PyTorch](https://pytorch.org) neural network,{func}`~momba.gym.checker.check_nn`:

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