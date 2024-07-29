(guide)=
# User Guide

This *user guide* covers popular use cases of Momba by example.
It is meant to be read from start to finish as the sections build on top of each other.

Dealing with formal models encompasses a variety of tasks which can be challenging from time to time—especially for newcomers.
Everything starts with the *construction* of a model or a family thereof.
Often there already exists a textual or other, more formal, description of the scenario to be modeled, such as a rough sketch of the desired behavior or a circuit diagram.
Then, after a formal model has been conceived, one has to *validate* that the model actually adequately models what should be modeled.
In this regard models are just like any other human artifact, inadequate initially but over time it gets better.
Only after confidence in the model has been established, one is able to harvest the benefits by handing over the model to *analysis* tools, e.g., a model checker.
Momba strives to deliver an integrated and intuitive experience to aid the process of model construction, validation, and analysis.

In [Model Construction](construction), we explain how Momba's model construction API can be leveraged to programmatically construct models based upon existing *scenario descriptions*.
Scenario descriptions go beyond what is possible with mere parametrization and existing tools.
A scenario description may be any kind of Python object encoding details of the system to be modeled.

In [Model Exploration](exploration), we demonstrate how Momba's explicit state space exploration engine can be used to validate a model by interactive simulation or by connecting it to a testing framework.
The exploration engine can be used to harvest Python's vast ecosystem by interfacing with all kinds of libraries, for instance, for visualization.
Furthermore, it is possible to integrate logic written in Python with a model, e.g., have a controller written in Python control a modelled plant.

In [Model Analysis](analysis), we show how to invoke external model analysis tools via unified interfaces provided by Momba.
Thanks to Momba's first-class support for [JANI-model](https://jani-spec.org), several state-of-the-art analysis tools are readily available for model checking and other analysis tasks.


```{toctree}
:maxdepth: 2
:hidden:

construction
exploration
analysis
```
