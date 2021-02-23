# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2021, Saarland University
# Copyright (C) 2019-2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import enum

from . import types


_JANI_NAME_MAP: t.Dict[str, DistributionType] = {}


class DistributionType(enum.Enum):
    """
    An enum of distribution type.
    """

    DISCRETE_UNIFORM = (
        "DiscreteUniform",
        (
            types.INT,
            types.INT,
        ),
        types.INT,
    )
    """ Discrete uniform distribution. """

    BERNOULLI = "Bernoulli", (types.REAL,)
    """ Bernoulli distribution. """

    BINOMIAL = "Binomial", (types.REAL,)
    """ Binomial distribution. """

    NEGATIVE_BINOMIAL = "NegativeBinomial", (
        types.REAL,
        types.REAL,
    )
    """ Negative binomial distribution. """

    POISSON = "Poisson", (types.REAL,)
    """ Poisson distribution. """

    GEOMETRIC = "Geometric", (types.REAL,)
    """ Geometric distribution. """

    HYPERGEOMETRIC = "Hypergeometric", (
        types.INT,
        types.INT,
        types.INT,
    )
    """ Hypergeometric distribution. """

    CONWAY_MAXWELL_POISSON = "ConwayMaxwellPoisson", (
        types.REAL,
        types.REAL,
    )
    """ Conway Maxwell Poisson distribution. """

    ZIPF = "ZipF", (types.REAL,)
    """ ZipF distribution. """

    UNIFORM = "Uniform", (
        types.REAL,
        types.REAL,
    )
    """ Uniform distribution. """

    NORMAL = "Normal", (
        types.REAL,
        types.REAL,
    )
    """ Normal distribution. """

    LOG_NORMAL = "LogNormal", (
        types.REAL,
        types.REAL,
    )
    """ Logarithmic normal distribution. """

    BETA = "Beta", (
        types.REAL,
        types.REAL,
    )
    """ Beta distribution. """

    CAUCHY = "Cauchy", (
        types.REAL,
        types.REAL,
    )
    """ Cauchy distribution. """

    CHI = "Chi", (types.INT,)
    """ Chi distribution. """

    CHI_SQUARED = "ChiSquared", (types.INT,)
    """ Chi squared distribution. """

    ERLANG = "Erlang", (
        types.INT,
        types.REAL,
    )
    """ Erlang distribution. """

    EXPONENTIAL = "Exponential", (types.REAL,)
    """ Exponential distribution. """

    FISHER_SNEDECOR = "FisherSnedecor", (
        types.REAL,
        types.REAL,
    )
    """ Fisher Snedecor distribution. """

    GAMMA = "Gamma", (
        types.REAL,
        types.REAL,
    )
    """ Gamma distribution. """

    INVERSE_GAMMA = "InverseGamma", (
        types.REAL,
        types.REAL,
    )
    """ Inverse gamma distribution. """

    LAPLACE = "Laplace", (
        types.REAL,
        types.REAL,
    )
    """ Laplace distribution. """

    PARETO = "Pareto", (
        types.REAL,
        types.REAL,
    )
    """ Pareto distribution. """

    RAYLEIGH = "Rayleigh", (types.REAL,)
    """ Rayleigh distribution. """

    STABLE = "Stable", (
        types.REAL,
        types.REAL,
        types.REAL,
        types.REAL,
    )
    """ Stable distribution. """

    STUDENT_T = "StudentT", (
        types.REAL,
        types.REAL,
        types.REAL,
    )
    """ StudentT distribution. """

    WEIBULL = "Weibull", (
        types.REAL,
        types.REAL,
    )
    """ Weibull distribution. """

    TRIANGULAR = "Triangular", (
        types.REAL,
        types.REAL,
        types.REAL,
    )
    """ Triangular distribution. """

    jani_name: str
    parameter_types: t.Tuple[types.Type, ...]
    result_type: types.Type

    def __init__(
        self,
        jani_name: str,
        parameter_types: t.Tuple[types.Type, ...],
        result_type: types.Type = types.REAL,
    ) -> None:
        _JANI_NAME_MAP[jani_name] = self
        self.jani_name = jani_name
        self.parameter_types = parameter_types
        self.result_type = result_type

    @property
    def arity(self) -> int:
        return len(self.parameter_types)

    @staticmethod
    def by_name(jani_name: str) -> DistributionType:
        return _JANI_NAME_MAP[jani_name]
