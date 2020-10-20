# -*- coding:utf-8 -*-
#
# Copyright (C) 2019-2020, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import enum

from . import types


_MAP: t.Dict[str, DistributionType] = {}


class DistributionType(enum.Enum):
    DISCRETE_UNIFORM = (
        "DiscreteUniform",
        (
            types.INT,
            types.INT,
        ),
        types.INT,
    )
    BERNOULLI = "Bernoulli", (types.REAL,)
    BINOMIAL = "Binomial", (types.REAL,)
    NEGATIVE_BINOMIAL = "NegativeBinomial", (
        types.REAL,
        types.REAL,
    )
    POISSON = "Poisson", (types.REAL,)
    GEOMETRIC = "Geometric", (types.REAL,)
    HYPERGEOMETRIC = "Hypergeometric", (
        types.INT,
        types.INT,
        types.INT,
    )
    CONWAY_MAXWELL_POISSON = "ConwayMaxwellPoisson", (
        types.REAL,
        types.REAL,
    )
    ZIPF = "ZipF", (types.REAL,)
    UNIFORM = "Uniform", (
        types.REAL,
        types.REAL,
    )
    NORMAL = "Normal", (
        types.REAL,
        types.REAL,
    )
    LOG_NORMAL = "LogNormal", (
        types.REAL,
        types.REAL,
    )
    BETA = "Beta", (
        types.REAL,
        types.REAL,
    )
    CAUCHY = "Cauchy", (
        types.REAL,
        types.REAL,
    )
    CHI = "Chi", (types.INT,)
    CHI_SQUARED = "ChiSquared", (types.INT,)
    ERLANG = "Erlang", (
        types.INT,
        types.REAL,
    )
    EXPONENTIAL = "Exponential", (types.REAL,)
    FISHER_SNEDECOR = "FisherSnedecor", (
        types.REAL,
        types.REAL,
    )
    GAMMA = "Gamma", (
        types.REAL,
        types.REAL,
    )
    INVERSE_GAMMA = "InverseGamma", (
        types.REAL,
        types.REAL,
    )
    LAPLACE = "Laplace", (
        types.REAL,
        types.REAL,
    )
    PARETO = "Pareto", (
        types.REAL,
        types.REAL,
    )
    RAYLEIGH = "Rayleigh", (types.REAL,)
    STABLE = "Stable", (
        types.REAL,
        types.REAL,
        types.REAL,
        types.REAL,
    )
    STUDENT_T = "StudentT", (
        types.REAL,
        types.REAL,
        types.REAL,
    )
    WEIBULL = "Weibull", (
        types.REAL,
        types.REAL,
    )
    TRIANGULAR = "Triangular", (
        types.REAL,
        types.REAL,
        types.REAL,
    )

    jani_name: str
    parameter_types: t.Tuple[types.Type, ...]
    result_type: types.Type

    def __init__(
        self,
        jani_name: str,
        parameter_types: t.Tuple[types.Type, ...],
        result_type: types.Type = types.REAL,
    ) -> None:
        _MAP[jani_name] = self
        self.jani_name = jani_name
        self.parameter_types = parameter_types
        self.result_type = result_type

    @property
    def arity(self) -> int:
        return len(self.parameter_types)

    @staticmethod
    def by_name(jani_name: str) -> DistributionType:
        return _MAP[jani_name]
